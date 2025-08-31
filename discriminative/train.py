import os
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 自定义模块
from .jsonl_cls_dataset import JsonlClsDataset      # 处理 JSONL 格式的分类数据
from .classifier import DiscriminativeClassifier    # 判别式分类头模型
from utils.data_utils import load_jsonl             # 从 JSONL 文件加载数据

def train_discriminative(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # tokenizer & backbone
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "hidden_dim", None))
    if hidden_size is None:
        raise ValueError("无法从模型配置中获取 hidden_size，请检查模型")

    # 加载并处理数据
    train_recs = load_jsonl(args.train_file)
    valid_recs = load_jsonl(args.valid_file)
    test_recs = load_jsonl(args.test_file) if args.test_file else []

    # 确保所有数据集使用相同的标签映射
    all_labels = sorted(list({r["label"] for r in train_recs + valid_recs + test_recs}))
    label2id = {lab: i for i, lab in enumerate(all_labels)}
    id2label = {i: lab for i, lab in enumerate(all_labels)}

    # 打印标签统计信息
    print(f"总标签数量: {len(all_labels)}")
    print(f"训练集标签数量: {len(set(r['label'] for r in train_recs))}")
    print(f"验证集标签数量: {len(set(r['label'] for r in valid_recs))}")
    if test_recs:
        print(f"测试集标签数量: {len(set(r['label'] for r in test_recs))}")

    # 保存标签映射
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_labels))

    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    # Dataset & DataLoader
    train_ds = JsonlClsDataset(train_recs, tok, label2id, args.max_length)
    valid_ds = JsonlClsDataset(valid_recs, tok, label2id, args.max_length)
    collator = DataCollatorWithPadding(tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 模型
    model = DiscriminativeClassifier(backbone, hidden_size, num_labels=len(all_labels),
                                     mlp_hidden=args.mlp_hidden, dropout=args.dropout)
    model.to(device)

    # 优化器 & Scheduler
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps=int(0.05 * total_steps),
                                            num_training_steps=total_steps)

    # 训练循环
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_dl))

        # 验证
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in valid_dl:
                labels_t = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                logits = out["logits"].cpu()
                pred = logits.argmax(dim=-1)
                preds.extend(pred.tolist())
                gts.extend(labels_t.tolist())
        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="macro")
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  val_acc={acc:.4f}  val_macro_f1={f1:.4f}")

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "state_dict": model.state_dict(),
                "labels": all_labels,
                "label2id": label2id,
                "id2label": id2label,
                "model_name_or_path": args.model_name_or_path,
                "config": {
                    "hidden_size": hidden_size,
                    "mlp_hidden": args.mlp_hidden,
                    "dropout": args.dropout,
                    "max_length": args.max_length,
                }
            }, os.path.join(args.output_dir, "best_route1.pt"))
            print("[Route1] 新最佳模型已保存")

    # 测试集报告
    if test_recs:
        # 确保使用与训练时相同的标签映射
        test_ds = JsonlClsDataset(test_recs, tok, label2id, args.max_length)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

        # 加载最佳模型
        ckpt = torch.load(os.path.join(args.output_dir, "best_route1.pt"), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])

        # 确保使用模型保存时的标签映射
        id2label = ckpt["id2label"]

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in test_dl:
                labels_t = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                pred = out["logits"].argmax(dim=-1).cpu().tolist()
                preds.extend(pred)
                gts.extend(labels_t.tolist())

        # 将预测的ID转换回标签名称
        pred_labels = [id2label[p] for p in preds]
        gt_labels = [id2label[g] for g in gts]

        # 确保分类报告使用的标签与预测一致
        unique_labels = sorted(set(gt_labels + pred_labels))

        print("\n[Test Report]\n")
        # 构建 DataFrame，每行对应一条样本
        df = pd.DataFrame({
            "ground_truth": gt_labels,
            "prediction": pred_labels
        })

        # 保存测试结果 CSV 文件（列式）
        csv_path = os.path.join(args.output_dir, "test_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 另外保存整体指标（accuracy / f1）到 JSON
        metrics = {
            "accuracy": accuracy_score(gt_labels, pred_labels),
            "macro_f1": f1_score(gt_labels, pred_labels, average="macro"),
            "weighted_f1": f1_score(gt_labels, pred_labels, average="weighted")
        }

        metrics_path = os.path.join(args.output_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)