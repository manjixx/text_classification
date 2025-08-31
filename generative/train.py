import os
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from utils.data_utils import load_jsonl
from .jsonl_sft_dataset import JsonlSFTDataset
from .data_collator import SimpleDataCollator
from .lable_score import label_word_scoring
from peft import LoraConfig, get_peft_model


# -------------------- 训练循环（LoRA） --------------------

def train_lora(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    labels = [s.strip() for s in args.labels.split(',') if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        # 对部分 CausalLM，需要将 pad_token 指向 eos_token
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # 构建 LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(',') if args.lora_targets else ["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(device)  # 确保模型在正确的设备上
    print(f"Model device: {next(model.parameters()).device}")

    train_recs = load_jsonl(args.train_file)
    valid_recs = load_jsonl(args.valid_file)
    test_recs = load_jsonl(args.test_file) if hasattr(args, 'test_file') and args.test_file else None

    train_ds = JsonlSFTDataset(train_recs, tok, labels, max_length=args.max_length)
    valid_ds = JsonlSFTDataset(valid_recs, tok, labels, max_length=args.max_length)

    collator = SimpleDataCollator(pad_token_id=tok.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.05 * total_steps),
                                            num_training_steps=total_steps)

    best_f1 = -1.0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_dl))

        # 验证：使用 label-word scoring
        model.eval()
        valid_texts = [r["text"] for r in valid_recs]
        valid_labels = [r["label"] for r in valid_recs]
        preds = []
        with torch.no_grad():
            for i in range(0, len(valid_texts), args.eval_bs):
                chunk = valid_texts[i:i + args.eval_bs]
                scores = label_word_scoring(model, tok, labels, chunk, max_length=args.max_length, device=device)
                preds.extend([lab for lab, _ in scores])
        acc = accuracy_score(valid_labels, preds)
        f1 = f1_score(valid_labels, preds, average="macro")
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  val_acc={acc:.4f}  val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            # 保存最佳模型状态
            best_model_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'labels': labels,
                'lora_config': lora_cfg
            }
            # 只保存 LoRA 适配器权重
            model.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(labels))
            print("[Route2-LoRA] 新最佳适配器已保存")

    # 测试集报告
    if test_recs:
        print("\n开始测试集评估...")

        # 确保使用与训练时相同的标签
        test_texts = [r["text"] for r in test_recs]
        test_labels = [r["label"] for r in test_recs]

        # 加载最佳模型
        model.load_state_dict(best_model_state["state_dict"])
        model.eval()

        # 进行预测
        test_preds = []
        test_confs = []
        with torch.no_grad():
            for i in range(0, len(test_texts), args.eval_bs):
                chunk = test_texts[i:i + args.eval_bs]
                scores = label_word_scoring(model, tok, labels, chunk, max_length=args.max_length, device=device)
                # 正确提取预测标签和置信度
                preds_chunk = [lab for lab, conf_map in scores]
                # 获取最高置信度
                confs_chunk = [max(conf_map.values()) for lab, conf_map in scores]
                test_preds.extend(preds_chunk)
                test_confs.extend(confs_chunk)

        # 将预测结果转换为标签名称
        pred_labels = test_preds
        gt_labels = test_labels

        # 确保分类报告使用的标签与预测一致
        unique_labels = sorted(set(gt_labels + pred_labels))

        print("\n[Test Report]\n")
        # 构建 DataFrame，每行对应一条样本
        df = pd.DataFrame({
            "ground_truth": gt_labels,
            "prediction": pred_labels,
            "confidence": test_confs
        })

        # 保存测试结果 CSV 文件（列式）
        csv_path = os.path.join(args.output_dir, "test_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"测试结果已保存至: {csv_path}")

        # 另外保存整体指标（accuracy / f1）到 JSON
        metrics = {
            "accuracy": accuracy_score(gt_labels, pred_labels),
            "macro_f1": f1_score(gt_labels, pred_labels, average="macro"),
            "weighted_f1": f1_score(gt_labels, pred_labels, average="weighted")
        }

        metrics_path = os.path.join(args.output_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"测试指标已保存至: {metrics_path}")

        # 打印详细分类报告
        from sklearn.metrics import classification_report
        print("\n详细分类报告:")
        print(classification_report(gt_labels, pred_labels, labels=unique_labels))