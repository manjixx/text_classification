import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from jsonl_cls_dataset import JsonlClsDataset
from classifier import DiscriminativeClassifier
from utils.data_utils import load_jsonl

# -------------------- 训练循环（判别式） --------------------

def train_discriminative(args):
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # 一些 DeepSeek/LLM 可能没有 AutoModel 池化输出，选择 AutoModel（非 LM 头）提特征更轻量
    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "hidden_dim", None))
    if hidden_size is None:
        raise ValueError("无法从模型配置中获取 hidden_size，请检查模型是否为 Transformer 编码器/解码器并包含隐藏维度。")

    # 标签集：从训练与验证集中收集
    train_recs = load_jsonl(args.train_file)
    valid_recs = load_jsonl(args.valid_file)
    test_recs = load_jsonl(args.test_file) if args.test_file else []

    labels = sorted(list({r["label"] for r in train_recs + valid_recs + test_recs}))
    label2id = {lab: i for i, lab in enumerate(labels)}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    train_ds = JsonlClsDataset(train_recs, tok, label2id, args.max_length)
    valid_ds = JsonlClsDataset(valid_recs, tok, label2id, args.max_length)

    collator = DataCollatorWithPadding(tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = DiscriminativeClassifier(backbone, hidden_size, num_labels=len(labels), mlp_hidden=args.mlp_hidden, dropout=args.dropout)
    model.to(device)

    # 仅优化分类头参数
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

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

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "state_dict": model.state_dict(),
                "labels": labels,
                "model_name_or_path": args.model_name_or_path,
                "config": {
                    "hidden_size": hidden_size,
                    "mlp_hidden": args.mlp_hidden,
                    "dropout": args.dropout,
                    "max_length": args.max_length,
                }
            }, os.path.join(args.output_dir, "best_route1.pt"))
            print("[Route1] 新最佳模型已保存")

    # 可选：测试集报告
    if test_recs:
        test_ds = JsonlClsDataset(test_recs, tok, label2id, args.max_length)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        ckpt = torch.load(os.path.join(args.output_dir, "best_route1.pt"), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
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
        print("\n[Test Report]\n", classification_report(gts, preds, target_names=labels, digits=4))
