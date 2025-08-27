import os
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.data_utils import load_jsonl
from .jsonl_sft_dataset import JsonlSFTDataset
from .data_collator import SimpleDataCollator
from .lable_score import label_word_scoring
from peft import LoraConfig, get_peft_model, PeftModel


# -------------------- 训练循环（LoRA） --------------------

def train_lora(args):
    device = torch.device("cpu")
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

    train_recs = load_jsonl(args.train_file)
    valid_recs = load_jsonl(args.valid_file)

    train_ds = JsonlSFTDataset(train_recs, tok, labels, max_length=args.max_length)
    valid_ds = JsonlSFTDataset(valid_recs, tok, labels, max_length=args.max_length)

    collator = SimpleDataCollator(pad_token_id=tok.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

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
                chunk = valid_texts[i:i+args.eval_bs]
                scores = label_word_scoring(model, tok, labels, chunk, max_length=args.max_length)
                preds.extend([lab for lab, _ in scores])
        acc = accuracy_score(valid_labels, preds)
        f1 = f1_score(valid_labels, preds, average="macro")
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  val_acc={acc:.4f}  val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            # 只保存 LoRA 适配器权重
            model.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(labels))
            print("[Route2-LoRA] 新最佳适配器已保存")
