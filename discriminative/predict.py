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
from classifier import DiscriminativeClassifier

# -------------------- 预测（判别式） --------------------

def predict_discriminative(args):
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # 读取标签
    if args.label_set_file and os.path.exists(args.label_set_file):
        with open(args.label_set_file, 'r', encoding='utf-8') as f:
            labels = [l.strip() for l in f if l.strip()]
    else:
        raise ValueError("必须提供 label_set_file (训练时保存的 labels.txt)")
    label2id = {lab: i for i, lab in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "hidden_dim", None))

    model = DiscriminativeClassifier(backbone, hidden_size, num_labels=len(labels), mlp_hidden=args.mlp_hidden, dropout=args.dropout)
    ckpt = torch.load(os.path.join(args.ckpt, "best_route1.pt"), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # 批量编码
    enc = tok(list(args.texts), max_length=args.max_length, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])  # logits
        probs = torch.softmax(out["logits"], dim=-1)
        conf, pred = torch.max(probs, dim=-1)

    for t, p, c in zip(args.texts, pred.tolist(), conf.tolist()):
        print(f"[Pred] text={t} => label={labels[p]}  confidence={c:.4f}")

