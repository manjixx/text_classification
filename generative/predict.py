import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report

from peft import LoraConfig, get_peft_model, PeftModel
from config.generative import INSTR_SYSTEM, INSTR_TEMPLATE


def predict_lora(args):
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, args.ckpt)
    model.eval()

    results = label_word_scoring(
        model,
        tok,
        list(labels),
        list(args.texts),
        max_length=args.max_length
    )

    for t, (lab, conf_map) in zip(args.texts, results):
        print(f"[Pred] text={t} => label={lab}  confidence={conf_map.get(lab, 0.0):.4f}")
        # 可选：打印 Top-K
        topk = sorted(conf_map.items(), key=lambda x: x[1], reverse=True)
        print("  topk:", ", ".join([f"{k}:{v:.3f}" for k, v in topk]))

