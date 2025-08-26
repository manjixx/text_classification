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

def label_word_scoring(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    labels: List[str],
    texts: List[str],
    max_length: int = 512,
) -> List[Tuple[str, Dict[str, float]]]:
    """对每个文本，计算每个标签词的 log 概率，返回最佳标签与得分明细。CPU 友好：逐文本评估。"""
    model.eval()
    results = []
    with torch.no_grad():
        for text in texts:
            prompt = INSTR_TEMPLATE.format(system=INSTR_SYSTEM, labels=", ".join(labels), text=text)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 取最后一个位置的 logits，或对整个标签序列进行条件概率评分（更稳健）
            # 这里采用序列评分：P(label_tokens | prompt)
            label_scores = {}
            for lab in labels:
                lab_ids = tokenizer(lab, add_special_tokens=False, return_tensors="pt")["input_ids"]
                # 把 prompt + label 拼接，再前移对齐，累加 label 段的 log prob
                full_ids = torch.cat([input_ids, lab_ids], dim=1)
                attn = torch.ones_like(full_ids)
                out = model(input_ids=full_ids, attention_mask=attn)
                logits = out.logits  # [1, L, V]
                logprobs = torch.log_softmax(logits, dim=-1)
                # 标签 tokens 位于最后 lab_len 位置，逐 token 累计对齐的条件概率
                lab_len = lab_ids.size(1)
                total_logp = 0.0
                for i in range(lab_len):
                    token_id = lab_ids[0, i]
                    # 预测位置是 prompt+i 的下一个 token
                    pos = input_ids.size(1) + i - 1
                    # 防止越界
                    pos = max(0, pos)
                    total_logp += float(logprobs[0, pos, token_id])
                label_scores[lab] = total_logp
            # 归一化为 softmax 置信度
            max_lab = max(label_scores, key=label_scores.get)
            # softmax for confidence
            vals = torch.tensor(list(label_scores.values()))
            probs = torch.softmax(vals, dim=0).tolist()
            conf_map = {lab: p for lab, p in zip(label_scores.keys(), probs)}
            results.append((max_lab, conf_map))
    return results



def predict_lora(args):
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
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

