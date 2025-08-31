
import torch
from typing import List, Tuple, Dict,Optional

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)

from config.generative import INSTR_SYSTEM, INSTR_TEMPLATE


def label_word_scoring(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        labels: List[str],
        texts: List[str],
        max_length: int = 512,
        device: Optional[torch.device] = None
) -> List[Tuple[str, Dict[str, float]]]:
    """对每个文本，计算每个标签词的 log 概率，返回最佳标签与得分明细。"""
    model.eval()

    # 确定设备（如果未指定，则使用模型所在的设备）
    if device is None:
        device = next(model.parameters()).device

    results = []
    with torch.no_grad():
        for text in texts:
            prompt = INSTR_TEMPLATE.format(system=INSTR_SYSTEM, labels=", ".join(labels), text=text)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)

            # 将所有张量移动到正确的设备
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            label_scores = {}
            for lab in labels:
                lab_ids = tokenizer(lab, add_special_tokens=False, return_tensors="pt")["input_ids"]
                lab_ids = lab_ids.to(device)  # 移动到正确的设备
                lab_len = lab_ids.size(1)

                # 使用原始attention_mask构建完整掩码（更高效准确）
                lab_mask = torch.ones((1, lab_len), dtype=attention_mask.dtype, device=device)
                full_attention_mask = torch.cat([attention_mask, lab_mask], dim=1)

                # 拼接输入IDs
                full_ids = torch.cat([input_ids, lab_ids], dim=1)

                # 使用正确的掩码
                out = model(input_ids=full_ids, attention_mask=full_attention_mask)
                logits = out.logits
                logprobs = torch.log_softmax(logits, dim=-1)

                # 计算标签概率
                total_logp = 0.0
                for i in range(lab_len):
                    token_id = lab_ids[0, i]
                    pos = input_ids.size(1) + i - 1
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