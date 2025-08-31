import torch
from typing import List, Dict, Any, Optional, Tuple
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
import config.generative as config

class JsonlSFTDataset(Dataset):
    """LoRA 监督微调：把分类任务转成指令+单标签输出的生成任务。"""
    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, labels: List[str], max_length: int = 512):
        self.records = records
        self.tok = tokenizer
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text"]
        label = rec["label"]
        prompt = config.INSTR_TEMPLATE.format(system=config.INSTR_SYSTEM, labels=", ".join(self.labels), text=text)
        target = label  # 仅输出标签词
        full = prompt + " " + target
        enc = self.tok(full, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # 构造 labels：仅对 target 段计算损失。简化做法：除了最后 len(target_tokens) 外全部置 -100
        # 注意：中文可能被分成多 token，这里按长度截尾标注。
        tgt_ids = self.tok(target, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)
        n = input_ids.size(0)
        m = tgt_ids.size(0)
        start = max(0, n - m)
        labels[start:] = input_ids[start:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



