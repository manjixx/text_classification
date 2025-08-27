import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SimpleDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        # 手动 padding 到 batch 最长
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            L = f["input_ids"].size(0)
            pad_len = max_len - L
            input_ids.append(torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([f["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)]))
            lab = f["labels"]
            labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
        return batch
