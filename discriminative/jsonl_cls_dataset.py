import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader

# -------------------- 数据集定义（判别式） --------------------

class JsonlClsDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, label2id: Dict[str, int], max_length: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text"]
        label = rec["label"]
        enc = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label2id[label], dtype=torch.long)
        return item
