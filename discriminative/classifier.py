import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
# -------------------- 判别式分类器（Route 1） --------------------

class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # hidden_states: [B, L, D]; attention_mask: [B, L]
        mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        masked = hidden_states * mask
        summed = masked.sum(dim=1)  # [B, D]
        denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return summed / denom


class DiscriminativeClassifier(nn.Module):
    def __init__(self, backbone: AutoModel, hidden_size: int, num_labels: int, mlp_hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.pooler = MeanPooler()
        self.dropout = nn.Dropout(dropout)
        if mlp_hidden is None:
            self.head = nn.Linear(hidden_size, num_labels)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, num_labels)
            )

        # 冻结 backbone 参数（仅训练分类头）
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        last_hidden = outputs.last_hidden_state  # [B, L, D]
        pooled = self.pooler(last_hidden, attention_mask)  # [B, D]
        pooled = self.dropout(pooled)
        logits = self.head(pooled)  # [B, C]
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
