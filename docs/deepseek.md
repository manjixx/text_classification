我理解您希望将代码拆分成模块化结构并添加关键注释。以下是拆分后的代码，按照功能模块进行组织，并添加了详细的中文注释：

## 模块 1: data_utils.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具模块
包含数据加载、数据集定义和数据整理器
"""

import json
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL格式的数据文件
    
    Args:
        path: JSONL文件路径
        
    Returns:
        包含所有记录的列表
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


class JsonlClsDataset(Dataset):
    """
    判别式分类任务的数据集类
    用于路线1: 特征提取 + 线性/MLP头
    """
    
    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, 
                 label2id: Dict[str, int], max_length: int = 256):
        """
        初始化数据集
        
        Args:
            records: 数据记录列表
            tokenizer: 分词器
            label2id: 标签到ID的映射字典
            max_length: 最大序列长度
        """
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
        # 对文本进行编码
        enc = self.tokenizer(text, max_length=self.max_length, truncation=True, 
                            padding=False, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.label2id[label], dtype=torch.long)
        return item


class JsonlSFTDataset(Dataset):
    """
    LoRA监督微调任务的数据集类
    用于路线2: DeepSeek + LoRA指令化微调
    """
    
    # 指令模板常量
    INSTR_SYSTEM = "你是一个场景分类助手。只输出一个标签，不要解释。允许的标签集合见 [Labels]。"
    INSTR_TEMPLATE = (
        "[System] {system}\n[Labels] {labels}\n[User] {text}\n[Output] 请选择一个标签："
    )
    
    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, 
                 labels: List[str], max_length: int = 512):
        """
        初始化数据集
        
        Args:
            records: 数据记录列表
            tokenizer: 分词器
            labels: 标签列表
            max_length: 最大序列长度
        """
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
        
        # 构建指令提示
        prompt = self.INSTR_TEMPLATE.format(
            system=self.INSTR_SYSTEM, 
            labels=", ".join(self.labels), 
            text=text
        )
        target = label  # 仅输出标签词
        full = prompt + " " + target
        
        # 对完整文本进行编码
        enc = self.tok(full, max_length=self.max_length, truncation=True, 
                      padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # 构造标签: 仅对目标段计算损失
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


class SimpleDataCollator:
    """
    简单的数据整理器，用于LoRA训练
    手动将批次内的样本填充到相同长度
    """
    
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        # 找到批次中最长的序列长度
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention_mask, labels = [], [], []
        
        # 对每个样本进行填充
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
```

## 模块 2: model_utils.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具模块
包含模型定义、LoRA配置和标签词评分函数
"""

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Tuple, Dict, Any, Optional


class MeanPooler(nn.Module):
    """
    均值池化层
    用于将变长的序列表示转换为定长的句子表示
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态张量 [B, L, D]
            attention_mask: 注意力掩码 [B, L]
            
        Returns:
            池化后的句子表示 [B, D]
        """
        # 扩展掩码维度以匹配隐藏状态
        mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        # 应用掩码
        masked = hidden_states * mask
        # 求和
        summed = masked.sum(dim=1)  # [B, D]
        # 计算有效长度（避免除以零）
        denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        # 返回均值池化结果
        return summed / denom


class DiscriminativeClassifier(nn.Module):
    """
    判别式分类器
    用于路线1: 特征提取 + 线性/MLP头
    """
    
    def __init__(self, backbone: AutoModel, hidden_size: int, num_labels: int, 
                 mlp_hidden: Optional[int] = None, dropout: float = 0.1):
        """
        初始化分类器
        
        Args:
            backbone: 预训练模型 backbone
            hidden_size: 隐藏层大小
            num_labels: 标签数量
            mlp_hidden: MLP隐藏层大小（可选）
            dropout: Dropout率
        """
        super().__init__()
        self.backbone = backbone
        self.pooler = MeanPooler()
        self.dropout = nn.Dropout(dropout)
        
        # 构建分类头
        if mlp_hidden is None:
            self.head = nn.Linear(hidden_size, num_labels)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden), 
                nn.ReLU(), 
                nn.Dropout(dropout), 
                nn.Linear(mlp_hidden, num_labels)
            )

        # 冻结 backbone 参数（仅训练分类头）
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            labels: 标签（可选，用于训练）
            
        Returns:
            包含损失和logits的字典
        """
        # 通过backbone获取隐藏状态
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, 
                               output_hidden_states=False)
        last_hidden = outputs.last_hidden_state  # [B, L, D]
        
        # 池化获取句子表示
        pooled = self.pooler(last_hidden, attention_mask)  # [B, D]
        pooled = self.dropout(pooled)
        
        # 通过分类头获取logits
        logits = self.head(pooled)  # [B, C]
        
        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            
        return {"loss": loss, "logits": logits}


def label_word_scoring(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    labels: List[str],
    texts: List[str],
    max_length: int = 512,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    标签词评分函数
    对每个文本，计算每个标签词的log概率，返回最佳标签与得分明细
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        labels: 标签列表
        texts: 待分类文本列表
        max_length: 最大序列长度
        
    Returns:
        每个文本的最佳标签及其所有标签的置信度映射
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for text in texts:
            # 构建提示
            prompt = JsonlSFTDataset.INSTR_TEMPLATE.format(
                system=JsonlSFTDataset.INSTR_SYSTEM, 
                labels=", ".join(labels), 
                text=text
            )
            
            # 编码提示
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            
            # 获取模型输出
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 对每个标签计算条件概率
            label_scores = {}
            for lab in labels:
                # 编码标签
                lab_ids = tokenizer(lab, add_special_tokens=False, return_tensors="pt")["input_ids"]
                
                # 拼接提示和标签
                full_ids = torch.cat([input_ids, lab_ids], dim=1)
                attn = torch.ones_like(full_ids)
                
                # 获取完整序列的logits
                out = model(input_ids=full_ids, attention_mask=attn)
                logits = out.logits  # [1, L, V]
                
                # 计算log概率
                logprobs = torch.log_softmax(logits, dim=-1)
                
                # 计算标签序列的条件概率
                lab_len = lab_ids.size(1)
                total_logp = 0.0
                for i in range(lab_len):
                    token_id = lab_ids[0, i]
                    # 预测位置是prompt+i的下一个token
                    pos = input_ids.size(1) + i - 1
                    pos = max(0, pos)  # 防止越界
                    total_logp += float(logprobs[0, pos, token_id])
                    
                label_scores[lab] = total_logp
                
            # 找到最佳标签
            max_lab = max(label_scores, key=label_scores.get)
            
            # 将log概率转换为softmax置信度
            vals = torch.tensor(list(label_scores.values()))
            probs = torch.softmax(vals, dim=0).tolist()
            conf_map = {lab: p for lab, p in zip(label_scores.keys(), probs)}
            
            results.append((max_lab, conf_map))
            
    return results
```

## 模块 3: train_discriminative.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判别式分类训练模块
用于路线1: 特征提取 + 线性/MLP头
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_utils import load_jsonl, JsonlClsDataset
from model_utils import DiscriminativeClassifier


def train_discriminative(args):
    """
    训练判别式分类器
    
    Args:
        args: 命令行参数
    """
    # 设置设备为CPU并优化线程数
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # 加载分词器和模型
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    
    # 获取隐藏层大小
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "hidden_dim", None))
    if hidden_size is None:
        raise ValueError("无法从模型配置中获取 hidden_size")

    # 加载数据并构建标签映射
    train_recs = load_jsonl(args.train_file)
    valid_recs = load_jsonl(args.valid_file)
    test_recs = load_jsonl(args.test_file) if args.test_file else []

    labels = sorted(list({r["label"] for r in train_recs + valid_recs + test_recs}))
    label2id = {lab: i for i, lab in enumerate(labels)}

    # 创建输出目录并保存标签
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    # 创建数据集和数据加载器
    train_ds = JsonlClsDataset(train_recs, tok, label2id, args.max_length)
    valid_ds = JsonlClsDataset(valid_recs, tok, label2id, args.max_length)

    collator = DataCollatorWithPadding(tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 初始化模型
    model = DiscriminativeClassifier(backbone, hidden_size, num_labels=len(labels), 
                                    mlp_hidden=args.mlp_hidden, dropout=args.dropout)
    model.to(device)

    # 设置优化器和学习率调度器
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.05 * total_steps), 
                                           num_training_steps=total_steps)

    # 训练循环
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        # 训练批次
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            
            # 反向传播和优化
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
                
        # 计算指标
        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="macro")
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  val_acc={acc:.4f}  val_macro_f1={f1:.4f}")

        # 保存最佳模型
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

    # 测试集评估（可选）
    if test_recs:
        test_ds = JsonlClsDataset(test_recs, tok, label2id, args.max_length)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        
        # 加载最佳模型
        ckpt = torch.load(os.path.join(args.output_dir, "best_route1.pt"), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        # 进行预测
        preds, gts = [], []
        with torch.no_grad():
            for batch in test_dl:
                labels_t = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                pred = out["logits"].argmax(dim=-1).cpu().tolist()
                preds.extend(pred)
                gts.extend(labels_t.tolist())
                
        # 输出分类报告
        print("\n[Test Report]\n", classification_report(gts, preds, target_names=labels, digits=4))


def predict_discriminative(args):
    """
    使用判别式分类器进行预测
    
    Args:
        args: 命令行参数
    """
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # 读取标签
    if args.label_set_file and os.path.exists(args.label_set_file):
        with open(args.label_set_file, 'r', encoding='utf-8') as f:
            labels = [l.strip() for l in f if l.strip()]
    else:
        raise ValueError("必须提供 label_set_file (训练时保存的 labels.txt)")
        
    label2id = {lab: i for i, lab in enumerate(labels)}

    # 加载分词器和模型
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "hidden_dim", None))

    # 初始化模型并加载权重
    model = DiscriminativeClassifier(backbone, hidden_size, num_labels=len(labels), 
                                    mlp_hidden=args.mlp_hidden, dropout=args.dropout)
    ckpt = torch.load(os.path.join(args.ckpt, "best_route1.pt"), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # 对输入文本进行编码和预测
    enc = tok(list(args.texts), max_length=args.max_length, truncation=True, 
             padding=True, return_tensors="pt")
    
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        probs = torch.softmax(out["logits"], dim=-1)
        conf, pred = torch.max(probs, dim=-1)

    # 输出预测结果
    for t, p, c in zip(args.texts, pred.tolist(), conf.tolist()):
        print(f"[Pred] text={t} => label={labels[p]}  confidence={c:.4f}")
```

## 模块 4: train_lora.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA训练模块
用于路线2: DeepSeek + LoRA指令化微调
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_jsonl, JsonlSFTDataset, SimpleDataCollator
from model_utils import label_word_scoring
from peft import LoraConfig, get_peft_model, PeftModel


def train_lora(args):
    """
    训练LoRA适配器
    
    Args:
        args: 命令行参数
    """
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # 解析标签
    labels = [s.strip() for s in args.labels.split(',') if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存标签
    with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    # 加载分词器和模型
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        # 对部分CausalLM，需要将pad_token指向eos_token
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # 配置LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(',') if args.lora_targets else ["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 应用LoRA配置
    model = get_peft_model(base_model, lora_cfg)

    # 加载数据
    train_recs = load_jsonl(args.train_file)
    valid_recs = load_jsonl(args.valid_file)

    # 创建数据集和数据加载器
    train_ds = JsonlSFTDataset(train_recs, tok, labels, max_length=args.max_length)
    valid_ds = JsonlSFTDataset(valid_recs, tok, labels, max_length=args.max_length)

    collator = SimpleDataCollator(pad_token_id=tok.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # 设置优化器和学习率调度器
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.05 * total_steps), 
                                           num_training_steps=total_steps)

    # 训练循环
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        # 训练批次
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            
            # 反向传播和优化
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()
            
            total_loss += float(loss.item())
            
        avg_loss = total_loss / max(1, len(train_dl))

        # 验证：使用label-word scoring
        model.eval()
        valid_texts = [r["text"] for r in valid_recs]
        valid_labels = [r["label"] for r in valid_recs]
        preds = []
        
        with torch.no_grad():
            # 分批处理验证集
            for i in range(0, len(valid_texts), args.eval_bs):
                chunk = valid_texts[i:i+args.eval_bs]
                scores = label_word_scoring(model, tok, labels, chunk, max_length=args.max_length)
                preds.extend([lab for lab, _ in scores])
                
        # 计算指标
        acc = accuracy_score(valid_labels, preds)
        f1 = f1_score(valid_labels, preds, average="macro")
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  val_acc={acc:.4f}  val_macro_f1={f1:.4f}")

        # 保存最佳适配器
        if f1 > best_f1:
            best_f1 = f1
            # 只保存LoRA适配器权重
            model.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(labels))
            print("[Route2-LoRA] 新最佳适配器已保存")


def predict_lora(args):
    """
    使用LoRA适配器进行预测
    
    Args:
        args: 命令行参数
    """
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # 解析标签
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    
    # 加载基础模型和适配器
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, args.ckpt)
    model.eval()

    # 使用label-word scoring进行预测
    results = label_word_scoring(model, tok, labels, list(args.texts), max_length=args.max_length)
    
    # 输出预测结果
    for t, (lab, conf_map) in zip(args.texts, results):
        print(f"[Pred] text={t} => label={lab}  confidence={conf_map.get(lab, 0.0):.4f}")
        # 可选：打印Top-K
        topk = sorted(conf_map.items(), key=lambda x: x[1], reverse=True)
        print("  topk:", ", ".join([f"{k}:{v:.3f}" for k, v in topk]))
```

## 模块 5: main.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口模块
解析命令行参数并调用相应的训练或预测函数
"""

import argparse
import random
import torch

from train_discriminative import train_discriminative, predict_discriminative
from train_lora import train_lora, predict_lora


def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser():
    """
    构建命令行参数解析器
    
    Returns:
        参数解析器
    """
    p = argparse.ArgumentParser(description="DeepSeek 文本分类（CPU）：判别式 + LoRA")
    p.add_argument('--route', choices=['discriminative', 'generative'], required=True,
                  help='选择路线：判别式或 LoRA')
    p.add_argument('--model_name_or_path', type=str, required=True, 
                  help='预训练 DeepSeek 模型名称或本地路径')

    # 通用参数
    p.add_argument('--max_length', type=int, default=256, 
                  help='最大序列长度')
    p.add_argument('--batch_size', type=int, default=16, 
                  help='批次大小')
    p.add_argument('--epochs', type=int, default=8, 
                  help='训练轮数')
    p.add_argument('--lr', type=float, default=1e-3, 
                  help='学习率')
    p.add_argument('--weight_decay', type=float, default=1e-2, 
                  help='权重衰减')
    p.add_argument('--dropout', type=float, default=0.1, 
                  help='Dropout率')
    p.add_argument('--mlp_hidden', type=int, default=None, 
                  help='MLP隐藏层大小（可选）')

    # 数据参数
    p.add_argument('--train_file', type=str, 
                  help='训练数据文件路径')
    p.add_argument('--valid_file', type=str, 
                  help='验证数据文件路径')
    p.add_argument('--test_file', type=str, default=None, 
                  help='测试数据文件路径（可选）')
    p.add_argument('--output_dir', type=str, default='./checkpoints', 
                  help='输出目录')

    # 预测参数
    p.add_argument('--predict', action='store_true', 
                  help='是否执行预测')
    p.add_argument('--texts', nargs='+', default=[], 
                  help='预测输入文本')
    p.add_argument('--ckpt', type=str, 
                  help='已训练模型/适配器目录')
    p.add_argument('--label_set_file', type=str, default=None, 
                  help='判别式预测时的标签文件')

    # LoRA专属参数
    p.add_argument('--labels', type=str, default='旅行,支付,物流,售后,账户,其他', 
                  help='标签列表（逗号分隔）')
    p.add_argument('--lora_rank', type=int, default=8, 
                  help='LoRA秩')
    p.add_argument('--lora_alpha', type=int, default=16, 
                  help='LoRA alpha参数')
    p.add_argument('--lora_dropout', type=float, default=0.05, 
                  help='LoRA dropout率')
    p.add_argument('--lora_targets', type=str, default='q_proj,v_proj', 
                  help='LoRA目标模块（逗号分隔）')
    p.add_argument('--eval_bs', type=int, default=16, 
                  help='评估批次大小')

    return p


def main():
    """
    主函数
    """
    # 设置随机种子
    set_seed(42)
    
    # 解析命令行参数
    args = build_argparser().parse_args()

    # 根据路线选择执行相应的训练或预测
    if args.route == 'discriminative':
        if args.predict:
            predict_discriminative(args)
        else:
            assert args.train_file and args.valid_file and args.output_dir, \
                   "判别式训练需要 train/valid/output_dir"
            train_discriminative(args)
    elif args.route == 'generative':
        if args.predict:
            assert args.ckpt and args.labels, "LoRA 预测需要 ckpt 和 labels"
            predict_lora(args)
        else:
            assert args.train_file and args.valid_file and args.output_dir, \
                   "LoRA 训练需要 train/valid/output_dir"
            train_lora(args)


if __name__ == '__main__':
    main()
```

## 使用说明

1. 将上述代码保存到相应的文件中：
   - `data_utils.py`
   - `model_utils.py`
   - `train_discriminative.py`
   - `train_lora.py`
   - `main.py`

2. 安装必要的依赖：
   ```bash
   pip install torch transformers datasets peft scikit-learn
   ```

3. 运行相应的命令：
   ```bash
   # 判别式训练
   python main.py --route discriminative --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat --train_file ./data/train.jsonl --valid_file ./data/valid.jsonl --output_dir ./checkpoints/route1
   
   # 判别式预测
   python main.py --route discriminative --predict --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat --ckpt ./checkpoints/route1 --label_set_file ./checkpoints/route1/labels.txt --texts "无法登录账户" "申请退款不到账"
   
   # LoRA训练
   python main.py --route lora --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat --train_file ./data/train.jsonl --valid_file ./data/valid.jsonl --output_dir ./checkpoints/route2_lora --labels "旅行,支付,物流,售后,账户,其他"
   
   # LoRA预测
   python main.py --route lora --predict --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat --ckpt ./checkpoints/route2_lora --labels "旅行,支付,物流,售后,账户,其他" --texts "无法登录账户" "申请退款不到账"
   ```

这个模块化的代码结构使得每个部分的功能更加清晰，便于维护和扩展。每个模块都有详细的注释，解释了其功能和实现细节。
