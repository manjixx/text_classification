#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判别式文本分类预测（CPU）
支持 DeepSeek / BERT / RoBERTa 等模型 + 自定义 MLP 分类头
"""
import os
import torch
from transformers import AutoTokenizer, AutoModel
from .classifier import DiscriminativeClassifier  # 自定义分类头


def predict_discriminative(args):
    """
    判别式文本分类预测
    args 必须包含：
        - model_name_or_path: 预训练模型路径或名称
        - ckpt: 判别式训练好的 checkpoint 文件夹
        - label_set_file: 训练时保存的 labels.txt
        - texts: 待预测文本列表
        - max_length: 最大文本长度
        - batch_size: 批量大小
        - mlp_hidden, dropout: 分类头参数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # -------------------- 加载标签 --------------------
    if not args.labels or not os.path.exists(args.labels):
        raise ValueError("必须提供 label_set_file (训练时保存的 labels.txt)")
    with open(args.labels, 'r', encoding='utf-8') as f:
        labels = [l.strip() for l in f if l.strip()]
    label2id = {lab: i for i, lab in enumerate(labels)}

    # -------------------- tokenizer + backbone --------------------
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    backbone = AutoModel.from_pretrained(args.model_name_or_path)
    hidden_size = getattr(backbone.config, "hidden_size", getattr(backbone.config, "hidden_dim", None))
    if hidden_size is None:
        raise ValueError("无法获取 hidden_size，请检查模型类型。")

    # -------------------- 构建分类模型 --------------------
    model = DiscriminativeClassifier(
        backbone,
        hidden_size,
        num_labels=len(labels),
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout
    )

    # -------------------- 加载训练权重 --------------------
    ckpt_file = os.path.join(args.ckpt, "best_route1.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # 处理可能的多GPU训练保存的state_dict
    if list(ckpt["state_dict"].keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        ckpt["state_dict"] = new_state_dict

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # -------------------- 批量预测 --------------------
    batch_size = args.batch_size or 16
    texts = list(args.texts)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tok(batch_texts, max_length=args.max_length, truncation=True, padding=True, return_tensors="pt")

        # 确保输入数据也在相同的设备上
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs = torch.softmax(out["logits"], dim=-1)
            conf, pred = torch.max(probs, dim=-1)

        for t, p, c in zip(batch_texts, pred.tolist(), conf.tolist()):
            print(f"[Pred] text={t} => label={labels[p]}  confidence={c:.4f}")