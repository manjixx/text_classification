#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 文本场景分类（纯 CPU）—两条主路线的可运行实现：
  - Route 1: 判别式分类（DeepSeek 作为特征提取器 + 线性/MLP 头）
  - Route 2: 生成式分类（DeepSeek + LoRA 指令化微调；推理使用 label-word scoring）

依赖：
  pip install -U torch transformers datasets peft scikit-learn

数据：
  采用 JSONL，每行形如：{"text": "付款后订单未发货", "label": "物流"}

示例用法：
  # 1) 判别式训练
  python deepseek_text_classification_cpu.py \
      --route discriminative \
      --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat \
      --train_file ./data/train.jsonl \
      --valid_file ./data/valid.jsonl \
      --test_file  ./data/test.jsonl \
      --output_dir ./checkpoints/route1 \
      --max_length 256 --batch_size 16 --epochs 8 --lr 1e-3

  # 推理（判别式）
  python deepseek_text_classification_cpu.py \
      --route discriminative --predict \
      --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat \
      --ckpt ./checkpoints/route1 \
      --label_set_file ./checkpoints/route1/labels.txt \
      --texts "无法登录账户" "申请退款不到账"

  # 2) LoRA 指令化微调
  python deepseek_text_classification_cpu.py \
      --route generative \
      --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat \
      --train_file ./data/train.jsonl \
      --valid_file ./data/valid.jsonl \
      --output_dir ./checkpoints/route2_lora \
      --labels "旅行,支付,物流,售后,账户,其他" \
      --max_length 512 --batch_size 4 --epochs 3 --lr 8e-5

  # 推理（LoRA + label-word scoring）
  python deepseek_text_classification_cpu.py \
      --route generative --predict \
      --model_name_or_path deepseek-ai/deepseek-llm-7b-lite-chat \
      --ckpt ./checkpoints/route2_lora \
      --labels "旅行,支付,物流,售后,账户,其他" \
      --texts "无法登录账户" "申请退款不到账"

注意：请优先选择小参数 DeepSeek 模型（1B~3B 量级）以适配 CPU；上方模型名称仅示例。
"""
import argparse
from utils.model_utils import set_seed
from discriminative.predict import predict_discriminative
from discriminative.train import train_discriminative
from generative.predict import predict_lora
from generative.train import train_lora
# -------------------- 主函数与参数 --------------------

def build_argparser():
    p = argparse.ArgumentParser(description="DeepSeek 文本分类（CPU）：判别式 + LoRA")
    p.add_argument('--route', choices=['discriminative', 'generative'], required=True, help='选择路线：判别式或 LoRA')
    p.add_argument('--model_name_or_path', type=str, required=True, help='预训练 DeepSeek 模型名称或本地路径')

    # 通用
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--mlp_hidden', type=int, default=None)

    # 判别式/LoRA 训练数据
    p.add_argument('--train_file', type=str)
    p.add_argument('--valid_file', type=str)
    p.add_argument('--test_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='./checkpoints')

    # 预测
    p.add_argument('--predict', action='store_true', help='是否执行预测')
    p.add_argument('--texts', nargs='+', default=[], help='预测输入文本')
    p.add_argument('--ckpt', type=str, help='已训练模型/适配器目录')
    p.add_argument('--label_set_file', type=str, default=None, help='判别式预测时的标签文件')

    # LoRA 专属
    p.add_argument('--labels', type=str, default='旅行,支付,物流,售后,账户,其他')
    p.add_argument('--lora_rank', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_targets', type=str, default='q_proj,v_proj')
    p.add_argument('--eval_bs', type=int, default=16)

    return p


def main():
    set_seed(42)
    args = build_argparser().parse_args()

    if args.route == 'discriminative':
        if args.predict:
            predict_discriminative(args)
        else:
            assert args.train_file and args.valid_file and args.output_dir, "判别式训练需要 train/valid/output_dir"
            train_discriminative(args)
    elif args.route == 'generative':
        if args.predict:
            assert args.ckpt and args.labels, "LoRA 预测需要 ckpt 和 labels"
            predict_lora(args)
        else:
            assert args.train_file and args.valid_file and args.output_dir, "LoRA 训练需要 train/valid/output_dir"
            train_lora(args)


if __name__ == '__main__':
    main()
