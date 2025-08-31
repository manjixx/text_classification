#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from utils.model_utils import set_seed

# 判别式
from discriminative.train import train_discriminative
from discriminative.predict import predict_discriminative

# 生成式 LoRA
from generative.train import train_lora
from generative.predict import predict_lora


def build_argparser():
    p = argparse.ArgumentParser(description="文本分类训练（CPU）")

    # 路线
    p.add_argument('--route', choices=['discriminative', 'generative'], required=True)
    p.add_argument('--model_type', choices=['bert-base', 'roberta-wwm', 'deepseek-1.5b'], required=True)
    p.add_argument('--model_name_or_path', type=str, required=True)

    # 数据文件
    p.add_argument('--train_file', type=str)
    p.add_argument('--valid_file', type=str)
    p.add_argument('--test_file', type=str)
    p.add_argument('--output_dir', type=str, default='./checkpoints')

    # 超参数
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-5)

    # 预测
    p.add_argument('--predict', action='store_true')
    p.add_argument('--texts', nargs='+', default=[])
    p.add_argument('--ckpt', type=str)
    p.add_argument('--labels', type=str, default="旅行,支付,物流,售后,账户,其他")
    p.add_argument("--mlp_hidden", type=int, default=256,
                   help="Hidden size for MLP classifier in discriminative model")
    p.add_argument("--dropout", type=float, default=0.1, help="MLP dropout rate")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")

    # LoRA 专属
    p.add_argument('--lora_rank', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_targets', type=str, default='q_proj,v_proj')
    p.add_argument('--eval_bs', type=int, default=8)

    return p


def main():
    set_seed(42)
    args = build_argparser().parse_args()

    if args.route == 'discriminative':
        # 判别式训练/预测
        if args.predict:
            predict_discriminative(args)
        else:
            assert args.train_file and args.valid_file
            train_discriminative(args)

    elif args.route == 'generative':
        # LoRA 微调 DeepSeek 1.5B
        if args.predict:
            assert args.ckpt and args.labels
            predict_lora(args)
        else:
            assert args.train_file and args.valid_file
            train_lora(args)


if __name__ == "__main__":

    '''
    判别式模型训练指令：
    python main.py --route discriminative --model_type bert-base --model_name_or_path bert-base-chinese --train_file ./data/train.json --valid_file ./data/valid.json --output_dir ./checkpoints/bert_base --max_length 256 --batch_size 16 --epochs 3 --lr 2e-5
    
    判别式模型预测指令：
    python main.py --route discriminative --model_type bert-base --model_name_or_path bert-base-chinese --predict --ckpt ./checkpoints/bert_base/best_model.pt --labels "旅行,支付,物流,售后,账户,其他" --texts "我想查询我的订单状态" "怎么退款"
    
    生成式模型训练指令（LoRA微调）：
    python main.py --route generative --model_type deepseek-1.5b --model_name_or_path deepseek-ai/deepseek-llm-1.5b --train_file ./data/train.json --valid_file ./data/valid.json --output_dir ./checkpoints/deepseek_lora --max_length 512 --batch_size 8 --epochs 3 --lr 1e-4 --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 --lora_targets "q_proj,v_proj"
    
    生成式模型预测指令：
    python main.py --route generative --model_type deepseek-1.5b --model_name_or_path deepseek-ai/deepseek-llm-1.5b --predict --ckpt ./checkpoints/deepseek_lora --labels "旅行,支付,物流,售后,账户,其他" --texts "我想查询我的订单状态"
    '''
    main()
