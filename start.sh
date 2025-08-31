# 判别式模型训练指令（BERT）
python main.py --route discriminative --model_type bert-base --model_name_or_path ./models/original/bert-base-chinese --train_file ./data/train.jsonl --valid_file ./data/valid.jsonl --test_file ./data/test.jsonl --labels ./data/labels.yaml --output_dir ./checkpoints/bert_base --max_length 256 --batch_size 64 --epochs 16 --lr 1e-3 --mlp_hidden 256 --dropout 0.1 --weight_decay 0.0

# 判别式模型训练指令（RoBERTa）
python main.py --route discriminative --model_type roberta-wwm --model_name_or_path ./models/original/chinese-roberta-wwm-ext --train_file ./data/train.jsonl --valid_file ./data/valid.jsonl --test_file ./data/test.jsonl --labels ./data/labels.yaml --output_dir ./checkpoints/chinese_roberta_wwm_ext --max_length 256 --batch_size 64 --epochs 16 --lr 1e-3 --mlp_hidden 256 --dropout 0.1 --weight_decay 0.0

# 判别式模型预测指令（BERT）
python main.py --route discriminative --model_type bert-base --model_name_or_path ./models/original/bert-base-chinese --predict --ckpt ./checkpoints/bert_base --labels ./checkpoints/bert_base/labels.txt --texts "手机屏幕碎了怎么办" "电池耗电特别快" "充电时手机发烫"

# 判别式模型预测指令（RoBERTa）
python main.py --route discriminative --model_type roberta-wwm --model_name_or_path ./models/original/chinese-roberta-wwm-ext --predict --ckpt ./checkpoints/chinese_roberta_wwm_ext --labels ./checkpoints/chinese_roberta_wwm_ext/labels.txt --texts "手机屏幕碎了怎么办" "电池耗电特别快" "充电时手机发烫"

# 生成式模型训练指令（LoRA微调）
python main.py --route generative --model_type deepseek-1.5b --model_name_or_path ./models/original/deepseek-llm-1.5b --train_file ./data/train.jsonl --valid_file ./data/valid.jsonl --test_file ./data/test.jsonl --labels ./data/labels.yaml --output_dir ./checkpoints/deepseek_lora --max_length 512 --batch_size 8 --epochs 10 --lr 1e-4 --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 --lora_targets "q_proj,v_proj" --eval_bs 8

# 生成式模型预测指令
python main.py --route generative --model_type deepseek-1.5b --model_name_or_path ./models/original/deepseek-llm-1.5b --predict --ckpt ./checkpoints/deepseek_lora/best_model.pt --labels ./data/labels.yaml --texts "手机屏幕碎了怎么办"