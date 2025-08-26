#!/usr/bin/env python3
import yaml
import json
from transformers import AutoTokenizer

def load_labels(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        label_tree = yaml.safe_load(f)
    leaf_labels = []
    # 递归函数，收集所有叶子节点路径
    def traverse(path, node):
        if isinstance(node, dict):
            for k, v in node.items():
                traverse(path + [k], v)
        elif isinstance(node, list):
            for item in node:
                full_path = " > ".join(path + [item])
                leaf_labels.append(full_path)
    traverse([], label_tree)
    return leaf_labels

def main():
    labels = load_labels('./data/labels.yaml')
    
    # 1. 生成 labels.txt (for Route 1)
    with open('./checkpoints/route1/labels.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(labels))
    print(f"Generated {len(labels)} labels for Route1.")

    # 2. 生成 vocab_mask.json (for Route 2 restricted decoding)
    # 假设使用DeepSeek模型
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-lite-chat", trust_remote_code=True)
    allowed_token_ids = set()
    for lab in labels:
        # 获取标签文本对应的token IDs
        token_ids = tokenizer(lab, add_special_tokens=False)['input_ids']
        allowed_token_ids.update(token_ids)
    # 确保一些必要的特殊token也在允许范围内
    necessary_tokens = [tokenizer.pad_token_id, tokenizer.eos_token_id]
    allowed_token_ids.update([x for x in necessary_tokens if x is not None])
    
    vocab_mask = {str(i): (i in allowed_token_ids) for i in range(len(tokenizer))}
    with open('./server/vocab_mask.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_mask, f, ensure_ascii=False, indent=2)
    print(f"Generated vocab mask with {len(allowed_token_ids)} allowed tokens for Route2.")

    # 3. (Optional) 生成一个用于快速验证的样本文件
    sample_texts = ["设备无法启动", "应用频繁崩溃", "网络连接不稳定"]
    with open('./data/sample_predict.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(sample_texts))
    print("Generated sample prediction input file.")

if __name__ == '__main__':
    main()