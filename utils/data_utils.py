import os
import json
import yaml
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def map_labels_to_hierarchy(labels_yaml_path: str, input_files: List[str], output_files: List[str]):
    """
    将 JSONL 文件中的标签映射为完整层级标签（一级>二级>三级）
    支持：
        1. JSONL 标签已经是完整三级结构
        2. JSONL 标签只是叶子节点
    :param labels_yaml_path: labels.yaml 路径
    :param input_files: 输入 JSONL 文件路径列表
    :param output_files: 输出 JSONL 文件路径列表
    """
    # 确保输出目录存在
    for output_file in output_files:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # 读取 labels.yaml
    with open(labels_yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    # 获取顶层节点列表
    if isinstance(yaml_data, dict) and "labels" in yaml_data:
        labels_tree = yaml_data["labels"]
    elif isinstance(yaml_data, list):
        labels_tree = yaml_data
    else:
        raise ValueError("无法识别 labels.yaml 顶层结构，请确保是 'labels' 字段或列表")

    # 构建完整路径和叶子节点映射
    def build_maps(nodes, prefix=None):
        if prefix is None:
            prefix = []
        full_map = {}
        leaf_map = {}

        for node in nodes:
            if isinstance(node, str):
                path = ">".join(prefix + [node]).strip()
                full_map[path] = path
                leaf_map[node.strip()] = path
            elif isinstance(node, dict) and "name" in node:
                name = node["name"].strip()
                children = node.get("children", [])
                child_full_map, child_leaf_map = build_maps(children, prefix + [name])
                full_map.update(child_full_map)
                leaf_map.update(child_leaf_map)
            elif isinstance(node, list):
                child_full_map, child_leaf_map = build_maps(node, prefix)
                full_map.update(child_full_map)
                leaf_map.update(child_leaf_map)
            elif isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, list):
                        child_full_map, child_leaf_map = build_maps(value, prefix + [key])
                        full_map.update(child_full_map)
                        leaf_map.update(child_leaf_map)
        return full_map, leaf_map

    full_map, leaf_map = build_maps(labels_tree)

    # 遍历输入文件并写入输出文件
    for in_file, out_file in zip(input_files, output_files):
        data = []
        with open(in_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

        with open(out_file, "w", encoding="utf-8") as fout:
            for item in data:
                original_label = item["label"].strip()
                mapped_label = None

                # 先尝试用完整路径匹配
                if original_label in full_map:
                    mapped_label = full_map[original_label]
                # 再尝试用叶子节点匹配
                elif original_label.split(">")[-1] in leaf_map:
                    leaf = original_label.split(">")[-1]
                    mapped_label = leaf_map[leaf]

                if mapped_label:
                    item["label"] = mapped_label
                else:
                    print(f"警告: 找不到标签映射: {original_label}")

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"已生成映射文件: {out_file}")
if __name__ == "__main__":
    # 使用不同的输入和输出路径
    map_labels_to_hierarchy(
        labels_yaml_path="../data/labels.yaml",
        input_files=["../data/train_original.jsonl", "../data/valid_original.jsonl", "../data/test_original.jsonl"],
        output_files=["../data/train.jsonl", "../data/valid.jsonl", "../data/test.jsonl"]
    )