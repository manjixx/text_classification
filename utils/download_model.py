# utils/download_model.py

import os
from huggingface_hub import hf_hub_download, list_repo_files, login

def download_model(repo_id: str, save_dir: str):
    """
    自动下载 Hugging Face 模型到指定目录。
    会自动检测权重文件（model.safetensors 或 pytorch_model.bin），
    并下载配置文件和 tokenizer。
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] 开始下载模型 {repo_id} 到 {save_dir} ...")

    # 获取仓库文件列表
    files = list_repo_files(repo_id)
    print(f"[INFO] 仓库文件列表: {files}")

    # 权重文件优先使用 model.safetensors，如果不存在再尝试 pytorch_model.bin
    weight_file = None
    for candidate in ["model.safetensors", "pytorch_model.bin"]:
        if candidate in files:
            weight_file = candidate
            break

    if not weight_file:
        raise FileNotFoundError("仓库中没有找到权重文件 (model.safetensors / pytorch_model.bin)")

    # 下载权重文件
    print(f"[INFO] 下载权重文件: {weight_file}")
    hf_hub_download(repo_id=repo_id, filename=weight_file, cache_dir=save_dir)

    # 下载常用配置文件
    for f in ["config.json", "tokenizer.json", "special_tokens_map.json"]:
        if f in files:
            print(f"[INFO] 下载文件: {f}")
            hf_hub_download(repo_id=repo_id, filename=f, cache_dir=save_dir)

    print(f"[SUCCESS] 模型下载完成，保存在 {save_dir}")

# ----------------------------
# 一键执行下载
# ----------------------------
if __name__ == "__main__":
    # 1. 设置本地保存路径
    save_dir = r"E:\text_classification\models\original\DeepSeek-R1-Distill-Qwen-1.5B"

    # 2. 输入 Hugging Face Token 并登录
    hf_token = input("请输入 Hugging Face Token: ").strip()
    if not hf_token.startswith("hf_"):
        print("[ERROR] Token 格式不正确，请以 hf_ 开头")
        exit(1)

    login(token=hf_token)
    print("[INFO] 登录成功")

    # 3. 模型仓库 ID
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # 4. 下载模型
    download_model(repo_id, save_dir)
