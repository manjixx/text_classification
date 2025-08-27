# utils/download_model.py

import os
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files, login


def download_model(repo_id: str, save_dir: str, use_auth_token: bool = False):
    """
    自动下载 Hugging Face 模型到指定目录。
    会自动检测权重文件（model.safetensors 或 pytorch_model.bin），
    并下载配置文件和 tokenizer。

    参数:
        repo_id (str): Hugging Face 模型仓库ID，例如 "bert-base-chinese" 或 "hfl/chinese-roberta-wwm-ext"
        save_dir (str): 本地保存目录
        use_auth_token (bool): 是否使用认证token（对于gated model）
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] 开始下载模型 {repo_id} 到 {save_dir} ...")

    # 获取仓库文件列表
    try:
        files = list_repo_files(repo_id, use_auth_token=use_auth_token)
        print(f"[INFO] 仓库文件列表: {files}")
    except Exception as e:
        print(f"[ERROR] 获取仓库文件列表失败: {e}")
        return False

    # 权重文件优先使用 model.safetensors，如果不存在再尝试 pytorch_model.bin
    weight_file = None
    for candidate in ["model.safetensors", "pytorch_model.bin"]:
        if candidate in files:
            weight_file = candidate
            break

    if not weight_file:
        print("[WARNING] 仓库中没有找到常见的权重文件 (model.safetensors / pytorch_model.bin)，尝试下载所有文件")
        # 如果找不到常见的权重文件，尝试使用snapshot_download下载整个仓库
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                use_auth_token=use_auth_token
            )
            print(f"[SUCCESS] 模型 {repo_id} 下载完成，保存在 {save_dir}")
            return True
        except Exception as e:
            print(f"[ERROR] 下载失败: {e}")
            return False

    # 下载权重文件
    print(f"[INFO] 下载权重文件: {weight_file}")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=weight_file,
            local_dir=save_dir,
            resume_download=True,
            use_auth_token=use_auth_token
        )
    except Exception as e:
        print(f"[ERROR] 下载权重文件失败: {e}")
        return False

    # 下载常用配置文件
    config_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                    "vocab.json"]
    for f in config_files:
        if f in files:
            print(f"[INFO] 下载文件: {f}")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f,
                    local_dir=save_dir,
                    resume_download=True,
                    use_auth_token=use_auth_token
                )
            except Exception as e:
                print(f"[WARNING] 下载 {f} 失败: {e}")

    print(f"[SUCCESS] 模型 {repo_id} 下载完成，保存在 {save_dir}")
    return True


# ----------------------------
# 一键执行下载
# ----------------------------
if __name__ == "__main__":
    # 需要下载的模型列表
    models_to_download = [
        "bert-base-chinese",  # BERT中文基础模型
        "hfl/chinese-roberta-wwm-ext",  # 哈工大ROBERTA中文全词掩码模型
    ]

    # 设置本地保存路径前缀
    base_save_dir = r"E:\text_classification\models\original"

    # 检查是否需要Token（例如，对于gated model）
    need_auth = False
    hf_token = None

    # 如果你的网络访问Hugging Face较慢，可以考虑使用国内镜像
    # 设置环境变量（可选）
    # os.environ['HF_ENDPOINT'] = 'https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models'

    # 如果需要Token，则输入并登录
    if need_auth:
        hf_token = input("请输入 Hugging Face Token: ").strip()
        if not hf_token.startswith("hf_"):
            print("[ERROR] Token 格式不正确，请以 hf_ 开头")
            exit(1)
        login(token=hf_token)
        print("[INFO] 登录成功")

    # 依次下载每个模型
    for repo_id in models_to_download:
        # 从repo_id提取模型名称作为子目录名
        if "/" in repo_id:
            model_name = repo_id.split("/")[1]
        else:
            model_name = repo_id

        save_dir = os.path.join(base_save_dir, model_name)
        print(f"\n{'=' * 50}")
        print(f"开始处理模型: {repo_id}")
        print(f"{'=' * 50}")

        success = download_model(repo_id, save_dir, use_auth_token=hf_token if need_auth else False)
        if success:
            print(f"[SUCCESS] 模型 {repo_id} 下载完成！")
        else:
            print(f"[FAILED] 模型 {repo_id} 下载失败！")

    print("\n所有模型下载任务已完成！")