# 纯 CPU：DeepSeek 小模型文本场景分类完整落地方案（最终版）

## 0. 目录与产出物

```
project/
├─ data/
│  ├─ classifier_data.jsonl        # 方案1（判别式）数据：多层标签 List
│  ├─ lora_data.jsonl              # 方案2（生成式）数据：指令/输入/输出（三段式）
│  ├─ labels.yaml                  # 层级标签字典（权威源）
│  ├─ train.jsonl  valid.jsonl  test.jsonl
├─ checkpoints/
│  ├─ route1/ ...                  # 判别式最佳权重、labels.txt
│  └─ route2_lora/ ...             # LoRA 适配器、tokenizer、labels.txt
├─ export/
│  ├─ route1.onnx                  # 判别式 ONNX 导出（部署用）
│  └─ route1-int8.onnx             # 动态量化版本（CPU 友好）
├─ server/
│  ├─ app.py                       # FastAPI 服务（判别式主力 + 生成式复核）
│  └─ vocab_mask.json              # 生成式受限解码的 label 词表掩码
├─ discriminative/                  # 判别式模型相关模块
│  ├─ train.py                     # 判别式训练
│  ├─ predict.py                   # 判别式预测
│  ├─ classifier.py                # 判别式分类器
│  ├─ jsonl_cls_dataset.py         # 判别式数据集
├─ generative/                     # 生成式模型相关模块
│  ├─ train.py                     # LoRA训练
│  ├─ predict.py                   # LoRA预测
│  ├─ jsonl_sft_dataset.py         # 生成式数据集
│  ├─ data_collator.py             # 数据收集器
│  ├─ label_score.py               # 标签词评分
├─ export/                         # ONNX导出模块
│  ├─ export_onnx.py               # ONNX导出脚本
├─ utils/                          # 工具模块
│  ├─ model_utils.py               # 模型工具
│  ├─ data_utils.py                # 数据工具
├─ config/                         # 配置模块
│  ├─ __init__.py                  # 指令模板配置
└─ deepseek_text_classification_cpu.py  # 主脚本（两条路线 + ONNX导出）
```

---

## 1. 任务定义与范式选择

### 1.1 目标

* 输入：用户自然语言**反馈/对话**。
* 输出：**场景分类标签**（支持**多层级**，如 "硬件故障 > 电源问题 > 无法开机"）。
* 约束：**纯 CPU 环境**（训练 + 推理）。

### 1.2 双路线主从组合（强烈推荐）

* **主力（高吞吐、低时延）**：**方案1 判别式**（DeepSeek 冻结 + 池化 + 线性/MLP 头）。
* **复核（高精度、强解释）**：**方案2 LoRA 生成式**（仅在主力**低置信度**时触发，使用标签词评分）。

---

## 2. 标签体系与"标准分类模型"的 I/O 规范

### 2.1 层级标签权威字典（单一信息源）

建立 `data/labels.yaml`：

```yaml
root:
  硬件故障:
    电源问题:
      - 无法开机
    屏幕问题:
      - 触控异常
      - 屏幕碎裂
    电池问题:
      - 过热/耗电
  软件故障:
    应用异常:
      - 相机闪退
  网络故障:
    无线网络:
      - 频繁掉线
  其他:
    其他类:
      - 其他
```

> 作为"规范源"，训练集与服务端启动时从这里同步 `labels.txt`。

### 2.2 方案1（判别式）标准 I/O

* **输入**：`text`（必填，字符串）
* **标签**：`labels`（**层级路径 list**，如 `["硬件故障","电源问题","无法开机"]`）
* **模型 API 输出**：

  ```json
  {
    "label": "硬件故障 > 电源问题 > 无法开机",
    "prob": 0.91,
    "topk": [
      ["硬件故障 > 屏幕问题 > 触控异常", 0.05],
      ["网络故障 > 无线网络 > 频繁掉线", 0.02]
    ]
  }
  ```

### 2.3 方案2（LoRA 生成式）标准 I/O

* **输入 Prompt（训练/推理一致）**：

  ```
  [System] 你是一个场景分类助手。请从以下标签中选择一个最合适的输出，不要任何解释。
  [Labels] 可选标签: {以' > '连接的完整叶子标签集合}
  [User] 文本: {原始文本}
  [Output] 分类结果:
  ```
* **输出**：严格限定为某个**叶子标签**（如 `硬件故障 > 电源问题 > 无法开机`）。
* **约束**：推理时使用**标签词评分**（label-word scoring），计算每个标签词的概率。

---

## 3. 数据规格与示例（多层标签）

### 3.1 方案1（判别式）数据：`data/classifier_data.jsonl`

```jsonl
{"text": "手机无法开机，长按电源键没有反应。", "label": "硬件故障 > 电源问题 > 无法开机"}
{"text": "升级系统后相机闪退，无法打开。", "label": "软件故障 > 应用异常 > 相机闪退"}
{"text": "触摸不灵敏，屏幕有黑点。", "label": "硬件故障 > 屏幕问题 > 触控异常"}
{"text": "WiFi 能连上但经常掉线。", "label": "网络故障 > 无线网络 > 频繁掉线"}
{"text": "充电时很烫，掉电很快。", "label": "硬件故障 > 电池问题 > 过热/耗电"}
```

### 3.2 方案2（LoRA）数据：`data/lora_data.jsonl`

```jsonl
{"text": "手机无法开机，长按电源键没有反应。", "label": "硬件故障 > 电源问题 > 无法开机"}
{"text": "升级系统后相机闪退，无法打开。", "label": "软件故障 > 应用异常 > 相机闪退"}
{"text": "触摸不灵敏，屏幕有黑点。", "label": "硬件故障 > 屏幕问题 > 触控异常"}
{"text": "WiFi 能连上但经常掉线。", "label": "网络故障 > 无线网络 > 频繁掉线"}
{"text": "充电时很烫，掉电很快。", "label": "硬件故障 > 电池问题 > 过热/耗电"}
```

> 数据划分：70/15/15 → `train.jsonl / valid.jsonl / test.jsonl`；对长尾类做**重采样/加权**。

---

## 4. 训练与推理（命令级别操作）

> 代码已在 `deepseek_text_classification_cpu.py`，以下是**直接可用命令**。

### 4.1 环境准备（纯 CPU）

```bash
pip install -U torch transformers datasets peft scikit-learn onnxruntime onnx pyyaml
export OMP_NUM_THREADS=$(nproc) ; export MKL_NUM_THREADS=$(nproc)
```

### 4.2 选择小模型（建议 1B~3B）

* 示例占位：`--model_name_or_path <your-deepseek-small-model>`
* 若模型无 `pad_token`，脚本会自动对齐到 `<eos>`。

### 4.3 方案1：判别式训练与预测

```bash
# 训练
python deepseek_text_classification_cpu.py \
  --route discriminative \
  --model_name_or_path <your-deepseek-small-model> \
  --train_file ./data/train.jsonl \
  --valid_file ./data/valid.jsonl \
  --test_file ./data/test.jsonl \
  --label_file ./data/labels.yaml \
  --output_dir ./checkpoints/route1 \
  --max_length 256 --batch_size 16 --epochs 10 --lr 1e-3 --mlp_hidden 256

# 推理
python deepseek_text_classification_cpu.py \
  --route discriminative --predict \
  --model_name_or_path <your-deepseek-small-model> \
  --ckpt ./checkpoints/route1 \
  --texts "无法登录账户" "申请退款不到账"
```

### 4.4 方案2：LoRA 训练与预测

```bash
# 训练
python deepseek_text_classification_cpu.py \
  --route generative \
  --model_name_or_path <your-deepseek-small-model> \
  --train_file ./data/train.jsonl \
  --valid_file ./data/valid.jsonl \
  --label_file ./data/labels.yaml \
  --output_dir ./checkpoints/route2_lora \
  --max_length 512 --batch_size 4 --epochs 3 --lr 8e-5

# 推理（label-word scoring）
python deepseek_text_classification_cpu.py \
  --route generative --predict \
  --model_name_or_path <your-deepseek-small-model> \
  --ckpt ./checkpoints/route2_lora \
  --label_file ./data/labels.yaml \
  --texts "手机开机黑屏" "充电发热很严重"
```

### 4.5 ONNX 导出与量化

```bash
# 导出ONNX模型
python deepseek_text_classification_cpu.py \
  --route export \
  --ckpt ./checkpoints/route1 \
  --quantize
```

---

## 5. 评估、校准与触发策略

### 5.1 指标

* **主指标**：**Macro-F1**（长尾鲁棒）、Accuracy。
* **辅助**：每类 P/R/F1、混淆矩阵、AUC（多类可做一对多）。

### 5.2 置信度与阈值

* 判别式：`max(softmax)` 作为置信度。
* 生成式：**label-word scoring** 的 softmax 归一化概率。
* 阈值 τ：在 `valid.jsonl` 上做 sweep（如 0.5~0.9），以**整体 Macro-F1 最大**为准。低于 τ → 触发**生成式复核**；若仍低 → **人工复核**。

### 5.3 校准

* **温度缩放**（简单有效）：学习 `T>0`，替换 logits 为 `logits/T` 再 softmax，提升"置信度 ≈ 真实正确率"一致性。

---

## 6. 部署（纯 CPU）

### 6.1 推荐部署形态

* **主力在线服务：ONNX Runtime（判别式）**
  * 模型：`export/route1-int8.onnx`
  * 批处理：聚合 `N=8~32` 请求，同步推理。
  * 典型响应：

    ```json
    { "label":"硬件故障 > 电源问题 > 无法开机", "confidence":0.92, "topk":[...], "trigger_check": false }
    ```
* **复核服务：LoRA 生成式（仅低置信度触发）**
  * 限速/队列，避免挤占 CPU。
  * 使用**标签词评分**进行预测。

### 6.2 FastAPI（形状）

* `POST /classify`: 使用判别式 → 若 `conf<τ` 则内部调用 `/refine`（生成式）复核。
* `POST /refine`: LoRA 标签词评分。
* 返回结构需含：`label / confidence / route_used / latency_ms / request_id`。

---

## 7. 主动学习闭环与数据治理

1. **收集**：线上日志（文本、预测、置信度、是否复核、最终人工标签）。
2. **筛选**：低置信度 / 高置信错样本 / 代表性聚类中心样本。
3. **标注**：面板化，显示标签树，强制"叶子标签"。
4. **再训练**：
   * **判别式**：只重训头（分钟级），快速上线；
   * **LoRA**：小步微调（混少量旧样本，防遗忘）。
5. **版本管理**：`data@vX`, `ckpt@vX`, `export@vX`，每次上线记录指标 + τ。

---

## 8. 超参与 CPU 优化建议（落地可用）

* **序列长度**：`256~512`（根据文本长度分布与显存/内存权衡）。
* **Batch**：判别式 `16~64`；LoRA `2~8`（更小以适配 CPU）。
* **学习率**：判别式头 `1e-3~3e-3`；LoRA `5e-5~1e-4`。
* **早停**：`patience=2~3`，监控 Macro-F1。
* **线程**：`OMP_NUM_THREADS=物理核心数`；`MKL_NUM_THREADS=物理核心数`。
* **ONNX**：启用 **OpenMP**/**MLAS**；动态量化 **Linear/GEMM**。
* **文本清洗**：去重、去表情、脱敏（手机号/地址），统一全半角与空白。
* **类别不平衡**：`class_weight`、重采样、阈值后移。

---

## 9. 风险与兜底

* **标签漂移**：新增场景→先落"其他"，收集≥50条后成类；更新 `labels.yaml` 并同步两路。
* **输出不合规（生成式）**：使用标签词评分确保输出合法标签；解析失败直接判"不确定→人工"。
* **对抗/脏输入**：长度/字符白名单；日志采样审计。
* **隐私/合规**：日志脱敏；可选开关"禁止持久化原文，仅保留哈希与标签"。

---

## 10. ASCII 架构图（可直接贴 PPT/终端）

```
[用户反馈] -> [清洗/脱敏/去重] -> [人工标注/质检] -> [JSONL(train/val/test)]
                                      |
                                      v
                         +-----------------------------+
                         |      CPU 训练与评估         |
                         |  路线1: 判别式(冻结+线性头) |
                         |  路线2: LoRA 指令微调       |
                         +-----------------------------+
                                      |
                                      v
                              [性能评估/校准/阈值]
                                      |
                                      v
              +------------------上线服务(纯CPU)------------------+
              | 判别式主力(ONNX, 批量推理)                       |
              | 置信度<τ ? --> 生成式复核(LoRA, 标签词评分)      |
              +--------------------------------------------------+
                                      |
                                      v
                           [反馈库/主动学习样本池]
                                      |
                                      v
                              [周期性增量再训练]
```

---

## 11. 你关心的几个关键点——明确回答

* **需要监督训练吗？**
  **需要。**两条路线都需要有监督样本。冷启动可用少量规则/零样本引导，**但必须人工回标**形成高质量集。

* **如何让它"像标准分类模型"那样输入输出？**
  * **判别式**本身就是标准分类：输入 `text`，输出 `softmax 概率 + label`。
  * **生成式**通过**标签词评分**，计算每个标签的概率；外层包装成标准响应 JSON，即可与判别式一致。
  * 两路都对齐到**统一的层级标签字典**（`labels.yaml`），并在训练/推理前生成 `labels.txt`。

* **多层标签如何训练？**
  * **简单做法（推荐）**：把**叶子路径**拼接成**平面类别**（如 `硬件故障 > 电源问题 > 无法开机`），**单一多类分类**即可；
  * **进阶**：多任务头（大类/中类/小类各一个头）或"分类器链"（先大类后小类），但 CPU 成本和工程复杂度更高，建议二期再上。

---

## 12. 下一步你可以直接做的事

1. 把上面的 **`classifier_data.jsonl / lora_data.jsonl / labels.yaml`** 放到 `data/`；
2. 用你的 DeepSeek 小模型名替换命令里的占位；
3. 先跑**方案1**（判别式），拿到基线与阈值 τ；
4. 打通 **ONNX + FastAPI** 的判别式服务；
5. 针对低置信度样本，补充数据并**启动 LoRA** 训练与复核链路；
6. 打开主动学习闭环（日志→标注→增量再训）。

---

## 13. 代码结构说明

### 13.1 模块化设计

代码采用模块化设计，主要分为：

1. **判别式模块** (`discriminative/`): 包含训练、预测、分类器和数据集处理
2. **生成式模块** (`generative/`): 包含LoRA训练、预测、数据集处理和标签评分
3. **导出模块** (`export/`): ONNX模型导出和量化
4. **工具模块** (`utils/`): 通用工具函数
5. **配置模块** (`config/`): 指令模板配置

### 13.2 主要特性

1. **统一标签管理**: 从YAML文件加载层级标签，自动提取叶子标签
2. **设备自适应**: 自动检测GPU/CPU环境，优化CPU多线程性能
3. **内存优化**: 支持低内存模式加载大模型
4. **标准化接口**: 训练和预测接口统一，便于扩展和维护

### 13.3 扩展性

代码设计考虑了扩展性，可以轻松支持：

1. 新的预训练模型
2. 不同的分类头设计
3. 额外的评估指标
4. 新的优化策略

通过模块化设计和清晰的接口定义，代码保持了良好的可维护性和可扩展性。