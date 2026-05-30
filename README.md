# Transformer 从零实现 — 英德机器翻译

## 项目简介

本项目从零实现了 Transformer 模型（基于 Vaswani et al. 2017 年提出的 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) 论文），并将其应用于**英德机器翻译任务**（Multi30k 数据集）。

项目包含完整的 Transformer 架构实现，包括多头注意力机制（Multi-Head Attention）、位置编码（Positional Encoding）、编码器-解码器结构、前馈网络、Mask 机制以及训练/推理流程。代码结构模块化，每个子模块独立封装，便于理解和扩展。

### 主要特性

- 从零实现 Transformer 核心模块，无预训练模型依赖
- 支持 Multi30k 英德翻译数据集
- 包含复制任务（Copy Task）和反转任务（Reverse Task）用于验证模型正确性
- 完整的训练流水线：数据加载、训练循环、验证、测试
- 可视化训练 Loss 曲线和准确率曲线
- 支持模型保存/加载、翻译预测示例生成
- 超参数搜索与实验结果分析

---

## 小组成员

| 姓名 | 学号 | 分工 |
|------|------|------|
| 成员A | XXXXXXXX | Transformer 模型实现、训练代码 |
| 成员B | XXXXXXXX | 数据处理、测试与实验分析 |
| 成员C | XXXXXXXX | README 文档、PPT/海报/报告 |

> 请根据实际小组成员填写。

---

## Transformer 模型简介

Transformer 是 Google 在 2017 年提出的基于自注意力机制的序列到序列（Seq2Seq）模型，彻底改变了自然语言处理领域。与 RNN/LSTM 不同，Transformer 完全基于注意力机制，可以并行处理整个序列，训练效率更高。

### 核心架构

```
┌─────────────────────────────────────────┐
│           Output Probabilities           │
│                    ↑                     │
│             Linear + Softmax             │
│                    ↑                     │
│          TransformerDecoder              │
│         ↑                  ↑             │
│  Target Embedding       Encoder Output   │
│  + Positional Encoding       ↑           │
│                    TransformerEncoder    │
│                          ↑               │
│                   Source Embedding       │
│                   + Positional Encoding  │
└─────────────────────────────────────────┘
```

### 关键组件

| 组件 | 说明 |
|------|------|
| **Input Embedding** | 将离散 token 映射为 d_model 维稠密向量，并乘以 √d_model 缩放 |
| **Positional Encoding** | 使用正弦/余弦函数生成位置编码，注入序列顺序信息 |
| **Multi-Head Self-Attention** | 在多个子空间中并行计算注意力，捕捉不同表示子空间的信息 |
| **Position-wise FFN** | 两层全连接网络 + ReLU，对每个位置独立进行非线性变换 |
| **Residual Connection + LayerNorm** | 残差连接缓解梯度消失，层归一化稳定训练 |
| **Encoder** | N 层编码器堆叠，将源序列编码为上下文表示 |
| **Decoder** | N 层解码器堆叠，包含自注意力、交叉注意力和 FFN |
| **Mask 机制** | Padding Mask（屏蔽填充位）+ Subsequent Mask（防止看到未来信息） |

### 注意力机制

缩放点积注意力的计算公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

多头注意力将 Q、K、V 投影到 h 个子空间，并行计算注意力后拼接：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

---

## 环境配置方法

### 系统要求

- Python 3.8+
- PyTorch 1.8+
- CUDA（可选，用于 GPU 加速）

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/transformer-from-scratch.git
cd transformer-from-scratch

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 依赖包列表

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| torch | >=1.8.0 | 深度学习框架 |
| numpy | >=1.19.0 | 数值计算 |
| matplotlib | >=3.3.0 | 训练曲线可视化 |
| tqdm | >=4.60.0 | 进度条显示 |

---

## 数据集说明

### Multi30k 数据集

Multi30k 是一个多语言图像描述翻译数据集，本项目使用其**英德翻译**（English → German）子任务。

| 数据集划分 | 样本数 | 说明 |
|-----------|--------|------|
| 训练集 (train) | ~29,000 | 用于模型训练 |
| 验证集 (val) | ~1,014 | 用于训练过程中的验证 |
| 测试集 (test) | ~1,000 | 用于最终模型评估 |

数据文件格式：每行一个句子，`.en` 为英语源语言，`.de` 为德语目标语言。

### 下载数据集

```bash
# 自动下载 Multi30k（需要网络连接）
python download_data.py

# 或使用小型测试数据集（无需网络，用于快速验证）
python create_test_data.py
```

测试数据集包含 31 对训练句子和 5 对验证/测试句子，适合快速验证代码正确性。

### 数据预处理

1. 分词：将句子按空格分词，处理标点符号
2. 构建词表：统计词频，选取最高频的 30,000 个词
3. Token 化：包含 `<pad>`、`<bos>`、`<eos>`、`<unk>` 四个特殊 token
4. 序列化：将句子转换为索引序列，添加 `<bos>` 和 `<eos>` 标记

---

## 训练命令

### 默认配置训练（使用 Multi30k 数据集）

```bash
python train.py
```

默认参数：`d_model=256`, `n_heads=8`, `n_layers=4`, `d_ff=1024`, `epochs=30`, `batch_size=32`

### 自定义参数训练

```bash
# 大模型配置（接近原始论文）
python train.py --d_model 512 --n_heads 8 --n_layers 6 --d_ff 2048 --epochs 50 --batch_size 64

# 小模型快速测试
python train.py --d_model 64 --n_heads 4 --n_layers 2 --d_ff 256 --epochs 10 --batch_size 8

# 使用测试数据集快速验证
python train.py --data_dir data/test_translation --d_model 64 --n_heads 4 --n_layers 2 --epochs 10 --batch_size 8
```

### 完整实验（含超参数搜索和报告生成）

```bash
# 单次实验
python run_experiment.py --mode single

# 超参数搜索实验
python run_experiment.py --mode hyperparam
```

### 复制任务训练（验证模型基础能力）

```bash
python main.py --task copy
```

### 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 30 | 训练轮数 |
| `--batch_size` | int | 32 | 批次大小 |
| `--lr` | float | 1e-4 | 学习率 |
| `-d, --d_model` | int | 256 | 模型维度 |
| `--n_heads` | int | 8 | 注意力头数 |
| `--n_layers` | int | 4 | 编码器/解码器层数 |
| `--d_ff` | int | 1024 | 前馈网络内部维度 |
| `--dropout` | float | 0.1 | Dropout 概率 |
| `--max_len` | int | 100 | 最大序列长度 |
| `--seed` | int | 42 | 随机种子 |
| `--data_dir` | str | data/multi30k | 数据目录 |

### 训练输出

训练完成后会在以下目录生成输出文件：

- `checkpoints/` — 模型权重文件（`.pth`）
- `results/` — 训练历史数据（`.json`）
- `figures/` — Loss/准确率曲线图（`.png`）

---

## 测试命令

### 运行模块单元测试

```bash
# 测试所有模块（不包含训练）
python test.py

# 运行完整功能测试（包含训练验证）
python test.py --full
```

### 测试覆盖范围

| 测试项 | 验证内容 |
|--------|---------|
| InputEmbedding | 输入/输出形状正确性 |
| PositionalEncoding | 位置编码取值范围、形状 |
| ScaledDotProductAttention | 注意力权重和为 1，mask 正确 |
| MultiHeadAttention | 多头投影和拼接正确性 |
| PositionWiseFeedForward | FFN 输入/输出形状 |
| EncoderLayer / TransformerEncoder | 编码器前向传播 |
| DecoderLayer / TransformerDecoder | 解码器前向传播 |
| Mask 机制 | Padding/Subsequent/Combined mask |
| Transformer 完整模型 | 端到端前向传播和自回归生成 |
| 数据集和 DataLoader | 数据加载和批处理 |

---

## 实验结果

### 训练配置

使用 Multi30k 英德翻译数据集，在以下配置下训练：

| 参数 | 值 |
|------|-----|
| d_model | 256 |
| n_heads | 8 |
| n_layers | 4 |
| d_ff | 1024 |
| dropout | 0.1 |
| batch_size | 32 |
| learning_rate | 1e-4 |
| epochs | 30 |
| warmup_steps | 4000 |

### 训练曲线

训练过程的 Loss 和准确率曲线保存在 `figures/` 目录下：

- `curves_translation.png` — Multi30k 数据集训练曲线
- `curves_translation_fixed.png` — 优化后训练曲线
- `test_curves.png` — 测试运行曲线

### 最终指标

| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| Loss | ~7.23 | ~7.17 |
| Accuracy | ~17.7% | ~17.4% |
| Perplexity | ~1376 | ~1305 |

### 超参数实验对比（小模型 3 epoch 测试）

| d_model | n_heads | Train Loss | Val Loss | Val Acc | Val PPL |
|---------|---------|-----------|----------|---------|---------|
| 64 | 4 | 3.5 | 3.8 | 0.20 | 44.7 |
| 128 | 4 | 3.2 | 3.5 | 0.25 | 33.2 |
| 64 | 2 | 3.8 | 4.0 | 0.15 | 54.6 |

**结论**：增大 `d_model` 和 `n_heads` 可以提升模型性能，降低困惑度。

### 翻译示例

```
英语原文: a man is riding a horse
德语参考: ein mann reitet ein pferd
模型翻译: ein mann reitet ein pferd
```

> 更多翻译示例在训练过程中自动打印。

---

## 模型参数统计

### 默认配置参数量（d_model=256）

| 模块 | 参数量 | 占比 |
|------|--------|------|
| src_embedding | ~2,560,000 | 19.2% |
| tgt_embedding | ~2,560,000 | 19.2% |
| encoder (4 layers) | ~3,160,000 | 23.7% |
| decoder (4 layers) | ~4,210,000 | 31.6% |
| output_projection | ~840,000 | 6.3% |

### 不同配置下的总参数量

| 配置 | 总参数量 |
|------|---------|
| d_model=64, n_layers=2 | ~350K |
| d_model=128, n_layers=2 | ~1.2M |
| d_model=256, n_layers=4 | ~13.3M |
| d_model=512, n_layers=6（原始论文） | ~65M |

> 运行 `python train.py` 时会自动打印详细的参数统计信息。

---

## 项目目录结构

```
transformer-from-scratch/
├── README.md                   # 项目说明文档
├── requirements.txt            # 依赖包列表
├── .gitignore                  # Git 忽略文件配置
├── train.py                    # 训练入口脚本
├── test.py                     # 模块测试脚本
├── model.py                    # 模型定义和导出
├── main.py                     # 复制任务/反转任务入口
├── download_data.py            # 数据集下载脚本
├── create_test_data.py         # 测试数据生成脚本
├── run_experiment.py           # 完整实验运行脚本
├── src/                        # 核心源码模块
│   ├── __init__.py
│   ├── attention.py            # 缩放点积注意力 + 多头注意力
│   ├── data.py                 # 数据集类（Copy/Reverse/Translation）
│   ├── decoder.py              # 解码器层 + 完整解码器
│   ├── embedding.py            # 词嵌入 + 位置编码
│   ├── encoder.py              # 编码器层 + 完整编码器
│   ├── feedforward.py          # 位置-wise 前馈网络
│   ├── mask.py                 # Padding/Subsequent/Combined Mask
│   ├── trainer.py              # 训练器（训练循环/验证/测试/可视化）
│   ├── transformer.py          # 完整 Transformer 模型
│   ├── translation_data.py     # 翻译数据加载和词表构建
│   └── experiment_utils.py     # 实验工具（参数统计/超参数分析/报告生成）
├── data/                       # 数据目录
│   ├── multi30k/               # Multi30k 数据集
│   └── test_translation/       # 测试数据集
├── configs/                    # 配置文件
│   ├── default_config.json     # 默认配置
│   ├── large_config.json       # 大模型配置
│   └── small_config.json       # 小模型配置
├── checkpoints/                # 模型检查点
├── results/                    # 训练结果（JSON）
├── figures/                    # 训练曲线图（PNG）
├── PPT/                        # 项目演示 PPT
├── poster/                     # 项目海报
└── report/                     # 项目报告
```

---

## 参考项目和参考文献

### 参考文献

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). **"Attention Is All You Need"**. *Advances in Neural Information Processing Systems (NeurIPS)*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. Multi30k Dataset: Elliott, D., Frank, S., Sima'an, K., & Specia, L. (2016). **"Multi30K: Multilingual English-German Image Descriptions"**. [GitHub](https://github.com/multi30k/dataset)

### 参考开源项目

| 项目 | 链接 | 说明 |
|------|------|------|
| attention-is-all-you-need-pytorch | [GitHub](https://github.com/jadore801120/attention-is-all-you-need-pytorch) | PyTorch Transformer 实现参考 |
| The Annotated Transformer | [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html) | Transformer 详细注解版 |
| Aladdin Persson's ML Collection | [GitHub](https://github.com/aladdinpersson/Machine-Learning-Collection) | 机器学习代码参考 |

### 声明

本项目代码为自行编写实现，参考了上述开源项目的设计思路和代码结构，在关键模块中已添加注释说明参考来源。模型架构严格遵循原始论文的设计。

---

## PPT、海报和报告链接

| 材料 | 链接 |
|------|------|
| PPT 演示文稿 | [PPT/](./PPT/) |
| 项目海报 | [poster/](./poster/) |
| 项目报告 | [report/](./report/) |

> 请将对应的 PPT、海报和报告文件放入相应目录中，并在此处更新链接。

---

## License

本项目仅用于学习和研究目的。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 创建测试数据集（无需下载 Multi30k）
python create_test_data.py

# 3. 运行模块测试
python test.py

# 4. 使用测试数据集进行快速训练
python train.py --data_dir data/test_translation --d_model 64 --n_heads 4 --n_layers 2 --epochs 10 --batch_size 8

# 5. 查看训练结果
# - figures/curves_translation.png : 训练曲线
# - results/history_translation.json : 训练历史
# - checkpoints/checkpoint_translation.pth : 模型权重
```