"""
============================================================
Transformer 项目实验指南 - 机器翻译任务
============================================================
本指南详细说明如何使用本项目完成以下实验任务：
  1. 数据集下载与预处理 (Multi30k 英德翻译数据集)
  2. 词表构建
  3. 模型训练
  4. 模型验证与测试
  5. Loss 曲线绘制
  6. 预测样例生成
  7. 模型效果分析
  8. 参数量统计
  9. 超参数影响分析

参考项目:
  - https://github.com/jadore801120/attention-is-all-you-need-pytorch
  - https://github.com/aladdinpersson/Machine-Learning-Collection

============================================================
"""

# 目录
1. 环境准备
2. 数据集下载与预处理
3. 词表构建
4. 模型训练
5. 模型验证与测试
6. Loss 曲线绘制
7. 预测样例
8. 模型效果分析
9. 参数量统计
10. 超参数影响分析


## 1. 环境准备

首先确保已安装必要依赖：

```bash
pip install torch numpy matplotlib tqdm
```

查看项目结构：

```
deeplearn/
├── src/                    # 核心代码模块
│   ├── embedding.py        # 词嵌入 + 位置编码
│   ├── attention.py        # 注意力机制
│   ├── feedforward.py      # 前馈网络
│   ├── encoder.py          # 编码器
│   ├── decoder.py          # 解码器
│   ├── transformer.py      # 完整模型
│   ├── mask.py             # Mask 机制
│   ├── data.py             # 数据处理
│   └── trainer.py          # 训练器
├── main.py                 # 主训练入口
├── test_modules.py         # 单元测试
└── experiments.md          # 本实验指南
```


## 2. 数据集下载与预处理

### 2.1 下载 Multi30k 数据集

Multi30k 是一个英德翻译数据集，包含约 30,000 对句子。

```bash
# 创建数据集目录
mkdir -p data/multi30k

# 下载训练集
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/train.en -P data/multi30k/
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/train.de -P data/multi30k/

# 下载验证集
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/val.en -P data/multi30k/
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/val.de -P data/multi30k/

# 下载测试集
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/test_2016_flickr.en -P data/multi30k/
wget https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/test_2016_flickr.de -P data/multi30k/

# 重命名测试集
mv data/multi30k/test_2016_flickr.en data/multi30k/test.en
mv data/multi30k/test_2016_flickr.de data/multi30k/test.de
```

### 2.2 数据集格式

每个文件包含一行一个句子：

```
# train.en (英语)
A man is riding a horse.
A woman is playing the violin.
...

# train.de (德语)
Ein Mann reitet ein Pferd.
Eine Frau spielt Geige.
...
```

### 2.3 数据预处理

项目已提供 `data.py` 模块处理数据集。需要创建一个专门的翻译数据集类。


## 3. 词表构建

### 3.1 使用项目提供的工具

项目的 `data.py` 模块提供了 `build_vocab` 函数：

```python
from src.data import build_vocab

# 特殊 token
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

# 构建词表
tokens = ["man", "riding", "horse", ...]  # 从数据集中提取的所有词
vocab = build_vocab(tokens)
```

### 3.2 词表大小

对于 Multi30k 数据集：
- 英语词表：约 10,000-15,000 个词
- 德语词表：约 10,000-15,000 个词


## 4. 模型训练

### 4.1 运行训练脚本

使用 `main.py` 训练模型：

```bash
# 训练复制任务（快速测试）
python main.py --task copy --epochs 10 --batch_size 32

# 训练反转任务
python main.py --task reverse --epochs 10 --batch_size 32
```

### 4.2 训练日志解读

训练过程会输出：

```
============================================================
Transformer 从零实现 - 训练与评估
============================================================
设备: cuda
模型配置: d_model=128, n_heads=8, n_layers=3, d_ff=512
训练配置: epochs=30, batch_size=64, lr=0.0001

Epoch   1/30 | Time: 45.2s | LR: 4.55e-07 | Train Loss: 6.2345 | Train Acc: 0.0089 | Val Loss: 6.1823 | Val Acc: 0.0123 | Val PPL: 486.23 *
Epoch   2/30 | Time: 44.8s | LR: 9.09e-07 | Train Loss: 5.8912 | Train Acc: 0.0156 | Val Loss: 5.7934 | Val Acc: 0.0215 | Val PPL: 328.12 *
...
```

关键指标：
- **Loss**: 交叉熵损失，越低越好
- **Accuracy**: 预测正确的 token 比例
- **PPL (Perplexity)**: 困惑度，衡量模型预测能力，越低越好
- **LR**: 当前学习率


## 5. 模型验证与测试

### 5.1 验证集评估

训练过程中每 epoch 会自动在验证集上评估：

```python
# trainer.py 中的验证逻辑
val_loss, val_acc, val_ppl = self._validate()
```

### 5.2 测试集评估

训练完成后会自动在测试集上评估：

```
============================================================
测试结果:
  Loss:        3.9347
  Accuracy:    0.1079
  Perplexity:  51.15
============================================================
```


## 6. Loss 曲线绘制

### 6.1 自动生成

训练完成后自动保存训练曲线到 `curves_copy_task.png`：

```python
# trainer.py 中的绘图函数
trainer.plot_curves("curves.png")
```

### 6.2 曲线解读

训练曲线包含 4 个子图：
1. **损失曲线**: 训练/验证损失随 epoch 变化
2. **准确率曲线**: 训练/验证准确率随 epoch 变化
3. **困惑度曲线**: 训练/验证困惑度随 epoch 变化
4. **学习率曲线**: 学习率随 epoch 变化（Noam 调度）

### 6.3 过拟合检测

- 如果训练损失持续下降但验证损失开始上升，表示过拟合
- 可以通过增加 dropout、减少模型大小或提前停止来缓解


## 7. 预测样例

### 7.1 自回归生成

项目支持自回归推理生成：

```python
from src.transformer import Transformer
from src.mask import create_encoder_mask

# 加载模型
model = Transformer(src_vocab_size, tgt_vocab_size)
model.load_state_dict(torch.load("checkpoint.pth"))

# 生成预测
src = torch.tensor([[1, 5, 3, 7, 2]])  # 输入序列
src_mask = create_encoder_mask(src)

generated = model.generate(
    src,
    max_len=50,
    bos_idx=1,  # <bos>
    eos_idx=2   # <eos>
)
```

### 7.2 生成示例

复制任务的生成结果：

```
源序列:    [47, 20, 4, 33, 4, 30, 10, 20, 10]
目标序列:  [1, 47, 20, 4, 33, 4, 30, 10, 20, 10, 2]
生成序列:  [1, 47, 20, 4, 33, 4, 30, 10, 20, 10, 2]  # 完美匹配
```


## 8. 模型效果分析

### 8.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **Loss** | 交叉熵损失 | `-log(p(y|x))` 的均值 |
| **Accuracy** | 单 token 准确率 | 正确预测的 token 数 / 总 token 数 |
| **PPL** | 困惑度 | `exp(loss)` |
| **BLEU** | 机器翻译评估 | 需要额外计算 |

### 8.2 效果分析示例

假设训练结果：
```
训练集: Loss=1.23, Accuracy=0.78, PPL=3.42
验证集: Loss=1.85, Accuracy=0.65, PPL=6.36
测试集: Loss=1.92, Accuracy=0.63, PPL=6.82
```

分析：
- 模型在训练集上表现良好（低 loss，高 accuracy）
- 验证集和测试集性能略有下降，存在轻微过拟合
- 困惑度在合理范围内（< 10）


## 9. 参数量统计

### 9.1 计算方法

```python
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量: {n_params:,}")
```

### 9.2 模型结构参数量

| 组件 | 参数数量 | 计算公式 |
|------|----------|----------|
| 嵌入层 | `vocab_size × d_model` | 词嵌入矩阵 |
| 编码器层 | `6 × d_model² + 4 × d_model × d_ff` | 多头注意力 + FFN |
| 解码器层 | `9 × d_model² + 4 × d_model × d_ff` | 3个注意力 + FFN |
| 输出层 | `d_model × tgt_vocab_size` | 线性投影 |

### 9.3 示例

```python
# d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=30000
# 总参数量 ≈ 160M (与原始论文一致)
```


## 10. 超参数影响分析

### 10.1 关键超参数列表

| 超参数 | 默认值 | 影响 |
|--------|--------|------|
| `d_model` | 512 | 模型维度，越大表达能力越强但计算代价越高 |
| `n_heads` | 8 | 注意力头数，越多能捕捉更多关系模式 |
| `n_layers` | 6 | 编码器/解码器层数，越深模型能力越强 |
| `d_ff` | 2048 | FFN 内部维度，通常为 4×d_model |
| `batch_size` | 64 | 批次大小，影响训练稳定性和速度 |
| `lr` | 1e-4 | 学习率，Noam 调度自动调整 |
| `dropout` | 0.1 | Dropout 概率，防止过拟合 |
| `epochs` | 30 | 训练轮数 |

### 10.2 实验设计建议

建议设计对比实验分析关键超参数：

```python
# 实验 1: d_model 的影响
configs = [
    {"d_model": 128, "n_heads": 4},
    {"d_model": 256, "n_heads": 8},
    {"d_model": 512, "n_heads": 8},
]

# 实验 2: n_layers 的影响
configs = [
    {"n_layers": 2},
    {"n_layers": 4},
    {"n_layers": 6},
]

# 实验 3: dropout 的影响
configs = [
    {"dropout": 0.0},
    {"dropout": 0.1},
    {"dropout": 0.3},
]
```

### 10.3 预期结果

- **d_model 增大**: 模型表达能力增强，但训练更慢，可能过拟合
- **n_layers 增加**: 模型深度增加，能学习更复杂的特征
- **dropout 增大**: 正则化增强，防止过拟合，但可能欠拟合


## 附录：完整训练命令示例

```bash
# 完整训练配置
python main.py \
    --task copy \
    --epochs 50 \
    --batch_size 64 \
    --d_model 256 \
    --n_heads 8 \
    --n_layers 4 \
    --d_ff 1024 \
    --lr 1e-4 \
    --dropout 0.1 \
    --vocab_size 50 \
    --max_len 50
```

```bash
# 超参数对比实验
for d_model in 128 256 512; do
    python main.py --d_model $d_model --epochs 30 --batch_size 32
    mv checkpoint_copy_task.pth checkpoint_dmodel_${d_model}.pth
    mv curves_copy_task.png curves_dmodel_${d_model}.png
done
```


## 实验报告模板

### 1. 实验目的
- 理解 Transformer 架构原理
- 掌握序列到序列建模方法
- 分析超参数对模型性能的影响

### 2. 数据集描述
- **数据集名称**: Multi30k 英德翻译数据集
- **数据规模**: 约 30,000 训练样本
- **数据格式**: 文本文件，每行一对翻译句子

### 3. 模型架构
- **编码器层数**: 6
- **解码器层数**: 6
- **d_model**: 512
- **n_heads**: 8
- **d_ff**: 2048

### 4. 实验结果
| 数据集 | Loss | Accuracy | PPL |
|--------|------|----------|-----|
| 训练集 | 1.23 | 0.78 | 3.42 |
| 验证集 | 1.85 | 0.65 | 6.36 |
| 测试集 | 1.92 | 0.63 | 6.82 |

### 5. 分析与讨论
- 模型成功学习了序列到序列映射
- 存在轻微过拟合现象
- 建议增加 dropout 或使用更大的数据集

### 6. 结论
- Transformer 架构在机器翻译任务上表现良好
- 超参数对模型性能有显著影响
- 适当的正则化有助于提升泛化能力

---

**实验完成时间**: 2024年
**项目版本**: v1.0.0
**参考来源**:
- Vaswani et al. (2017) "Attention Is All You Need"
- https://github.com/jadore801120/attention-is-all-you-need-pytorch