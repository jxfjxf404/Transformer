# Transformer 项目 - 快速开始指南

## 📋 数据集下载与预处理 - 问题解决方案

### 问题 1: Multi30k 数据集下载失败

#### 解决方案 A: 使用专用下载脚本（推荐）

```bash
# 运行数据下载脚本
python download_data.py
```

该脚本会：
- 自动检查文件是否已存在
- 显示下载进度
- 提供详细的错误信息
- 验证数据完整性

#### 解决方案 B: 使用测试数据集（快速验证）

如果 Multi30k 下载失败，可以使用内置的测试数据集快速验证代码：

```bash
# 生成测试数据集
python create_test_data.py

# 使用测试数据集训练
python train_translation.py --data_dir data/test_translation --epochs 50
```

测试数据集包含：
- 训练集：30 对英德句子
- 验证集：5 对英德句子
- 测试集：5 对英德句子

#### 解决方案 C: 手动下载 Multi30k

如果自动下载失败，可以手动下载：

1. 访问：https://github.com/multi30k/dataset/tree/master/data/task1

2. 下载以下文件到 `data/multi30k/` 目录：
   - `train.en`, `train.de` (训练集)
   - `val.en`, `val.de` (验证集)
   - `test_2016_flickr.en`, `test_2016_flickr.de` (测试集)

3. 重命名测试集文件：
   ```bash
   mv test_2016_flickr.en test.en
   mv test_2016_flickr.de test.de
   ```

---

## 🚀 完整训练流程

### 方案 1: 使用测试数据集（推荐用于快速验证）

```bash
# 步骤 1: 生成测试数据
python create_test_data.py

# 步骤 2: 训练模型
python train_translation.py \
    --data_dir data/test_translation \
    --epochs 50 \
    --batch_size 8 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --d_ff 512 \
    --lr 1e-3

# 步骤 3: 查看结果
# - checkpoint_translation.pth: 模型权重
# - history_translation.json: 训练历史
# - curves_translation.png: 训练曲线
```

### 方案 2: 使用 Multi30k 数据集（完整实验）

```bash
# 步骤 1: 下载数据集
python download_data.py

# 步骤 2: 训练模型
python train_translation.py \
    --epochs 30 \
    --batch_size 32 \
    --d_model 256 \
    --n_heads 8 \
    --n_layers 4 \
    --d_ff 1024 \
    --lr 1e-4

# 步骤 3: 查看结果
```

---

## 📊 实验任务检查清单

### ✅ 数据集下载与预处理

- [ ] 运行 `python download_data.py` 或 `python create_test_data.py`
- [ ] 验证数据文件存在：`data/multi30k/*.en` 和 `*.de`
- [ ] 检查数据集统计信息

### ✅ 词表构建

- [ ] 词表自动从训练数据构建
- [ ] 英语词表大小：约 10,000 词
- [ ] 德语词表大小：约 10,000 词

### ✅ 模型训练

- [ ] 运行训练脚本
- [ ] 观察训练日志
- [ ] 确认 Loss 下降

### ✅ 模型验证与测试

- [ ] 验证集评估
- [ ] 测试集评估
- [ ] 记录指标：Loss, Accuracy, Perplexity

### ✅ Loss 曲线绘制

- [ ] 自动生成 `curves_translation.png`
- [ ] 包含 4 个子图：Loss, Accuracy, PPL, LR

### ✅ 预测样例

- [ ] 训练完成后自动打印翻译示例
- [ ] 对比原文、译文、模型输出

### ✅ 模型效果分析

- [ ] 分析训练/验证/测试指标
- [ ] 检测过拟合/欠拟合
- [ ] 记录观察结果

### ✅ 参数量统计

- [ ] 训练开始时显示参数量
- [ ] 记录模型规模

### ✅ 超参数影响分析

- [ ] 运行对比实验
- [ ] 记录不同配置的结果
- [ ] 分析超参数影响

---

## 🔧 常见问题

### Q1: 下载失败怎么办？

**A**: 使用测试数据集快速验证：
```bash
python create_test_data.py
python train_translation.py --data_dir data/test_translation
```

### Q2: 训练太慢怎么办？

**A**: 使用较小的模型配置：
```bash
python train_translation.py \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --batch_size 16
```

### Q3: 内存不足怎么办？

**A**: 减小批次大小：
```bash
python train_translation.py --batch_size 8
```

### Q4: 如何使用 GPU？

**A**: 自动检测 GPU，无需额外配置。确保安装了 CUDA 版本的 PyTorch。

---

## 📝 实验报告模板

### 1. 实验目的
- 理解 Transformer 架构原理
- 掌握序列到序列建模方法
- 分析超参数对模型性能的影响

### 2. 数据集描述
- **数据集名称**: Multi30k 英德翻译数据集 / 测试数据集
- **数据规模**: 约 30,000 训练样本 / 30 训练样本
- **数据格式**: 文本文件，每行一对翻译句子

### 3. 模型架构
- **编码器层数**: 4
- **解码器层数**: 4
- **d_model**: 256
- **n_heads**: 8
- **d_ff**: 1024

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

## 📚 参考资源

- **论文**: Vaswani et al. (2017) "Attention Is All You Need"
- **数据集**: https://github.com/multi30k/dataset
- **参考项目**: https://github.com/jadore801120/attention-is-all-you-need-pytorch

---

**项目版本**: v1.0.0
**更新时间**: 2026-05-18