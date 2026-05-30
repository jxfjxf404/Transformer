"""
============================================================
Transformer 从零实现 - 嵌入层模块
============================================================
包含:
  1. InputEmbedding  - 词嵌入层, 将输入 token 索引转换为稠密向量
  2. PositionalEncoding - 正弦位置编码, 为嵌入向量注入位置信息

数学原理:
  - 词嵌入: E ∈ R^{vocab_size × d_model}, 通过查表将离散 token 映射为连续向量
  - 位置编码: PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
              PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    其中 pos 是位置索引, i 是维度索引
  - 最终输入: X = Embedding(token) * sqrt(d_model) + PE
    缩放因子 sqrt(d_model) 用于防止嵌入值过小导致梯度消失

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - https://github.com/jadore801120/attention-is-all-you-need-pytorch
============================================================
"""

import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    """
    ============================================================
    词嵌入层 (Input Embedding Layer)
    ============================================================
    将离散的 token 索引序列转换为连续的稠密向量表示。

    数学原理:
      对于输入序列中的每个 token 索引 i, 通过嵌入矩阵 W ∈ R^{vocab_size × d_model}
      查找第 i 行得到嵌入向量 x ∈ R^{d_model}。

    实现细节:
      - 使用 PyTorch 的 nn.Embedding 作为底层实现
      - 对嵌入结果乘以 sqrt(d_model), 使得嵌入向量的方差与位置编码的方差
        处于同一量级, 便于两者相加
      - 该缩放策略在原始论文中被采用

    与参考项目的差异:
      - 将 Embedding 和缩放逻辑封装为独立可复用模块
      - 在 forward 中直接进行缩放, 避免外部重复代码
    ============================================================
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        """
        参数:
            vocab_size: 词汇表大小
            d_model:   模型维度 (嵌入向量的维度)
            padding_idx: 填充 token 的索引, 其嵌入始终为零向量
        """
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入 token 索引, shape = (batch_size, seq_len)

        返回:
            嵌入向量, shape = (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    ============================================================
    正弦位置编码 (Sinusoidal Positional Encoding)
    ============================================================
    为序列中的每个位置生成唯一的编码向量, 使模型能感知 token 的顺序。

    数学原理:
      对于位置 pos 和维度 i:
        PE(pos, 2i)   = sin(pos / 10000^{2i / d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})

      其中:
        - pos ∈ [0, max_len) 表示序列中的位置
        - i   ∈ [0, d_model/2) 表示维度索引
        - 10000 是温度参数, 控制频率衰减速度

    为什么使用正弦/余弦?
      1. 不同频率的正弦/余弦函数使每个位置产生唯一编码
      2. 对于固定偏移 k, PE(pos+k) 可表示为 PE(pos) 的线性函数,
         有助于模型学习相对位置关系
      3. 值域在 [-1, 1] 之间, 数值稳定

    实现细节:
      - 预计算所有位置 (0 到 max_len-1) 的编码并注册为 buffer
      - buffer 不会被优化器更新, 但会随模型一起保存/加载
      - 前向传播时仅取前 seq_len 个位置的编码

    与参考项目的差异:
      - 使用 register_buffer 确保编码随模型持久化
      - 将位置编码作为独立模块, 便于替换为可学习位置编码
    ============================================================
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        参数:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout 概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = self._compute_positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def _compute_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """
        计算正弦位置编码矩阵。

        参数:
            max_len: 最大序列长度
            d_model: 模型维度

        返回:
            pe: shape = (1, max_len, d_model), 位置编码矩阵
        """
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 将位置编码加到输入嵌入上。

        参数:
            x: 输入嵌入, shape = (batch_size, seq_len, d_model)

        返回:
            添加位置编码后的输出, shape = (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)