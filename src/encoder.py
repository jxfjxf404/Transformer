"""
============================================================
Transformer 从零实现 - 编码器模块
============================================================
包含:
  1. EncoderLayer       - 编码器单层结构
  2. TransformerEncoder - 完整编码器 (多个 EncoderLayer 堆叠)

数学原理:
  编码器由 N 个相同的层堆叠而成, 每层包含两个子层:
    1. 多头自注意力子层 (Multi-Head Self-Attention)
    2. 位置-wise 前馈神经网络子层 (Position-wise FFN)

  每个子层后接残差连接 (Residual Connection) 和层归一化 (Layer Normalization):
    SubLayer(x) + x → LayerNorm

  即:
    x = LayerNorm(x + MultiHeadAttention(x, x, x, mask))
    x = LayerNorm(x + FeedForward(x))

  残差连接 (Add):
    将子层的输入直接加到输出上:
      output = Sublayer(x) + x
    作用: 缓解深层网络中的梯度消失问题, 使信息能直接跨层传播。

  层归一化 (Norm):
    对每个样本的特征维度进行归一化:
      LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
    其中 μ 和 σ² 是每个样本所有特征的均值和方差。
    作用: 稳定训练, 加速收敛。

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - https://github.com/jadore801120/attention-is-all-you-need-pytorch
============================================================
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """
    ============================================================
    编码器单层 (Encoder Layer)
    ============================================================
    包含两个子层:
      - 多头自注意力 (Self-Attention): 捕捉序列内部的依赖关系
      - 前馈神经网络 (FFN): 对每个位置的表示进行非线性变换

    每个子层后跟残差连接和层归一化。

    数学公式:
      x = LayerNorm(x + MultiHeadSelfAttention(x))
      x = LayerNorm(x + FeedForward(x))
    ============================================================
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        """
        参数:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff:    前馈网络内部维度
            dropout: dropout 概率
        """
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            x:    输入张量, shape = (batch_size, seq_len, d_model)
            mask: 自注意力掩码, shape = 可广播

        返回:
            输出张量, shape = (batch_size, seq_len, d_model)
        """
        attn_output = self.self_attention(x, x, x, mask=mask)
        x = self.norm_1(x + self.dropout_1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm_2(x + self.dropout_2(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """
    ============================================================
    完整 Transformer 编码器 (Transformer Encoder)
    ============================================================
    由 N 个相同的 EncoderLayer 堆叠而成。

    结构:
      Input → EncoderLayer_1 → EncoderLayer_2 → ... → EncoderLayer_N → Output

    每个 EncoderLayer 包含自注意力和 FFN 两个子层。
    堆叠多层使模型能逐层提取更高级别的抽象特征。

    与参考项目的差异:
      - 不包含嵌入层, 编码器只负责从已编码的向量序列进行特征提取
      - 嵌入层和位置编码在外部处理, 保持模块职责单一
    ============================================================
    """

    def __init__(self, num_layers: int = 6, d_model: int = 512,
                 n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        """
        参数:
            num_layers: 编码器层数
            d_model:    模型维度
            n_heads:    注意力头数
            d_ff:       前馈网络内部维度
            dropout:    dropout 概率
        """
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播: 依次通过所有编码器层。

        参数:
            x:    输入张量 (已嵌入 + 位置编码), shape = (batch_size, seq_len, d_model)
            mask: 自注意力掩码

        返回:
            编码器输出, shape = (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x