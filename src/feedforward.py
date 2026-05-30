"""
============================================================
Transformer 从零实现 - 前馈神经网络模块
============================================================
包含:
  PositionWiseFeedForward - 位置-wise 前馈神经网络

数学原理:
  FFN(x) = max(0, x·W1 + b1)·W2 + b2

  其中:
    W1 ∈ R^{d_model × d_ff}, b1 ∈ R^{d_ff}
    W2 ∈ R^{d_ff × d_model}, b2 ∈ R^{d_model}
    d_ff = 4 × d_model (原始论文中的超参数)

  每层由两个线性变换和一个 ReLU 激活函数组成。
  两个线性变换在序列的每个位置独立应用 (position-wise),
  即对每个 token 使用相同的参数, 但不同 token 间不共享信息。

  设计原因:
    1. 注意力层负责 token 之间的信息交互
    2. FFN 层负责对每个位置的表示进行非线性变换
    3. 两层结构增加了模型的表达能力 (类似 1×1 卷积)

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
============================================================
"""

import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """
    ============================================================
    位置-wise 前馈神经网络 (Position-wise Feed-Forward Network)
    ============================================================
    对序列中每个位置独立应用相同的两层全连接网络。

    数学原理:
      FFN(x) = ReLU(x·W1 + b1)·W2 + b2

      第一层将维度从 d_model 扩展到 d_ff (通常 4 倍),
      通过 ReLU 激活函数引入非线性。
      第二层将维度从 d_ff 压缩回 d_model。

    实现细节:
      - 使用 nn.Linear 实现两个线性变换
      - 使用 nn.ReLU 作为激活函数
      - 可以替换为其他激活函数 (如 GELU), 与原始论文保持一致使用 ReLU

    与参考项目的差异:
      - 将 FFN 作为独立模块, 便于在不同层中复用
      - 封装了完整的两个线性层和激活函数
    ============================================================
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        """
        参数:
            d_model: 模型维度 (输入/输出维度)
            d_ff:    前馈网络内部维度 (通常为 4 × d_model)
            dropout: dropout 概率
        """
        super(PositionWiseFeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入张量, shape = (batch_size, seq_len, d_model)

        返回:
            输出张量, shape = (batch_size, seq_len, d_model)
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x