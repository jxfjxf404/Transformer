"""
============================================================
Transformer 从零实现 - 注意力机制模块
============================================================
包含:
  1. ScaledDotProductAttention - 缩放点积注意力机制
  2. MultiHeadAttention        - 多头注意力机制

数学原理:
  - 缩放点积注意力:
    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    其中:
      Q ∈ R^{n_q × d_k} : 查询矩阵 (Query)
      K ∈ R^{n_k × d_k} : 键矩阵   (Key)
      V ∈ R^{n_k × d_v} : 值矩阵   (Value)
      d_k              : 键向量的维度 (缩放因子)

    计算步骤:
      1. 计算注意力分数: scores = Q·K^T / sqrt(d_k)
      2. (可选) 应用 mask: scores = masked_fill(scores, mask, -1e9)
      3. 注意力权重: attn_weights = softmax(scores)
      4. (可选) 应用 dropout: attn_weights = dropout(attn_weights)
      5. 上下文向量: output = attn_weights · V

  - 多头注意力:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
    其中 head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)

    将 d_model 维度的 Q, K, V 投影到 h 个低维子空间 (每头 d_k = d_model / h),
    在各子空间中并行计算注意力, 最后拼接并通过线性变换映射回 d_model 维度。

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - https://github.com/jadore801120/attention-is-all-you-need-pytorch
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    ============================================================
    缩放点积注意力 (Scaled Dot-Product Attention)
    ============================================================
    实现原始的缩放点积注意力机制, 是 Transformer 的核心计算单元。

    数学原理:
      Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

      1. 点积 Q·K^T 计算查询与每个键的相似度
      2. 除以 sqrt(d_k) 进行缩放, 防止 d_k 过大时点积值过大,
         导致 softmax 进入梯度极小的饱和区
      3. softmax 将相似度归一化为概率分布 (注意力权重)
      4. 加权求和得到每个查询位置的上下文表示

    实现细节:
      - 支持可选的 mask 机制 (padding mask 或 subsequent mask)
      - mask 值为 True 的位置会被填充为 -inf (softmax 后权重为 0)
      - 支持返回注意力权重, 用于可视化分析

    与参考项目的差异:
      - 将 mask 的填充值设为 -1e9 (而非 -inf), 避免数值问题
      - 返回注意力权重用于后续分析
    ============================================================
    """

    def __init__(self, dropout: float = 0.1):
        """
        参数:
            dropout: 注意力权重 dropout 概率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        前向传播。

        参数:
            query:  查询矩阵, shape = (batch_size, n_heads, seq_len_q, d_k)
            key:    键矩阵,   shape = (batch_size, n_heads, seq_len_k, d_k)
            value:  值矩阵,   shape = (batch_size, n_heads, seq_len_k, d_v)
            mask:   注意力掩码, shape = 可广播到 (batch_size, n_heads, seq_len_q, seq_len_k)
                    True 表示需要屏蔽的位置
            return_attention: 是否返回注意力权重

        返回:
            output:        上下文向量, shape = (batch_size, n_heads, seq_len_q, d_v)
            attention:     (可选) 注意力权重, shape = (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        self.scale = math.sqrt(d_k)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        if return_attention:
            return output, attention_weights
        return output


class MultiHeadAttention(nn.Module):
    """
    ============================================================
    多头注意力 (Multi-Head Attention)
    ============================================================
    将输入投影到多个子空间, 并行计算注意力, 使模型能关注不同表示子空间的信息。

    数学原理:
      MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
      head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)

      其中:
        W^Q_i ∈ R^{d_model × d_k}    : 第 i 个头的查询投影矩阵
        W^K_i ∈ R^{d_model × d_k}    : 第 i 个头的键投影矩阵
        W^V_i ∈ R^{d_model × d_v}    : 第 i 个头的值投影矩阵
        W^O   ∈ R^{h·d_v × d_model}  : 输出投影矩阵
        d_k = d_v = d_model / h      : 每个头的维度

      多头设计的好处:
        1. 不同头可以关注不同的位置/特征, 类似于 CNN 的多通道
        2. 联合使用多个注意力头比单一注意力头效果更好
        3. 可以在不同子空间中捕捉不同类型的关系

    实现细节:
      - 使用线性变换批量投影 Q, K, V
      - 分裂 (split): 将 d_model 维度拆分为 (n_heads, d_k)
      - 转置 (transpose): 将头维度移到 batch 维度后, 便于并行计算
      - 拼接 (concat): 将多头输出拼回 d_model 维度
      - 最终线性投影得到多头注意力的输出

    与参考项目的差异:
      - 分离 mask 创建逻辑到独立的 mask 模块 (src/mask.py)
      - 支持同时返回注意力权重用于分析
    ============================================================
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        """
        参数:
            d_model: 模型总维度
            n_heads: 注意力头数
            dropout: dropout 概率
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入张量分裂为多头形式。

        输入:  (batch_size, seq_len, d_model)
        输出:  (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头输出拼接回原始维度。

        输入:  (batch_size, n_heads, seq_len, d_k)
        输出:  (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        前向传播。

        参数:
            query: shape = (batch_size, seq_len_q, d_model)
            key:   shape = (batch_size, seq_len_k, d_model)
            value: shape = (batch_size, seq_len_v, d_model)
            mask:  注意力掩码, shape = 可广播
            return_attention: 是否返回注意力权重

        返回:
            output:     多头注意力输出, shape = (batch_size, seq_len_q, d_model)
            attention:  (可选) 注意力权重
        """
        residual_query, residual_key, residual_value = query, key, value

        query = self._split_heads(self.W_Q(query))
        key = self._split_heads(self.W_K(key))
        value = self._split_heads(self.W_V(value))

        attn_out = self.attention(query, key, value, mask=mask, return_attention=return_attention)

        if return_attention:
            context, attention_weights = attn_out
        else:
            context = attn_out

        output = self._concat_heads(context)
        output = self.W_O(output)

        if return_attention:
            return output, attention_weights
        return output