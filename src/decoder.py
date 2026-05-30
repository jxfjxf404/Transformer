"""
============================================================
Transformer 从零实现 - 解码器模块
============================================================
包含:
  1. DecoderLayer       - 解码器单层结构
  2. TransformerDecoder - 完整解码器 (多个 DecoderLayer 堆叠)

数学原理:
  解码器由 N 个相同的层堆叠而成, 每层包含三个子层:
    1. 掩码多头自注意力子层 (Masked Multi-Head Self-Attention)
       - 使用 subsequent mask 防止关注未来位置
    2. 编码器-解码器注意力子层 (Cross-Attention)
       - Query 来自解码器, Key/Value 来自编码器输出
    3. 位置-wise 前馈神经网络子层 (Position-wise FFN)

  每个子层后接残差连接和层归一化:
    x = LayerNorm(x + MaskedSelfAttention(x))
    x = LayerNorm(x + CrossAttention(x, enc_out, enc_out))
    x = LayerNorm(x + FeedForward(x))

  编码器-解码器注意力 (Cross-Attention):
    解码器的每个位置可以关注编码器输出的所有位置,
    使解码器能利用源序列的完整上下文信息。

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


class DecoderLayer(nn.Module):
    """
    ============================================================
    解码器单层 (Decoder Layer)
    ============================================================
    包含三个子层:
      - 掩码多头自注意力: 防止看到未来位置
      - 编码器-解码器交叉注意力: 融合编码器输出的上下文信息
      - 前馈神经网络: 非线性变换

    每个子层后跟残差连接和层归一化。

    数学公式:
      x = LayerNorm(x + MaskedMultiHeadSelfAttention(x))
      x = LayerNorm(x + MultiHeadCrossAttention(x, enc_out, enc_out))
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
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            x:               解码器输入, shape = (batch_size, tgt_seq_len, d_model)
            encoder_output:  编码器输出, shape = (batch_size, src_seq_len, d_model)
            self_attn_mask:  自注意力掩码 (subsequent + padding mask)
            cross_attn_mask: 交叉注意力掩码 (encoder padding mask)

        返回:
            输出张量, shape = (batch_size, tgt_seq_len, d_model)
        """
        attn_output = self.self_attention(x, x, x, mask=self_attn_mask)
        x = self.norm_1(x + self.dropout_1(attn_output))

        cross_output = self.cross_attention(
            x, encoder_output, encoder_output, mask=cross_attn_mask
        )
        x = self.norm_2(x + self.dropout_2(cross_output))

        ff_output = self.feed_forward(x)
        x = self.norm_3(x + self.dropout_3(ff_output))

        return x


class TransformerDecoder(nn.Module):
    """
    ============================================================
    完整 Transformer 解码器 (Transformer Decoder)
    ============================================================
    由 N 个相同的 DecoderLayer 堆叠而成。

    结构:
      Input → DecoderLayer_1 → DecoderLayer_2 → ... → DecoderLayer_N → Output

    每个 DecoderLayer 包含:
      1. 掩码自注意力: 捕捉目标序列内部的依赖 (不能看到未来)
      2. 交叉注意力: 融合编码器的输出信息
      3. 前馈网络: 非线性变换

    与参考项目的差异:
      - 不包含嵌入层, 只负责从已编码的目标序列进行解码
      - 交叉注意力的 key/value 始终来自编码器输出
    ============================================================
    """

    def __init__(self, num_layers: int = 6, d_model: int = 512,
                 n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        """
        参数:
            num_layers: 解码器层数
            d_model:    模型维度
            n_heads:    注意力头数
            d_ff:       前馈网络内部维度
            dropout:    dropout 概率
        """
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播: 依次通过所有解码器层。

        参数:
            x:               目标序列嵌入, shape = (batch_size, tgt_seq_len, d_model)
            encoder_output:  编码器输出, shape = (batch_size, src_seq_len, d_model)
            self_attn_mask:  自注意力掩码
            cross_attn_mask: 交叉注意力掩码

        返回:
            解码器输出, shape = (batch_size, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        return x