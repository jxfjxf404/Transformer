"""
============================================================
Transformer 从零实现 - 完整 Transformer 模型
============================================================
整合编码器、解码器、嵌入层、位置编码和输出投影层,
构建端到端的 Transformer 序列到序列模型。

模型结构:
  1. 源语言嵌入 + 位置编码 → 编码器 → 编码器输出
  2. 目标语言嵌入 + 位置编码 → 解码器 → 输出投影 → softmax

数学原理 (完整前向传播):
  - 编码器:
    src_emb = InputEmbedding(src) * sqrt(d_model)
    src_emb = PositionalEncoding(src_emb)
    enc_out = TransformerEncoder(src_emb, src_mask)

  - 解码器:
    tgt_emb = InputEmbedding(tgt) * sqrt(d_model)
    tgt_emb = PositionalEncoding(tgt_emb)
    dec_out = TransformerDecoder(tgt_emb, enc_out, tgt_mask, src_mask)

  - 输出:
    logits = Linear(dec_out)  # 投影到词汇表维度

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - https://github.com/jadore801120/attention-is-all-you-need-pytorch
============================================================
"""

import torch
import torch.nn as nn
from typing import Optional

from .embedding import InputEmbedding, PositionalEncoding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    ============================================================
    完整 Transformer 模型 (End-to-End Transformer)
    ============================================================
    实现 Vaswani et al. (2017) 提出的完整 Transformer 架构。

    结构概览:
      ┌─────────────────────────────────────┐
      │           Output Probabilities       │
      │                ↑                     │
      │         Linear + Softmax             │
      │                ↑                     │
      │         TransformerDecoder           │
      │         ↑              ↑             │
      │  Target Embedding    Encoder Output  │
      │  + Positional Enc.        ↑          │
      │                  TransformerEncoder  │
      │                        ↑             │
      │                 Source Embedding      │
      │                 + Positional Enc.     │
      └─────────────────────────────────────┘

    关键超参数 (原始论文):
      - d_model = 512
      - n_heads = 8
      - num_layers = 6 (编码器和解码器各 6 层)
      - d_ff = 2048
      - dropout = 0.1

    与参考项目的差异:
      - 使用共享的权重矩阵进行输出投影 (与目标嵌入共享权重)
      - 模块化的 mask 创建逻辑, 在 forward 中自动处理
    ============================================================
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        share_embeddings: bool = False,
    ):
        """
        参数:
            src_vocab_size:       源语言词汇表大小
            tgt_vocab_size:       目标语言词汇表大小
            d_model:              模型维度
            n_heads:              注意力头数
            num_encoder_layers:   编码器层数
            num_decoder_layers:   解码器层数
            d_ff:                 前馈网络内部维度
            max_len:              最大序列长度
            dropout:              dropout 概率
            pad_idx:              填充 token 索引
            share_embeddings:     是否共享源和目标嵌入权重
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embedding = InputEmbedding(src_vocab_size, d_model, pad_idx)
        self.tgt_embedding = InputEmbedding(tgt_vocab_size, d_model, pad_idx)

        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding

        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, n_heads, d_ff, dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, n_heads, d_ff, dropout
        )

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        if share_embeddings:
            self.output_projection.weight = self.tgt_embedding.embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        """
        ============================================================
        参数初始化
        ============================================================
        使用 Xavier/Glorot 初始化, 这是 Transformer 的常见做法。
        对于偏置项使用零初始化。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码器前向传播: 将源序列编码为上下文表示。

        参数:
            src:      源序列, shape = (batch_size, src_seq_len)
            src_mask: 源序列 mask, shape = (batch_size, 1, 1, src_seq_len)

        返回:
            编码器输出, shape = (batch_size, src_seq_len, d_model)
        """
        src_emb = self.src_embedding(src)
        src_emb = self.positional_encoding(src_emb)
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        解码器前向传播: 基于编码器输出和目标序列生成输出表示。

        参数:
            tgt:             目标序列, shape = (batch_size, tgt_seq_len)
            encoder_output:  编码器输出, shape = (batch_size, src_seq_len, d_model)
            tgt_mask:        目标自注意力 mask, shape = (batch_size, 1, tgt_seq_len, tgt_seq_len)
            src_mask:        交叉注意力 mask, shape = (batch_size, 1, 1, src_seq_len)

        返回:
            解码器输出, shape = (batch_size, tgt_seq_len, d_model)
        """
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        return self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        完整前向传播。

        参数:
            src:      源序列, shape = (batch_size, src_seq_len)
            tgt:      目标序列, shape = (batch_size, tgt_seq_len)
            src_mask: 编码器 mask
            tgt_mask: 解码器自注意力 mask

        返回:
            输出 logits, shape = (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)
        return logits

    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ) -> torch.Tensor:
        """
        ============================================================
        自回归推理生成
        ============================================================
        逐个 token 生成目标序列, 每次使用已生成的序列作为解码器输入。

        参数:
            src:      源序列, shape = (batch_size, src_seq_len)
            src_mask: 编码器 mask
            max_len:  最大生成长度
            bos_idx:  起始 token 索引
            eos_idx:  结束 token 索引

        返回:
            生成的序列, shape = (batch_size, generated_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        encoder_output = self.encode(src, src_mask)

        generated = torch.full(
            (batch_size, 1), bos_idx, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            tgt_mask = self._create_tgt_mask(generated)
            decoder_output = self.decode(
                generated, encoder_output, tgt_mask, src_mask
            )

            logits = self.output_projection(decoder_output[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_idx).all():
                break

        return generated

    def _create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        创建目标序列的 subsequent mask。

        参数:
            tgt: 目标序列, shape = (batch_size, seq_len)

        返回:
            mask: shape = (1, 1, seq_len, seq_len)
        """
        from .mask import create_subsequent_mask
        seq_len = tgt.size(1)
        return create_subsequent_mask(seq_len).to(tgt.device)