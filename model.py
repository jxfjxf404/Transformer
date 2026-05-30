"""
============================================================
Transformer 从零实现 - 模型定义
============================================================
重新导出核心 Transformer 模型及相关模块，方便直接使用。

用法:
  from model import Transformer

  model = Transformer(
      src_vocab_size=10000,
      tgt_vocab_size=10000,
      d_model=512,
      n_heads=8,
      num_encoder_layers=6,
      num_decoder_layers=6,
      d_ff=2048,
      dropout=0.1,
  )

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
============================================================
"""

from src.transformer import Transformer

from src.embedding import InputEmbedding, PositionalEncoding
from src.attention import ScaledDotProductAttention, MultiHeadAttention
from src.feedforward import PositionWiseFeedForward
from src.encoder import EncoderLayer, TransformerEncoder
from src.decoder import DecoderLayer, TransformerDecoder
from src.mask import (
    create_padding_mask,
    create_subsequent_mask,
    create_combined_mask,
    create_encoder_mask,
    create_decoder_cross_mask,
)
from src.trainer import Trainer
from src.data import CopyTaskDataset, ReverseTaskDataset
from src.translation_data import TranslationDataset

__all__ = [
    "Transformer",
    "InputEmbedding",
    "PositionalEncoding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionWiseFeedForward",
    "EncoderLayer",
    "TransformerEncoder",
    "DecoderLayer",
    "TransformerDecoder",
    "create_padding_mask",
    "create_subsequent_mask",
    "create_combined_mask",
    "create_encoder_mask",
    "create_decoder_cross_mask",
    "Trainer",
    "CopyTaskDataset",
    "ReverseTaskDataset",
    "TranslationDataset",
]