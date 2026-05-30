"""
============================================================
Transformer 从零实现 - 包初始化
============================================================
基于 "Attention Is All You Need" (Vaswani et al., 2017) 的完整实现。

参考项目:
  - https://github.com/jadore801120/attention-is-all-you-need-pytorch
  - https://github.com/aladdinpersson/Machine-Learning-Collection

差异化实现:
  - 完全模块化设计, 每个核心组件独立封装
  - 使用 register_buffer 管理位置编码
  - 支持 embedding 权重共享
  - 支持自回归推理生成 (generate)
  - 内置多种验证任务 (Copy, Reverse)
  - 完善的可视化和日志系统
============================================================
"""

from .embedding import InputEmbedding, PositionalEncoding
from .attention import ScaledDotProductAttention, MultiHeadAttention
from .feedforward import PositionWiseFeedForward
from .encoder import EncoderLayer, TransformerEncoder
from .decoder import DecoderLayer, TransformerDecoder
from .transformer import Transformer
from .mask import (
    create_padding_mask,
    create_subsequent_mask,
    create_combined_mask,
    create_encoder_mask,
    create_decoder_cross_mask,
)
from .trainer import Trainer
from .data import (
    CopyTaskDataset,
    ReverseTaskDataset,
    create_dataloader,
    SPECIAL_TOKENS,
    build_vocab,
)

__version__ = "1.0.0"
__author__ = "Noir Setup TUI Team"