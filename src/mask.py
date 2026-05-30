"""
============================================================
Transformer 从零实现 - Mask 机制模块
============================================================
包含:
  1. create_padding_mask    - 创建 padding mask, 屏蔽填充位置的注意力
  2. create_subsequent_mask - 创建 subsequent (look-ahead) mask, 防止解码器看到未来信息

数学原理:
  - Padding Mask:
    对于变长序列, 短序列的尾部会被填充特殊 token (<pad>)。
    在计算注意力时, 填充位置不应参与计算。
    实现: 将 mask 位置对应的注意力分数设为 -inf, 使 softmax 后权重为 0。

  - Subsequent Mask (Look-ahead Mask):
    在解码器的自注意力中, 位置 i 只能关注自身及之前的位置 (1, 2, ..., i),
    不能关注未来位置 (i+1, ...), 保持自回归特性。
    实现: 使用上三角矩阵将未来位置屏蔽。

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
============================================================
"""

import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    ============================================================
    创建 padding mask
    ============================================================
    将输入序列中的填充位置标记为 True, 在注意力计算中被屏蔽。

    数学原理:
      对于 batch 中的每条序列, 填充部分不应贡献注意力分数。
      实现方式是将填充位置对应的注意力分数设为 -inf,
      经过 softmax 后这些位置的注意力权重变为 0。

    参数:
        seq:     输入 token 序列, shape = (batch_size, seq_len)
        pad_idx: 填充 token 的索引, 默认为 0

    返回:
        mask: shape = (batch_size, 1, 1, seq_len)
              True 表示需要屏蔽的位置, False 表示正常位置
    """
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_subsequent_mask(seq_len: int) -> torch.Tensor:
    """
    ============================================================
    创建 subsequent mask (look-ahead mask)
    ============================================================
    生成上三角掩码矩阵, 防止解码器在自注意力中看到未来位置的信息。

    数学原理:
      在自回归解码中, 位置 i 只能依赖位置 0, 1, ..., i-1 的信息。
      mask[i, j] = True 当 j > i (即位置 i 不能关注位置 j)。

      生成的上三角矩阵 (seq_len=4 为例):
        [[False,  True,  True,  True],   # pos 0 只能看到自己
         [False, False,  True,  True],   # pos 1 可以看到 0, 1
         [False, False, False,  True],   # pos 2 可以看到 0, 1, 2
         [False, False, False, False]]   # pos 3 可以看到全部

    参数:
        seq_len: 序列长度

    返回:
        mask: shape = (1, 1, seq_len, seq_len)
              True 表示需要屏蔽的位置
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def create_combined_mask(
    tgt_seq: torch.Tensor, pad_idx: int = 0
) -> torch.Tensor:
    """
    ============================================================
    创建组合 mask (用于解码器自注意力)
    ============================================================
    同时应用 padding mask 和 subsequent mask。
    任一条件满足即屏蔽该位置。

    参数:
        tgt_seq:  目标序列, shape = (batch_size, seq_len)
        pad_idx:  填充 token 的索引

    返回:
        combined_mask: shape = (batch_size, 1, seq_len, seq_len)
    """
    seq_len = tgt_seq.size(1)
    pad_mask = create_padding_mask(tgt_seq, pad_idx)
    sub_mask = create_subsequent_mask(seq_len).to(tgt_seq.device)
    combined_mask = pad_mask | sub_mask
    return combined_mask


def create_encoder_mask(src_seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    ============================================================
    创建编码器 mask (仅 padding mask)
    ============================================================
    编码器自注意力只需要 padding mask。

    参数:
        src_seq: 源序列, shape = (batch_size, seq_len)
        pad_idx: 填充 token 的索引

    返回:
        mask: shape = (batch_size, 1, 1, seq_len)
    """
    return create_padding_mask(src_seq, pad_idx)


def create_decoder_cross_mask(
    src_seq: torch.Tensor, pad_idx: int = 0
) -> torch.Tensor:
    """
    ============================================================
    创建解码器交叉注意力 mask
    ============================================================
    解码器的交叉注意力中, query 来自解码器, key/value 来自编码器输出。
    需要屏蔽编码器输出中的填充位置。

    参数:
        src_seq: 源序列, shape = (batch_size, seq_len)
        pad_idx: 填充 token 的索引

    返回:
        mask: shape = (batch_size, 1, 1, seq_len)
    """
    return create_padding_mask(src_seq, pad_idx)