"""
============================================================
Transformer 从零实现 - 数据处理模块
============================================================
包含:
  1. CopyTaskDataset     - 复制任务数据集 (用于验证模型正确性)
  2. ReverseTaskDataset  - 反转任务数据集
  3. TranslationDataset  - 通用翻译数据集基类
  4. create_dataloader   - 创建数据加载器
  5. build_vocab         - 构建词汇表

任务说明:
  复制任务 (Copy Task):
    输入序列:  ["A", "B", "C", "D"]
    输出序列:  ["A", "B", "C", "D"]
    目的: 验证 Transformer 能否学习序列到序列的映射

  反转任务 (Reverse Task):
    输入序列:  ["A", "B", "C", "D"]
    输出序列:  ["D", "C", "B", "A"]
    目的: 测试模型对序列顺序的感知能力

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
============================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import random
import numpy as np


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def build_vocab(tokens: List[str]) -> Dict[str, int]:
    """
    构建词汇表 (词到索引的映射)。

    特殊 token:
      <pad> (0): 填充 token
      <bos> (1): 句子起始 token
      <eos> (2): 句子结束 token
      <unk> (3): 未知词 token

    参数:
        tokens: 普通 token 列表

    返回:
        vocab: token 到索引的映射字典
    """
    vocab = dict(SPECIAL_TOKENS)
    for i, token in enumerate(tokens, start=len(SPECIAL_TOKENS)):
        vocab[token] = i
    return vocab


class CopyTaskDataset(Dataset):
    """
    ============================================================
    复制任务数据集 (Copy Task)
    ============================================================
    生成随机序列, 目标是与输入完全相同的序列。
    用于快速验证 Transformer 模型的基本功能。

    示例:
      输入:  [4, 7, 9, 2, 5]
      输出:  [4, 7, 9, 2, 5]

    每条序列:
      - 长度在 min_len 到 max_len 之间随机采样
      - token 值随机生成, 范围在 [vocab_start, vocab_end) 之间
      - 特殊 token 索引: <pad>=0, <bos>=1, <eos>=2, <unk>=3
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_len: int = 3,
        max_len: int = 20,
        vocab_size: int = 50,
    ):
        """
        参数:
            num_samples: 样本数量
            min_len:     最小序列长度
            max_len:     最大序列长度
            vocab_size:  词汇表大小 (包含特殊 token)
        """
        super().__init__()
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_idx = SPECIAL_TOKENS["<pad>"]
        self.bos_idx = SPECIAL_TOKENS["<bos>"]
        self.eos_idx = SPECIAL_TOKENS["<eos>"]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成一条样本。

        返回:
            src: 源序列, 不含 <bos>/<eos>, shape = (src_len,)
            tgt: 目标序列 (含 <bos> 和 <eos>), shape = (tgt_len,)
        """
        length = random.randint(self.min_len, self.max_len)
        sequence = torch.randint(
            low=len(SPECIAL_TOKENS),
            high=self.vocab_size,
            size=(length,),
            dtype=torch.long,
        )

        src = sequence.clone()
        tgt = torch.cat([
            torch.tensor([self.bos_idx], dtype=torch.long),
            sequence,
            torch.tensor([self.eos_idx], dtype=torch.long),
        ])
        return src, tgt


class ReverseTaskDataset(Dataset):
    """
    ============================================================
    反转任务数据集 (Reverse Task)
    ============================================================
    输入序列被反转后作为输出。
    用于测试模型对序列顺序的编码能力。

    示例:
      输入:  [4, 7, 9, 2, 5]
      输出:  [5, 2, 9, 7, 4]
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_len: int = 3,
        max_len: int = 15,
        vocab_size: int = 50,
    ):
        """
        参数:
            num_samples: 样本数量
            min_len:     最小序列长度
            max_len:     最大序列长度
            vocab_size:  词汇表大小
        """
        super().__init__()
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_idx = SPECIAL_TOKENS["<pad>"]
        self.bos_idx = SPECIAL_TOKENS["<bos>"]
        self.eos_idx = SPECIAL_TOKENS["<eos>"]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        length = random.randint(self.min_len, self.max_len)
        sequence = torch.randint(
            low=len(SPECIAL_TOKENS),
            high=self.vocab_size,
            size=(length,),
            dtype=torch.long,
        )

        src = sequence.clone()
        reversed_seq = sequence.flip(0)
        tgt = torch.cat([
            torch.tensor([self.bos_idx], dtype=torch.long),
            reversed_seq,
            torch.tensor([self.eos_idx], dtype=torch.long),
        ])
        return src, tgt


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ============================================================
    批次整理函数 (Collate Function)
    ============================================================
    将变长序列批次填充为相同长度。

    处理流程:
      1. 计算批次内源序列和目标序列的最大长度
      2. 将短序列填充到最大长度 (使用 pad_idx)
      3. 堆叠所有序列为批次张量

    参数:
        batch:   样本列表, 每个样本为 (src, tgt)
        pad_idx: 填充 token 索引

    返回:
        src_batch: 填充后的源序列, shape = (batch_size, max_src_len)
        tgt_batch: 填充后的目标序列, shape = (batch_size, max_tgt_len)
    """
    src_list, tgt_list = zip(*batch)

    max_src_len = max(s.size(0) for s in src_list)
    max_tgt_len = max(t.size(0) for t in tgt_list)

    src_batch = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt_len), pad_idx, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_batch[i, : src.size(0)] = src
        tgt_batch[i, : tgt.size(0)] = tgt

    return src_batch, tgt_batch


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    pad_idx: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """
    创建数据加载器。

    参数:
        dataset:     数据集实例
        batch_size:  批次大小
        shuffle:     是否打乱数据
        pad_idx:     填充 token 索引
        num_workers: 数据加载进程数

    返回:
        DataLoader 实例
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )