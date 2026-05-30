"""
============================================================
Transformer 从零实现 - 翻译数据集模块
============================================================
包含:
  1. TranslationDataset - 翻译数据集类
  2. build_vocab_from_file - 从文件构建词表
  3. tokenize_sentence - 句子分词
  4. load_multi30k - 加载 Multi30k 数据集

参考:
  - Multi30k 数据集: https://github.com/multi30k/dataset
============================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import os
import re


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def tokenize_sentence(sentence: str) -> List[str]:
    """
    简单的分词函数。

    参数:
        sentence: 输入句子

    返回:
        分词后的 token 列表
    """
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([.!?])", r" \1", sentence)
    sentence = re.sub(r"[^a-zA-ZäöüÄÖÜßàáâãäåçèéêëìíîïðñòóôõöùúûüýþÿ.!? ]", "", sentence)
    tokens = sentence.split()
    return tokens


def build_vocab_from_file(filepath: str, max_size: int = 30000) -> Dict[str, int]:
    """
    从文件构建词表。

    参数:
        filepath: 文本文件路径
        max_size: 词表最大大小

    返回:
        vocab: 词到索引的映射
    """
    word_counts = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize_sentence(line)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    sorted_words = sorted_words[:max_size - len(SPECIAL_TOKENS)]

    vocab = dict(SPECIAL_TOKENS)
    for i, (word, _) in enumerate(sorted_words, start=len(SPECIAL_TOKENS)):
        vocab[word] = i

    return vocab


def sentence_to_indices(sentence: str, vocab: Dict[str, int]) -> List[int]:
    """
    将句子转换为索引序列。

    参数:
        sentence: 输入句子
        vocab: 词表

    返回:
        索引列表
    """
    tokens = tokenize_sentence(sentence)
    indices = []
    for token in tokens:
        indices.append(vocab.get(token, SPECIAL_TOKENS["<unk>"]))
    return indices


class TranslationDataset(Dataset):
    """
    ============================================================
    翻译数据集 (Translation Dataset)
    ============================================================
    支持英德翻译等机器翻译任务。
    """

    def __init__(
        self,
        src_filepath: str,
        tgt_filepath: str,
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
    ):
        """
        参数:
            src_filepath: 源语言文件路径
            tgt_filepath: 目标语言文件路径
            src_vocab: 源语言词表
            tgt_vocab: 目标语言词表
        """
        super().__init__()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_idx = SPECIAL_TOKENS["<pad>"]
        self.bos_idx = SPECIAL_TOKENS["<bos>"]
        self.eos_idx = SPECIAL_TOKENS["<eos>"]

        self.src_sentences = []
        self.tgt_sentences = []

        with open(src_filepath, "r", encoding="utf-8") as src_f, \
             open(tgt_filepath, "r", encoding="utf-8") as tgt_f:

            for src_line, tgt_line in zip(src_f, tgt_f):
                src_indices = sentence_to_indices(src_line.strip(), src_vocab)
                tgt_indices = sentence_to_indices(tgt_line.strip(), tgt_vocab)

                if len(src_indices) > 0 and len(tgt_indices) > 0:
                    self.src_sentences.append(src_indices)
                    self.tgt_sentences.append(tgt_indices)

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回源序列和目标序列。

        返回:
            src: 源序列 (不含 <bos>/<eos>)
            tgt: 目标序列 (含 <bos> 和 <eos>)
        """
        src = torch.tensor(self.src_sentences[idx], dtype=torch.long)
        tgt = torch.cat([
            torch.tensor([self.bos_idx], dtype=torch.long),
            torch.tensor(self.tgt_sentences[idx], dtype=torch.long),
            torch.tensor([self.eos_idx], dtype=torch.long),
        ])
        return src, tgt


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批次整理函数。

    参数:
        batch: 样本列表
        pad_idx: 填充索引

    返回:
        src_batch: 填充后的源序列
        tgt_batch: 填充后的目标序列
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
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        pad_idx: 填充索引
        num_workers: 工作线程数

    返回:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )


def load_multi30k(data_dir: str = "data/multi30k") -> Tuple[
    TranslationDataset,
    TranslationDataset,
    TranslationDataset,
    Dict[str, int],
    Dict[str, int],
]:
    """
    加载 Multi30k 数据集。

    参数:
        data_dir: 数据目录

    返回:
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        src_vocab: 源语言词表 (英语)
        tgt_vocab: 目标语言词表 (德语)
    """
    train_en = os.path.join(data_dir, "train.en")
    train_de = os.path.join(data_dir, "train.de")
    val_en = os.path.join(data_dir, "val.en")
    val_de = os.path.join(data_dir, "val.de")
    test_en = os.path.join(data_dir, "test.en")
    test_de = os.path.join(data_dir, "test.de")

    src_vocab = build_vocab_from_file(train_en)
    tgt_vocab = build_vocab_from_file(train_de)

    train_dataset = TranslationDataset(train_en, train_de, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_en, val_de, src_vocab, tgt_vocab)
    test_dataset = TranslationDataset(test_en, test_de, src_vocab, tgt_vocab)

    print(f"加载完成!")
    print(f"  英语词表大小: {len(src_vocab)}")
    print(f"  德语词表大小: {len(tgt_vocab)}")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab


def indices_to_sentence(indices, vocab: Dict[str, int]) -> str:
    """
    将索引序列转换为句子。

    参数:
        indices: 索引序列 (可以是 torch.Tensor 或 list)
        vocab: 词表

    返回:
        句子字符串
    """
    idx_to_word = {idx: word for word, idx in vocab.items()}
    tokens = []
    
    # 确保 indices 是列表形式
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    
    for idx in indices:
        # 如果是张量，提取标量值
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        if idx == SPECIAL_TOKENS["<pad>"]:
            continue
        if idx == SPECIAL_TOKENS["<bos>"]:
            continue
        if idx == SPECIAL_TOKENS["<eos>"]:
            break
        tokens.append(idx_to_word.get(idx, "<unk>"))
    return " ".join(tokens)