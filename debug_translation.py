"""
============================================================
翻译功能调试脚本
============================================================
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.translation_data import (
    load_multi30k,
    indices_to_sentence,
    sentence_to_indices,
    SPECIAL_TOKENS,
    tokenize_sentence,
)

data_dir = "data/multi30k"
print(f"加载数据集: {data_dir}")

train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(data_dir)

print("\n=== 调试信息 ===")

# 检查词表
print("\n1. 词表检查:")
print(f"英语词表大小: {len(src_vocab)}")
print(f"德语词表大小: {len(tgt_vocab)}")

# 检查测试数据集的第一个样本
print("\n2. 测试数据集第一个样本:")
src, tgt = test_dataset[0]
print(f"源序列 (索引): {src}")
print(f"目标序列 (索引): {tgt}")

# 转换为句子
src_sentence = indices_to_sentence(src, src_vocab)
tgt_sentence = indices_to_sentence(tgt, tgt_vocab)
print(f"源序列 (句子): {src_sentence}")
print(f"目标序列 (句子): {tgt_sentence}")

# 检查原始文件内容
print("\n3. 原始文件内容:")
with open(os.path.join(data_dir, "test.en"), "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
print(f"test.en 第一行: {first_line}")

# 检查分词结果
print(f"\n4. 分词结果:")
tokens = tokenize_sentence(first_line)
print(f"分词结果: {tokens}")

# 检查每个token是否在词表中
print("\n5. token在词表中的情况:")
for token in tokens:
    in_vocab = token in src_vocab
    print(f"  '{token}': {'在词表中' if in_vocab else '不在词表中'}")