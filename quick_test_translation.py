"""
============================================================
Transformer 翻译功能快速测试
============================================================
使用小模型快速验证翻译功能是否正常工作。
============================================================
"""

import torch
import random
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer import Transformer
from src.translation_data import (
    load_multi30k,
    indices_to_sentence,
    SPECIAL_TOKENS,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_translation():
    """测试翻译功能。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    print("\n" + "=" * 70)
    print("加载数据集...")
    print("=" * 70)

    data_dir = "data/multi30k"
    if os.path.exists(data_dir):
        train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(data_dir)
    else:
        print("Multi30k 数据集不存在，使用测试数据")
        data_dir = "data/test_translation"
        train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(data_dir)

    print(f"\n词表大小: 英语={len(src_vocab)}, 德语={len(tgt_vocab)}")
    print(f"数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

    config = {
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_layers": 2,
        "dropout": 0.1,
        "max_len": 100,
    }

    print("\n" + "=" * 70)
    print("创建模型...")
    print("=" * 70)
    print(f"配置: d_model={config['d_model']}, n_heads={config['n_heads']}, "
          f"n_layers={config['n_layers']}, d_ff={config['d_ff']}")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_encoder_layers=config["n_layers"],
        num_decoder_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
        dropout=config["dropout"],
        pad_idx=SPECIAL_TOKENS["<pad>"],
        share_embeddings=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    print("\n" + "=" * 70)
    print("测试翻译生成...")
    print("=" * 70)

    model.eval()
    test_samples = min(5, len(test_dataset))
    indices = np.random.choice(len(test_dataset), test_samples, replace=False)

    for idx in indices:
        src, tgt_expected = test_dataset[idx]
        src = src.unsqueeze(0).to(device)

        with torch.no_grad():
            generated = model.generate(
                src,
                max_len=50,
                bos_idx=SPECIAL_TOKENS["<bos>"],
                eos_idx=SPECIAL_TOKENS["<eos>"],
            )

        src_sentence = indices_to_sentence(src[0], src_vocab)
        tgt_sentence = indices_to_sentence(tgt_expected, tgt_vocab)
        gen_sentence = indices_to_sentence(generated[0], tgt_vocab)

        print(f"\n英语原文: {src_sentence}")
        try:
            print(f"德语译文: {tgt_sentence}")
        except UnicodeEncodeError:
            print(f"德语译文: {tgt_sentence.encode('ascii', errors='replace').decode('ascii')}")
        try:
            print(f"模型翻译: {gen_sentence}")
        except UnicodeEncodeError:
            print(f"模型翻译: {gen_sentence.encode('ascii', errors='replace').decode('ascii')}")

        if gen_sentence.strip() == tgt_sentence.strip():
            print("  [匹配 ✓]")
        else:
            print("  [不匹配 - 这是正常的，因为模型未训练]")

    print("\n" + "=" * 70)
    print("翻译功能测试完成!")
    print("=" * 70)
    print("\n提示:")
    print("  1. 当前模型未训练，翻译结果是随机的")
    print("  2. 要获得正确翻译，需要训练模型")
    print("  3. 运行训练命令: python train_translation_fixed.py --epochs 30")


if __name__ == "__main__":
    set_seed(42)
    test_translation()