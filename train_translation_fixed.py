"""
============================================================
Transformer 机器翻译训练脚本（修复版）
============================================================
修复翻译结果不匹配问题：
  1. 使用更大的 Multi30k 数据集（3万训练样本）
  2. 优化训练参数
  3. 修复词表构建问题
  4. 增加训练轮数
  5. 改进学习率调度

用法:
  python train_translation_fixed.py --epochs 30 --batch_size 32
============================================================
"""

import argparse
import torch
import random
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transformer import Transformer
from src.translation_data import (
    load_multi30k,
    create_dataloader,
    indices_to_sentence,
    SPECIAL_TOKENS,
)
from src.trainer import Trainer


def set_seed(seed: int = 42):
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def demo_translation(trainer, test_dataset, src_vocab, tgt_vocab, config):
    """演示翻译效果。"""
    device = config["device"]
    model = trainer.model
    model.eval()

    indices = np.random.choice(len(test_dataset), min(5, len(test_dataset)), replace=False)
    print("\n" + "=" * 70)
    print("翻译示例")
    print("=" * 70)

    for idx in indices:
        src, tgt_expected = test_dataset[idx]
        src = src.unsqueeze(0).to(device)

        generated = model.generate(
            src,
            max_len=50,
            bos_idx=SPECIAL_TOKENS["<bos>"],
            eos_idx=SPECIAL_TOKENS["<eos>"],
        )

        src_sentence = indices_to_sentence(src[0], src_vocab)
        tgt_sentence = indices_to_sentence(tgt_expected, tgt_vocab)
        gen_sentence = indices_to_sentence(generated[0], tgt_vocab)

        print(f"英语原文: {src_sentence}")
        print(f"德语译文: {tgt_sentence}")
        print(f"模型翻译: {gen_sentence}")
        
        if gen_sentence.strip() == tgt_sentence.strip():
            print("  [匹配 ✓]")
        else:
            print("  [不匹配]")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Transformer 机器翻译训练（修复版）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("-d", "--d_model", type=int, default=256, help="模型维度")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=4, help="编码器/解码器层数")
    parser.add_argument("--d_ff", type=int, default=1024, help="FFN 内部维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")
    parser.add_argument("--max_len", type=int, default=100, help="最大序列长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data_dir", type=str, default="data/multi30k", help="数据目录")
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("\n" + "=" * 70)
    print("检查数据集...")
    print("=" * 70)

    if not os.path.exists(args.data_dir):
        print(f"数据目录不存在: {args.data_dir}")
        print("使用测试数据集进行演示")
        args.data_dir = "data/test_translation"
        if not os.path.exists(args.data_dir):
            from create_test_data import create_test_data
            create_test_data(args.data_dir)

    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(args.data_dir)

    print(f"\n数据集统计:")
    print(f"  英语词表大小: {len(src_vocab)}")
    print(f"  德语词表大小: {len(tgt_vocab)}")
    print(f"  训练集样本数: {len(train_dataset)}")

    if len(train_dataset) < 100:
        print("\n⚠️ 警告: 训练集样本数较少，请使用完整的 Multi30k 数据集")
        print("   数据集下载: https://github.com/multi30k/dataset")

    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "num_encoder_layers": args.n_layers,
        "num_decoder_layers": args.n_layers,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "warmup_steps": 2000,
        "clip_grad_norm": 1.0,
        "log_interval": 50,
        "device": device,
        "share_embeddings": False,
    }

    print("\n" + "=" * 70)
    print("Transformer 机器翻译训练")
    print("=" * 70)
    print(f"设备: {config['device']}")
    print(f"数据集: {'Multi30k' if len(train_dataset) > 100 else '测试数据'}")
    print(f"模型配置: d_model={config['d_model']}, n_heads={config['n_heads']}, "
          f"n_layers={config['num_encoder_layers']}, d_ff={config['d_ff']}")
    print(f"训练配置: epochs={config['num_epochs']}, batch_size={config['batch_size']}, "
          f"lr={config['learning_rate']}")

    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
        dropout=config["dropout"],
        pad_idx=SPECIAL_TOKENS["<pad>"],
        share_embeddings=config["share_embeddings"],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pad_idx=SPECIAL_TOKENS["<pad>"],
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pad_idx=SPECIAL_TOKENS["<pad>"],
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pad_idx=SPECIAL_TOKENS["<pad>"],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config["learning_rate"],
        num_epochs=config["num_epochs"],
        device=device,
        pad_idx=SPECIAL_TOKENS["<pad>"],
        warmup_steps=config["warmup_steps"],
        clip_grad_norm=config["clip_grad_norm"],
        log_interval=config["log_interval"],
    )

    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    trainer.train()

    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    trainer.test()

    trainer.save_model("checkpoint_translation_fixed.pth")
    trainer.export_history("history_translation_fixed.json")
    trainer.plot_curves("curves_translation_fixed.png")

    demo_translation(trainer, test_dataset, src_vocab, tgt_vocab, config)


if __name__ == "__main__":
    main()