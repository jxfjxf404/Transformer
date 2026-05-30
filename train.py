"""
============================================================
Transformer 从零实现 - 训练脚本
============================================================
使用 Multi30k 英德翻译数据集训练 Transformer 模型。

用法:
  python train.py                                    # 默认参数训练
  python train.py --epochs 30 --batch_size 32        # 自定义训练轮数和批次大小
  python train.py --d_model 256 --n_heads 8 --n_layers 4  # 自定义模型参数
  python train.py --data_dir data/test_translation   # 使用测试数据集快速验证
  python train.py --help                             # 查看所有参数

参考:
  - Multi30k 数据集: https://github.com/multi30k/dataset
  - "Attention Is All You Need" (Vaswani et al., 2017)
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
from src.experiment_utils import count_parameters


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_multi30k(data_dir: str = "data/multi30k"):
    os.makedirs(data_dir, exist_ok=True)

    required_files = ["train.en", "train.de", "val.en", "val.de", "test.en", "test.de"]
    all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)

    if all_exist:
        print("数据集已存在，跳过下载")
        return

    print("=" * 60)
    print("Multi30k 数据集下载")
    print("=" * 60)
    print("正在下载数据集，这可能需要几分钟...")
    print("如果下载失败，请运行: python download_data.py")
    print("=" * 60)

    urls = [
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/train.en", "train.en"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/train.de", "train.de"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/val.en", "val.en"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/val.de", "val.de"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/test_2016_flickr.en", "test.en"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/test_2016_flickr.de", "test.de"),
    ]

    try:
        import urllib.request

        for url, filename in urls:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"下载 {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                except Exception as e:
                    print(f"下载 {filename} 失败: {e}")
                    print("请运行: python download_data.py 手动下载")
                    raise

        print("数据集下载完成!")
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请运行以下命令手动下载数据集:")
        print("  python download_data.py")
        print("\n或访问: https://github.com/multi30k/dataset/tree/master/data/task1")
        raise


@torch.no_grad()
def demo_translation(trainer, test_dataset, src_vocab, tgt_vocab, config):
    device = config["device"]
    model = trainer.model
    model.eval()

    indices = np.random.choice(len(test_dataset), min(5, len(test_dataset)), replace=False)
    print("\n" + "=" * 60)
    print("翻译示例")
    print("=" * 60)

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
        print(f"德语参考: {tgt_sentence}")
        print(f"模型翻译: {gen_sentence}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Transformer 机器翻译训练")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
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

    download_multi30k(args.data_dir)

    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(args.data_dir)

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
        "warmup_steps": 4000,
        "clip_grad_norm": 1.0,
        "log_interval": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "share_embeddings": False,
    }

    print("\n" + "=" * 60)
    print("Transformer 机器翻译训练")
    print("=" * 60)
    print(f"设备: {config['device']}")
    print(f"数据集: Multi30k 英德翻译")
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

    total_params, params_by_module = count_parameters(model)
    print(f"\n模型总参数量: {total_params:,}")
    print("各模块参数量:")
    for name, params in sorted(params_by_module.items(), key=lambda x: -x[1]):
        if params > 0:
            print(f"  {name}: {params:,}")

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
        device=config["device"],
        pad_idx=SPECIAL_TOKENS["<pad>"],
        warmup_steps=config["warmup_steps"],
        clip_grad_norm=config["clip_grad_norm"],
        log_interval=config["log_interval"],
    )

    trainer.train()
    trainer.test()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    trainer.save_model("checkpoints/checkpoint_translation.pth")
    trainer.export_history("results/history_translation.json")
    trainer.plot_curves("figures/curves_translation.png")

    demo_translation(trainer, test_dataset, src_vocab, tgt_vocab, config)


if __name__ == "__main__":
    main()