"""
============================================================
Transformer 从零实现 - 主程序入口
============================================================
运行训练、验证和测试的完整流程。

用法:
  python main.py                  # 运行复制任务训练
  python main.py --task reverse   # 运行反转任务训练
  python main.py --test           # 仅运行测试
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
from src.data import (
    CopyTaskDataset,
    ReverseTaskDataset,
    create_dataloader,
    SPECIAL_TOKENS,
)
from src.trainer import Trainer


def set_seed(seed: int = 42):
    """设置随机种子, 确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(vocab_size: int, config: dict) -> Transformer:
    """
    创建 Transformer 模型。

    参数:
        vocab_size: 词汇表大小
        config:     模型配置字典

    返回:
        Transformer 模型实例
    """
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
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
    return model


def run_copy_task(config: dict):
    """运行复制任务 (Copy Task) 的训练和评估。"""
    print("\n" + "=" * 60)
    print("任务: 复制任务 (Copy Task)")
    print("=" * 60)
    print("源序列和目标序列完全相同, 验证模型的基本序列到序列映射能力。")

    train_dataset = CopyTaskDataset(
        num_samples=config["train_samples"],
        min_len=config["data_min_len"],
        max_len=config["data_max_len"],
        vocab_size=config["vocab_size"],
    )
    val_dataset = CopyTaskDataset(
        num_samples=config["val_samples"],
        min_len=config["data_min_len"],
        max_len=config["data_max_len"],
        vocab_size=config["vocab_size"],
    )
    test_dataset = CopyTaskDataset(
        num_samples=config["test_samples"],
        min_len=config["data_min_len"],
        max_len=config["data_max_len"],
        vocab_size=config["vocab_size"],
    )

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

    model = create_model(config["vocab_size"], config)

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

    trainer.save_model("checkpoint_copy_task.pth")
    trainer.export_history("history_copy_task.json")
    trainer.plot_curves("curves_copy_task.png")

    print("\n复制任务的生成示例:")
    demo_generate(trainer, test_dataset, config)


def run_reverse_task(config: dict):
    """运行反转任务 (Reverse Task) 的训练和评估。"""
    print("\n" + "=" * 60)
    print("任务: 反转任务 (Reverse Task)")
    print("=" * 60)
    print("目标序列是源序列的反转, 测试模型对序列顺序的感知和重组能力。")

    train_dataset = ReverseTaskDataset(
        num_samples=config["train_samples"],
        min_len=config["data_min_len"],
        max_len=config["data_max_len"],
        vocab_size=config["vocab_size"],
    )
    val_dataset = ReverseTaskDataset(
        num_samples=config["val_samples"],
        min_len=config["data_min_len"],
        max_len=config["data_max_len"],
        vocab_size=config["vocab_size"],
    )
    test_dataset = ReverseTaskDataset(
        num_samples=config["test_samples"],
        min_len=config["data_min_len"],
        max_len=config["data_max_len"],
        vocab_size=config["vocab_size"],
    )

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

    model = create_model(config["vocab_size"], config)

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

    trainer.save_model("checkpoint_reverse_task.pth")
    trainer.export_history("history_reverse_task.json")
    trainer.plot_curves("curves_reverse_task.png")

    print("\n反转任务的生成示例:")
    demo_generate(trainer, test_dataset, config)


@torch.no_grad()
def demo_generate(trainer, dataset, config):
    """演示自回归生成效果。"""
    device = config["device"]
    model = trainer.model
    model.eval()

    indices = np.random.choice(len(dataset), min(3, len(dataset)), replace=False)
    for idx in indices:
        src, tgt_expected = dataset[idx]
        src = src.unsqueeze(0).to(device)

        generated = model.generate(
            src,
            max_len=config["max_len"] + 3,
            bos_idx=SPECIAL_TOKENS["<bos>"],
            eos_idx=SPECIAL_TOKENS["<eos>"],
        )

        src_str = "[" + ", ".join(str(x) for x in src[0].tolist()) + "]"
        gen_str = "[" + ", ".join(str(x) for x in generated[0].tolist()) + "]"
        tgt_str = "[" + ", ".join(str(x) for x in tgt_expected.tolist()) + "]"

        print(f"  源序列:    {src_str}")
        print(f"  目标序列:  {tgt_str}")
        print(f"  生成序列:  {gen_str}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Transformer 从零实现 - 训练与评估")
    parser.add_argument(
        "--task",
        type=str,
        default="copy",
        choices=["copy", "reverse"],
        help="训练任务类型 (默认: copy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备 (默认: 自动检测)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数 (默认: 30)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率 (默认: 1e-4)")
    parser.add_argument("-d", "--d_model", type=int, default=128, help="模型维度 (默认: 128)")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数 (默认: 8)")
    parser.add_argument("--d_ff", type=int, default=512, help="FFN 内部维度 (默认: 512)")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="编码器/解码器层数 (默认: 3)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率 (默认: 0.1)")
    parser.add_argument("--vocab_size", type=int, default=50, help="词汇表大小 (默认: 50)")
    parser.add_argument("--max_len", type=int, default=50,
                        help="最大序列长度 (默认: 50)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    args = parser.parse_args()

    set_seed(args.seed)

    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "num_encoder_layers": args.n_layers,
        "num_decoder_layers": args.n_layers,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "vocab_size": args.vocab_size,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "warmup_steps": 2000,
        "clip_grad_norm": 1.0,
        "log_interval": 50,
        "train_samples": 10000,
        "val_samples": 1000,
        "test_samples": 1000,
        "data_min_len": 3,
        "data_max_len": 10,
        "device": args.device,
        "share_embeddings": True,
    }

    print("=" * 60)
    print("Transformer 从零实现 - 训练与评估")
    print("=" * 60)
    print(f"设备: {config['device']}")
    print(f"模型配置: d_model={config['d_model']}, n_heads={config['n_heads']}, "
          f"n_layers={config['num_encoder_layers']}, d_ff={config['d_ff']}")
    print(f"训练配置: epochs={config['num_epochs']}, batch_size={config['batch_size']}, "
          f"lr={config['learning_rate']}")

    if args.task == "copy":
        run_copy_task(config)
    elif args.task == "reverse":
        run_reverse_task(config)


if __name__ == "__main__":
    main()