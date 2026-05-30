"""
============================================================
Transformer 完整实验脚本
============================================================
包含所有实验要求：
  1. 数据集下载与预处理
  2. 词表构建
  3. 模型训练
  4. 模型验证与测试
  5. Loss 曲线绘制
  6. 预测样例生成
  7. 模型效果分析
  8. 参数量统计
  9. 超参数影响分析

用法:
  python run_experiment.py --mode single  # 单次实验
  python run_experiment.py --mode hyperparam  # 超参数实验
  python run_experiment.py --help

参考:
  - Multi30k 数据集: https://github.com/multi30k/dataset
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
    SPECIAL_TOKENS,
)
from src.trainer import Trainer
from src.experiment_utils import (
    count_parameters,
    generate_report,
    print_report,
    generate_predictions,
    print_predictions,
    run_hyperparameter_experiment,
)


def set_seed(seed: int = 42):
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_or_use_test_data(data_dir: str = "data/multi30k"):
    """下载数据集或使用测试数据。"""
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        if any(f.endswith('.en') or f.endswith('.de') for f in files):
            print(f"数据集已存在于: {data_dir}")
            return data_dir

    print(f"数据集目录不存在或为空: {data_dir}")
    print("使用内置测试数据集")

    test_dir = "data/test_translation"
    if not os.path.exists(test_dir):
        print("生成测试数据集...")
        from create_test_data import create_test_data
        create_test_data(test_dir)

    return test_dir


def train_single_model(config: dict):
    """训练单个模型。"""
    data_dir = download_or_use_test_data(config.get('data_dir', "data/multi30k"))

    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(data_dir)

    device = config['device']
    pad_idx = SPECIAL_TOKENS["<pad>"]

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_layers=config['n_layers'],
        num_decoder_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        pad_idx=pad_idx,
        share_embeddings=config.get('share_embeddings', False),
    )

    total_params, params_by_module = count_parameters(model)
    print(f"\n模型参数量统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  各模块参数量:")
    for name, count in params_by_module.items():
        print(f"    {name}: {count:,}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pad_idx=pad_idx,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pad_idx=pad_idx,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pad_idx=pad_idx,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config['lr'],
        num_epochs=config['epochs'],
        device=device,
        pad_idx=pad_idx,
        warmup_steps=config.get('warmup_steps', 4000),
        clip_grad_norm=config.get('clip_grad_norm', 1.0),
        log_interval=config.get('log_interval', 100),
    )

    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    trainer.train()

    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    test_loss, test_acc, test_ppl = trainer.test()

    checkpoint_name = f"checkpoint_{config.get('name', 'exp')}.pth"
    history_name = f"history_{config.get('name', 'exp')}.json"
    curve_name = f"curves_{config.get('name', 'exp')}.png"

    trainer.save_model(checkpoint_name)
    trainer.export_history(history_name)
    trainer.plot_curves(curve_name)

    print("\n" + "=" * 70)
    print("生成预测样例")
    print("=" * 70)
    predictions = generate_predictions(
        model,
        test_dataset,
        src_vocab,
        tgt_vocab,
        bos_idx=SPECIAL_TOKENS["<bos>"],
        eos_idx=SPECIAL_TOKENS["<eos>"],
        max_len=config['max_len'],
        num_samples=5,
        device=device,
    )
    print_predictions(predictions)

    return {
        'final_train_loss': trainer.history['train_loss'][-1] if trainer.history['train_loss'] else None,
        'final_val_loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else None,
        'final_val_acc': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else None,
        'final_val_ppl': trainer.history['val_ppl'][-1] if trainer.history['val_ppl'] else None,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_ppl': test_ppl,
        'total_params': total_params,
        'predictions': predictions,
    }


def run_hyperparameter_test():
    """运行超参数对比实验。"""
    default_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'batch_size': 8,
        'lr': 1e-3,
        'dropout': 0.1,
        'epochs': 20,
        'max_len': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'warmup_steps': 1000,
        'log_interval': 10,
    }

    experiments = []

    for d_model in [64, 128, 256]:
        config = default_config.copy()
        config['d_model'] = d_model
        config['name'] = f"dmodel_{d_model}"
        config['n_heads'] = min(8, d_model // 32)
        config['d_ff'] = d_model * 4
        experiments.append(config)

    for n_layers in [1, 2, 3]:
        config = default_config.copy()
        config['n_layers'] = n_layers
        config['name'] = f"nlayers_{n_layers}"
        experiments.append(config)

    for n_heads in [2, 4, 8]:
        config = default_config.copy()
        config['n_heads'] = n_heads
        config['name'] = f"nheads_{n_heads}"
        config['d_model'] = max(64, n_heads * 32)
        config['d_ff'] = config['d_model'] * 4
        experiments.append(config)

    for dropout in [0.0, 0.1, 0.3]:
        config = default_config.copy()
        config['dropout'] = dropout
        config['name'] = f"dropout_{dropout}"
        experiments.append(config)

    def train_wrapper(config):
        return train_single_model(config)

    results = run_hyperparameter_experiment(experiments, train_wrapper, verbose=True)

    report = generate_report(results, "hyperparameter_report.json")
    print_report(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Transformer 完整实验")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "hyperparam"],
        help="实验模式: single(单次实验) 或 hyperparam(超参数对比)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")
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

    if args.mode == "single":
        config = {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'd_ff': args.d_ff,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'dropout': args.dropout,
            'epochs': args.epochs,
            'max_len': args.max_len,
            'device': device,
            'data_dir': args.data_dir,
            'name': "single_exp",
            'warmup_steps': 4000,
            'log_interval': 50,
        }

        print("=" * 70)
        print("Transformer 单次实验")
        print("=" * 70)
        print(f"配置: {config}")
        print("=" * 70)

        result = train_single_model(config)

        print("\n" + "=" * 70)
        print("实验结果总结")
        print("=" * 70)
        print(f"训练 Loss: {result.get('final_train_loss', 'N/A'):.4f}")
        print(f"验证 Loss: {result.get('final_val_loss', 'N/A'):.4f}")
        print(f"验证 Accuracy: {result.get('final_val_acc', 'N/A'):.4f}")
        print(f"验证 Perplexity: {result.get('final_val_ppl', 'N/A'):.4f}")
        print(f"测试 Loss: {result.get('test_loss', 'N/A'):.4f}")
        print(f"测试 Accuracy: {result.get('test_acc', 'N/A'):.4f}")
        print(f"测试 Perplexity: {result.get('test_ppl', 'N/A'):.4f}")
        print(f"模型参数量: {result.get('total_params', 'N/A'):,}")
        print("=" * 70)

    elif args.mode == "hyperparam":
        run_hyperparameter_test()


if __name__ == "__main__":
    main()