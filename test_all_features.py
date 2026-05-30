"""
============================================================
Transformer 完整功能测试脚本
============================================================
验证所有实验功能是否正常工作：
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
  python test_all_features.py
============================================================
"""

import torch
import random
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int = 42):
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_all_features():
    """测试所有功能。"""
    print("=" * 70)
    print("Transformer 完整功能测试")
    print("=" * 70)

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备: %s\n" % device)

    # 1. 数据集下载与预处理
    print("[1/9] 数据集下载与预处理")
    print("-" * 70)
    try:
        from create_test_data import create_test_data
        data_dir = "data/test_translation"
        create_test_data(data_dir)
        print("  [OK] 测试数据集已创建: %s" % data_dir)
    except Exception as e:
        print("  [FAIL] 数据集创建失败: %s" % e)
        return

    # 2. 词表构建
    print("\n[2/9] 词表构建")
    print("-" * 70)
    try:
        from src.translation_data import load_multi30k
        train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(data_dir)
        print("  [OK] 英语词表大小: %d" % len(src_vocab))
        print("  [OK] 德语词表大小: %d" % len(tgt_vocab))
        print("  [OK] 训练集样本数: %d" % len(train_dataset))
        print("  [OK] 验证集样本数: %d" % len(val_dataset))
        print("  [OK] 测试集样本数: %d" % len(test_dataset))
    except Exception as e:
        print("  [FAIL] 词表构建失败: %s" % e)
        return

    # 3. 模型训练
    print("\n[3/9] 模型训练")
    print("-" * 70)
    try:
        from src.transformer import Transformer
        from src.translation_data import create_dataloader, SPECIAL_TOKENS
        from src.trainer import Trainer

        config = {
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 256,
            'batch_size': 8,
            'lr': 1e-3,
            'dropout': 0.1,
            'epochs': 3,
            'max_len': 50,
            'device': device,
        }

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
            pad_idx=SPECIAL_TOKENS["<pad>"],
        )

        train_loader = create_dataloader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pad_idx=SPECIAL_TOKENS["<pad>"],
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pad_idx=SPECIAL_TOKENS["<pad>"],
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pad_idx=SPECIAL_TOKENS["<pad>"],
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=config['lr'],
            num_epochs=config['epochs'],
            device=device,
            pad_idx=SPECIAL_TOKENS["<pad>"],
            warmup_steps=100,
            clip_grad_norm=1.0,
            log_interval=5,
        )

        print("  开始训练 (epochs=%d, batch_size=%d)" % (config['epochs'], config['batch_size']))
        trainer.train()
        print("  [OK] 模型训练完成")
    except Exception as e:
        print("  [FAIL] 模型训练失败: %s" % e)
        return

    # 4. 模型验证与测试
    print("\n[4/9] 模型验证与测试")
    print("-" * 70)
    try:
        print("  验证集评估...")
        val_loss, val_acc, val_ppl = trainer._validate()
        print("    Val Loss: %.4f" % val_loss)
        print("    Val Accuracy: %.4f" % val_acc)
        print("    Val Perplexity: %.4f" % val_ppl)

        print("  测试集评估...")
        test_results = trainer.test()
        test_loss = test_results.get('test_loss', 0.0)
        test_acc = test_results.get('test_accuracy', 0.0)
        test_ppl = test_results.get('test_perplexity', 0.0)
        print("    Test Loss: %.4f" % test_loss)
        print("    Test Accuracy: %.4f" % test_acc)
        print("    Test Perplexity: %.4f" % test_ppl)
        print("  [OK] 模型验证与测试完成")
    except Exception as e:
        print("  [FAIL] 模型验证与测试失败: %s" % e)
        return

    # 5. Loss 曲线绘制
    print("\n[5/9] Loss 曲线绘制")
    print("-" * 70)
    try:
        trainer.plot_curves("test_curves.png")
        print("  [OK] Loss 曲线已保存: test_curves.png")
    except Exception as e:
        print("  [FAIL] Loss 曲线绘制失败: %s" % e)

    # 6. 预测样例生成
    print("\n[6/9] 预测样例生成")
    print("-" * 70)
    try:
        from src.experiment_utils import generate_predictions, print_predictions

        predictions = generate_predictions(
            model,
            test_dataset,
            src_vocab,
            tgt_vocab,
            bos_idx=SPECIAL_TOKENS["<bos>"],
            eos_idx=SPECIAL_TOKENS["<eos>"],
            max_len=config['max_len'],
            num_samples=3,
            device=device,
        )
        print_predictions(predictions)
        print("  [OK] 预测样例生成完成")
    except Exception as e:
        print("  [FAIL] 预测样例生成失败: %s" % e)

    # 7. 模型效果分析
    print("\n[7/9] 模型效果分析")
    print("-" * 70)
    try:
        print("  训练历史分析:")
        print("    初始训练 Loss: %.4f" % trainer.history['train_loss'][0])
        print("    最终训练 Loss: %.4f" % trainer.history['train_loss'][-1])
        print("    Loss 下降: %.4f" % (trainer.history['train_loss'][0] - trainer.history['train_loss'][-1]))

        if trainer.history['val_loss']:
            print("    初始验证 Loss: %.4f" % trainer.history['val_loss'][0])
            print("    最终验证 Loss: %.4f" % trainer.history['val_loss'][-1])

        print("  [OK] 模型效果分析完成")
    except Exception as e:
        print("  [FAIL] 模型效果分析失败: %s" % e)

    # 8. 参数量统计
    print("\n[8/9] 参数量统计")
    print("-" * 70)
    try:
        from src.experiment_utils import count_parameters

        total_params, params_by_module = count_parameters(model)
        print("  总参数量: %d" % total_params)
        print("  各模块参数量:")
        for name, count in params_by_module.items():
            if count > 0:
                print("    %s: %d" % (name, count))
        print("  [OK] 参数量统计完成")
    except Exception as e:
        print("  [FAIL] 参数量统计失败: %s" % e)

    # 9. 超参数影响分析
    print("\n[9/9] 超参数影响分析")
    print("-" * 70)
    try:
        from src.experiment_utils import generate_report, print_report

        mock_results = [
            {'d_model': 64, 'n_heads': 4, 'n_layers': 2, 'batch_size': 8, 'lr': 1e-3, 'dropout': 0.1, 'epochs': 3,
             'final_train_loss': 3.5, 'final_val_loss': 3.8, 'final_val_acc': 0.2, 'final_val_ppl': 44.7},
            {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'batch_size': 8, 'lr': 1e-3, 'dropout': 0.1, 'epochs': 3,
             'final_train_loss': 3.2, 'final_val_loss': 3.5, 'final_val_acc': 0.25, 'final_val_ppl': 33.2},
            {'d_model': 64, 'n_heads': 2, 'n_layers': 2, 'batch_size': 8, 'lr': 1e-3, 'dropout': 0.1, 'epochs': 3,
             'final_train_loss': 3.8, 'final_val_loss': 4.0, 'final_val_acc': 0.15, 'final_val_ppl': 54.6},
        ]

        report = generate_report(mock_results, "test_report.json")
        print_report(report)
        print("  [OK] 超参数影响分析完成")
    except Exception as e:
        print("  [FAIL] 超参数影响分析失败: %s" % e)

    print("\n" + "=" * 70)
    print("所有功能测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_all_features()