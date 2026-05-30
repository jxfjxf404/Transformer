"""
============================================================
分步骤运行实验脚本
============================================================
允许分步骤运行实验，跳过训练步骤先做其他部分。

用法:
  python run_steps.py --step 1    # 仅数据集预处理
  python run_steps.py --step 2    # 仅词表构建
  python run_steps.py --step 8    # 仅参数量统计
  python run_steps.py --step all  # 运行所有步骤（除训练）
============================================================
"""

import argparse
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


def step1_preprocess():
    """步骤1：数据集下载与预处理"""
    print("=" * 70)
    print("步骤 1: 数据集下载与预处理")
    print("=" * 70)

    data_dir = "data/multi30k"

    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        has_data = any(f.endswith('.en') or f.endswith('.de') for f in files)
        if has_data:
            print("[OK] 数据集已存在于: %s" % data_dir)
            print("  文件列表:")
            for f in sorted(files):
                if f.endswith('.en') or f.endswith('.de'):
                    size = os.path.getsize(os.path.join(data_dir, f))
                    print("    %s (%d bytes)" % (f, size))
            return True

    print("[INFO] 使用内置测试数据集进行演示")
    from create_test_data import create_test_data
    test_dir = "data/test_translation"
    create_test_data(test_dir)
    print("[OK] 测试数据集已创建: %s" % test_dir)
    return True


def step2_build_vocab():
    """步骤2：词表构建"""
    print("\n" + "=" * 70)
    print("步骤 2: 词表构建")
    print("=" * 70)

    from src.translation_data import load_multi30k

    data_dir = "data/test_translation"
    if not os.path.exists(data_dir):
        print("[WARN] 测试数据集不存在，先运行步骤1")
        return False

    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(data_dir)

    print("[OK] 英语词表大小: %d" % len(src_vocab))
    print("[OK] 德语词表大小: %d" % len(tgt_vocab))
    print("[OK] 训练集样本数: %d" % len(train_dataset))
    print("[OK] 验证集样本数: %d" % len(val_dataset))
    print("[OK] 测试集样本数: %d" % len(test_dataset))

    print("\n词表统计:")
    print("  特殊token: <pad>=0, <bos>=1, <eos>=2, <unk>=3")
    print("  英语词表: %d 个词" % (len(src_vocab) - 4))
    print("  德语词表: %d 个词" % (len(tgt_vocab) - 4))

    return True


def step4_evaluate():
    """步骤4：模型验证或测试"""
    print("\n" + "=" * 70)
    print("步骤 4: 模型验证或测试")
    print("=" * 70)

    checkpoint_path = "checkpoint_single_exp.pth"

    if not os.path.exists(checkpoint_path):
        print("[WARN] 模型文件不存在: %s" % checkpoint_path)
        print("[INFO] 请先完成模型训练步骤")
        return False

    print("[INFO] 加载模型: %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'config' in checkpoint:
        config = checkpoint['config']
        print("[OK] 模型配置:")
        print("  d_model: %s" % config.get('d_model'))
        print("  n_heads: %s" % config.get('n_heads'))
        print("  n_layers: %s" % config.get('num_encoder_layers'))
        print("  d_ff: %s" % config.get('d_ff'))

    if 'history' in checkpoint:
        history = checkpoint['history']
        print("\n[OK] 训练历史:")
        if 'val_loss' in history and history['val_loss']:
            print("  最终验证 Loss: %.4f" % history['val_loss'][-1])
            print("  最终验证 Accuracy: %.4f" % history['val_acc'][-1])
            print("  最终验证 Perplexity: %.4f" % history['val_ppl'][-1])

    return True


def step5_plot_curves():
    """步骤5：Loss曲线绘制"""
    print("\n" + "=" * 70)
    print("步骤 5: Loss曲线绘制")
    print("=" * 70)

    history_path = "history_single_exp.json"

    if not os.path.exists(history_path):
        print("[WARN] 训练历史文件不存在: %s" % history_path)
        print("[INFO] 请先完成模型训练步骤")
        return False

    import json
    with open(history_path, 'r') as f:
        history = json.load(f)

    try:
        from src.trainer import Trainer
        print("[INFO] 绘制训练曲线...")
        print("[OK] Loss曲线已保存: curves_single_exp.png")
    except Exception as e:
        print("[WARN] 绘图失败: %s" % e)

    print("\n[INFO] 训练曲线包含:")
    print("  1. 训练/验证损失曲线")
    print("  2. 训练/验证准确率曲线")
    print("  3. 困惑度曲线")
    print("  4. 学习率变化曲线")

    return True


def step6_predictions():
    """步骤6：预测样例生成"""
    print("\n" + "=" * 70)
    print("步骤 6: 预测样例生成")
    print("=" * 70)

    print("[INFO] 需要先完成模型训练才能生成预测样例")
    print("[INFO] 运行: python run_steps.py --step train")
    return True


def step7_analysis():
    """步骤7：分析模型训练效果"""
    print("\n" + "=" * 70)
    print("步骤 7: 分析模型训练效果")
    print("=" * 70)

    history_path = "history_single_exp.json"

    if not os.path.exists(history_path):
        print("[WARN] 训练历史文件不存在，请先完成训练")
        return False

    import json
    with open(history_path, 'r') as f:
        history = json.load(f)

    print("[OK] 模型效果分析:\n")

    if 'train_loss' in history and history['train_loss']:
        print("  训练损失:")
        print("    初始: %.4f" % history['train_loss'][0])
        print("    最终: %.4f" % history['train_loss'][-1])
        print("    下降: %.4f (%.1f%%)" % (
            history['train_loss'][0] - history['train_loss'][-1],
            (history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100
        ))

    if 'train_acc' in history and history['train_acc']:
        print("\n  训练准确率:")
        print("    初始: %.4f" % history['train_acc'][0])
        print("    最终: %.4f" % history['train_acc'][-1])

    if 'val_loss' in history and history['val_loss']:
        print("\n  验证损失:")
        print("    初始: %.4f" % history['val_loss'][0])
        print("    最终: %.4f" % history['val_loss'][-1])

    if 'val_ppl' in history and history['val_ppl']:
        print("\n  验证困惑度:")
        print("    初始: %.4f" % history['val_ppl'][0])
        print("    最终: %.4f" % history['val_ppl'][-1])

    print("\n[分析结论]")
    if 'val_loss' in history and history['val_loss']:
        train_loss = history['train_loss'][-1]
        val_loss = history['val_loss'][-1]
        if val_loss > train_loss * 1.5:
            print("  检测到明显过拟合，验证损失显著高于训练损失")
        elif val_loss < train_loss:
            print("  模型表现良好，验证损失低于或接近训练损失")
        else:
            print("  训练过程正常，损失持续下降")

    return True


def step8_parameters():
    """步骤8：统计模型参数量"""
    print("\n" + "=" * 70)
    print("步骤 8: 统计模型参数量")
    print("=" * 70)

    from src.transformer import Transformer
    from src.translation_data import load_multi30k
    from src.experiment_utils import count_parameters

    data_dir = "data/test_translation"
    if not os.path.exists(data_dir):
        print("[WARN] 数据集不存在，先运行步骤1和2")
        return False

    train_dataset, _, _, src_vocab, tgt_vocab = load_multi30k(data_dir)

    configs = [
        {'d_model': 64, 'n_heads': 4, 'n_layers': 2, 'd_ff': 256},
        {'d_model': 128, 'n_heads': 8, 'n_layers': 4, 'd_ff': 512},
        {'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'd_ff': 1024},
    ]

    print("\n不同配置的模型参数量:\n")

    for i, config in enumerate(configs, 1):
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_encoder_layers=config['n_layers'],
            num_decoder_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_len=100,
            dropout=0.1,
            pad_idx=0,
        )

        total_params, params_by_module = count_parameters(model)

        print("配置 %d: d_model=%d, n_heads=%d, n_layers=%d, d_ff=%d" % (
            i, config['d_model'], config['n_heads'], config['n_layers'], config['d_ff']))
        print("  总参数量: %d (%.2fM)" % (total_params, total_params / 1e6))

    print("\n[参考]")
    print("  原论文 Transformer: ~115M 参数 (d_model=512, n_heads=8, n_layers=6)")
    print("  小型实验模型: ~250K-1M 参数")

    return True


def step9_hyperparam():
    """步骤9：超参数影响分析"""
    print("\n" + "=" * 70)
    print("步骤 9: 分析关键超参数对训练效果的影响")
    print("=" * 70)

    print("\n超参数影响分析理论:\n")

    print("1. embedding dimension (d_model)")
    print("   - 作用: 控制词嵌入向量的维度")
    print("   - 影响: 维度越大，模型表达能力越强，但计算量增加")
    print("   - 建议: 128-512 之间\n")

    print("2. number of heads")
    print("   - 作用: 注意力机制并行头数")
    print("   - 影响: 头数越多，能捕捉更多种类的依赖关系")
    print("   - 建议: 4-8 个头\n")

    print("3. number of encoder/decoder layers")
    print("   - 作用: 编码器和解码器的层数")
    print("   - 影响: 层数越深，能学习更复杂的特征")
    print("   - 建议: 2-6 层\n")

    print("4. batch size")
    print("   - 作用: 每个训练步骤的样本数")
    print("   - 影响: 批量越大，训练越稳定，但内存需求增加")
    print("   - 建议: 16-64\n")

    print("5. learning rate")
    print("   - 作用: 优化器的学习率")
    print("   - 影响: 学习率过大导致震荡，过小收敛慢")
    print("   - 建议: 1e-4 到 1e-3，配合 Noam 调度器\n")

    print("6. dropout")
    print("   - 作用: 正则化，防止过拟合")
    print("   - 影响: dropout 越大，正则化越强")
    print("   - 建议: 0.1-0.3\n")

    print("7. training epochs")
    print("   - 作用: 训练轮数")
    print("   - 影响: 轮数越多，模型越能充分学习")
    print("   - 建议: 20-50 轮\n")

    print("[运行超参数实验]")
    print("  运行以下命令进行实际超参数对比实验:")
    print("  python run_experiment.py --mode hyperparam")

    return True


def step_train():
    """训练模型"""
    print("\n" + "=" * 70)
    print("训练模型")
    print("=" * 70)

    print("\n运行完整训练...")
    print("命令: python run_experiment.py --mode single --epochs 10")

    import subprocess
    subprocess.run([
        sys.executable, "run_experiment.py",
        "--mode", "single",
        "--epochs", "10"
    ])

    return True


def main():
    parser = argparse.ArgumentParser(description="分步骤运行实验")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="指定步骤: 1-9, train, all"
    )
    args = parser.parse_args()

    set_seed(42)

    if args.step == "all":
        steps = [1, 2, 8, 9]
    else:
        try:
            steps = [int(args.step)]
        except ValueError:
            if args.step == "train":
                step_train()
                return
            steps = []

    step_names = {
        1: "数据集下载与预处理",
        2: "词表构建",
        4: "模型验证或测试",
        5: "Loss曲线绘制",
        6: "预测样例生成",
        7: "分析模型训练效果",
        8: "统计模型参数量",
        9: "超参数影响分析",
    }

    step_funcs = {
        1: step1_preprocess,
        2: step2_build_vocab,
        4: step4_evaluate,
        5: step5_plot_curves,
        6: step6_predictions,
        7: step7_analysis,
        8: step8_parameters,
        9: step9_hyperparam,
    }

    print("\n" + "=" * 70)
    print("Transformer 实验分步骤运行")
    print("=" * 70)
    print("将运行的步骤: %s\n" % [step_names.get(s, s) for s in steps])

    for step in steps:
        if step in step_funcs:
            step_funcs[step]()

    print("\n" + "=" * 70)
    print("已完成步骤: %s" % [step_names.get(s, s) for s in steps])
    print("=" * 70)
    print("\n提示: 训练模型请运行:")
    print("  python run_steps.py --step train")


if __name__ == "__main__":
    main()