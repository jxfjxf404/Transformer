"""
============================================================
Transformer 实验工具模块
============================================================
包含实验所需的辅助功能：
  1. 模型参数量统计
  2. 超参数分析工具
  3. 实验结果可视化
  4. 预测样例生成
  5. 实验报告生成
============================================================
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any


def count_parameters(model: torch.nn.Module) -> Tuple[int, Dict[str, int]]:
    """
    统计模型参数量。

    参数:
        model: 模型

    返回:
        total_params: 总参数量
        params_by_module: 各模块参数量
    """
    total_params = 0
    params_by_module = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and len(list(module.named_parameters())) > 0:
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            params_by_module[name] = module_params
            total_params += module_params

    return total_params, params_by_module


def analyze_hyperparameters(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析超参数对训练效果的影响。

    参数:
        results: 实验结果列表

    返回:
        analysis: 分析结果
    """
    analysis = {}

    if not results:
        return analysis

    keys = ['d_model', 'n_heads', 'n_layers', 'batch_size', 'lr', 'dropout', 'epochs']
    metrics = ['final_train_loss', 'final_val_loss', 'final_val_acc', 'final_val_ppl']

    for key in keys:
        analysis[key] = {
            'values': [],
            'correlations': {},
            'best': {'value': None, 'metrics': None}
        }

        values = list(set(r.get(key) for r in results if key in r))
        analysis[key]['values'] = sorted(values)

        for metric in metrics:
            values_list = []
            metrics_list = []
            for r in results:
                if key in r and metric in r:
                    values_list.append(r[key])
                    metrics_list.append(r[metric])

            if len(values_list) >= 2:
                correlation = np.corrcoef(values_list, metrics_list)[0, 1]
                analysis[key]['correlations'][metric] = float(correlation)

        best_idx = None
        best_ppl = float('inf')
        for i, r in enumerate(results):
            if key in r and 'final_val_ppl' in r and r['final_val_ppl'] < best_ppl:
                best_ppl = r['final_val_ppl']
                best_idx = i

        if best_idx is not None:
            analysis[key]['best']['value'] = results[best_idx][key]
            analysis[key]['best']['metrics'] = {m: results[best_idx].get(m) for m in metrics}

    return analysis


def generate_report(results: List[Dict[str, Any]], output_path: str = "experiment_report.json"):
    """
    生成实验报告。

    参数:
        results: 实验结果列表
        output_path: 报告输出路径
    """
    report = {
        'summary': {
            'total_experiments': len(results),
            'best_overall': None,
            'average_metrics': {}
        },
        'hyperparameter_analysis': analyze_hyperparameters(results),
        'detailed_results': results
    }

    if results:
        best_idx = None
        best_ppl = float('inf')
        for i, r in enumerate(results):
            if 'final_val_ppl' in r and r['final_val_ppl'] < best_ppl:
                best_ppl = r['final_val_ppl']
                best_idx = i

        if best_idx is not None:
            report['summary']['best_overall'] = results[best_idx]

        metrics = ['final_train_loss', 'final_val_loss', 'final_val_acc', 'final_val_ppl']
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                report['summary']['average_metrics'][metric] = float(np.mean(values))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"实验报告已保存至: {output_path}")
    return report


def print_report(report: Dict[str, Any]):
    """
    打印实验报告。

    参数:
        report: 实验报告
    """
    print("=" * 70)
    print("Transformer 实验报告")
    print("=" * 70)

    print(f"\n[实验概述]")
    print(f"  实验总数: {report['summary']['total_experiments']}")

    if report['summary']['average_metrics']:
        print(f"\n[平均指标]")
        for metric, value in report['summary']['average_metrics'].items():
            print(f"  {metric}: {value:.4f}")

    if report['summary']['best_overall']:
        print(f"\n[最佳模型配置]")
        best = report['summary']['best_overall']
        print(f"  d_model: {best.get('d_model')}")
        print(f"  n_heads: {best.get('n_heads')}")
        print(f"  n_layers: {best.get('n_layers')}")
        print(f"  batch_size: {best.get('batch_size')}")
        print(f"  lr: {best.get('lr')}")
        print(f"  dropout: {best.get('dropout')}")
        print(f"  epochs: {best.get('epochs')}")
        print(f"\n  验证集指标:")
        print(f"    Loss: {best.get('final_val_loss', 'N/A'):.4f}")
        print(f"    Accuracy: {best.get('final_val_acc', 'N/A'):.4f}")
        print(f"    Perplexity: {best.get('final_val_ppl', 'N/A'):.4f}")

    print("\n[超参数影响分析]")
    for param, info in report['hyperparameter_analysis'].items():
        print(f"\n  {param}:")
        print(f"    测试值: {info['values']}")

        if info['correlations']:
            print(f"    相关性分析:")
            for metric, corr in info['correlations'].items():
                if corr > 0.5:
                    trend = "(正相关)"
                elif corr < -0.5:
                    trend = "(负相关)"
                else:
                    trend = "(弱相关)"
                print(f"      {metric}: {corr:.3f} {trend}")

        if info['best']['value'] is not None:
            print(f"    最佳值: {info['best']['value']}")

    print("\n" + "=" * 70)


@torch.no_grad()
def generate_predictions(
    model,
    dataset,
    src_vocab,
    tgt_vocab,
    bos_idx: int,
    eos_idx: int,
    max_len: int = 50,
    num_samples: int = 5,
    device: str = "cpu"
) -> List[Dict[str, str]]:
    """
    生成预测样例。

    参数:
        model: 模型
        dataset: 数据集
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        bos_idx: 开始标记索引
        eos_idx: 结束标记索引
        max_len: 最大生成长度
        num_samples: 生成样本数
        device: 设备

    返回:
        predictions: 预测结果列表
    """
    model.eval()
    predictions = []

    src_idx_to_word = {idx: word for word, idx in src_vocab.items()}
    tgt_idx_to_word = {idx: word for word, idx in tgt_vocab.items()}

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        src, tgt_expected = dataset[idx]
        src = src.unsqueeze(0).to(device)

        generated = model.generate(
            src,
            max_len=max_len,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

        def idx_to_sentence(indices, idx_to_word, eos_idx):
            tokens = []
            for idx in indices:
                if idx == eos_idx:
                    break
                tokens.append(idx_to_word.get(idx, "<unk>"))
            return " ".join(tokens)

        src_sentence = idx_to_sentence(src[0].cpu().numpy(), src_idx_to_word, eos_idx)
        tgt_sentence = idx_to_sentence(tgt_expected.cpu().numpy(), tgt_idx_to_word, eos_idx)
        gen_sentence = idx_to_sentence(generated[0].cpu().numpy(), tgt_idx_to_word, eos_idx)

        predictions.append({
            'source': src_sentence,
            'target': tgt_sentence,
            'generated': gen_sentence
        })

    return predictions


def print_predictions(predictions: List[Dict[str, str]]):
    """
    打印预测样例。

    参数:
        predictions: 预测结果列表
    """
    print("\n" + "=" * 70)
    print("翻译预测样例")
    print("=" * 70)

    for i, pred in enumerate(predictions, 1):
        print(f"\n示例 {i}:")
        print(f"  源语言: {pred['source']}")
        print(f"  目标语言: {pred['target']}")
        print(f"  模型输出: {pred['generated']}")

        if pred['target'].strip() == pred['generated'].strip():
            print("  [匹配] ✓")
        else:
            print("  [不匹配]")

    print("\n" + "=" * 70)


def run_hyperparameter_experiment(
    configs: List[Dict[str, Any]],
    train_func,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    运行超参数对比实验。

    参数:
        configs: 配置列表
        train_func: 训练函数
        verbose: 是否打印详细信息

    返回:
        results: 实验结果列表
    """
    results = []

    print("=" * 70)
    print("超参数对比实验")
    print("=" * 70)
    print(f"实验数量: {len(configs)}")
    print("=" * 70)

    for i, config in enumerate(configs, 1):
        if verbose:
            print(f"\n[实验 {i}/{len(configs)}]")
            print(f"配置: {config}")

        try:
            result = train_func(config)
            result.update(config)
            results.append(result)

            if verbose:
                print(f"  训练完成")
                print(f"  验证 Loss: {result.get('final_val_loss', 'N/A'):.4f}")
                print(f"  验证 Accuracy: {result.get('final_val_acc', 'N/A'):.4f}")
                print(f"  验证 Perplexity: {result.get('final_val_ppl', 'N/A'):.4f}")

        except Exception as e:
            print(f"  实验失败: {e}")
            results.append({'config': config, 'error': str(e)})

    print("\n" + "=" * 70)
    print("超参数实验完成!")
    print("=" * 70)

    return results