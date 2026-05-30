"""
============================================================
Transformer 从零实现 - 训练器模块
============================================================
包含:
  1. Trainer - 完整训练循环, 验证/测试, 模型保存/加载, 可视化

训练流程:
  1. 数据加载: 从 DataLoader 获取批次数据
  2. 前向传播: 模型处理源序列和目标序列, 输出 logits
  3. 损失计算: 交叉熵损失, 忽略填充位置的损失
  4. 反向传播: 计算梯度
  5. 参数优化: 使用 Adam 优化器更新参数 (含学习率调度)

评估指标:
  - 损失 (Loss): 交叉熵损失
  - 准确率 (Accuracy): 预测正确的 token 比例
  - 困惑度 (Perplexity): exp(loss), 衡量模型对序列的预测能力

可视化:
  - 训练/验证损失曲线
  - 训练/验证准确率曲线
  - 学习率变化曲线

参考:
  - "Attention Is All You Need" (Vaswani et al., 2017)
============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
import json
import time
import numpy as np

from .mask import (
    create_encoder_mask,
    create_combined_mask,
)


class Trainer:
    """
    ============================================================
    训练器 (Trainer)
    ============================================================
    封装完整的训练、验证、测试、模型保存加载和可视化功能。

    使用示例:
      trainer = Trainer(model, train_loader, val_loader, config)
      trainer.train()
      trainer.save_model("checkpoint.pth")
      trainer.plot_curves("training_curves.png")
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pad_idx: int = 0,
        warmup_steps: int = 4000,
        label_smoothing: float = 0.0,
        clip_grad_norm: float = 1.0,
        log_interval: int = 100,
    ):
        """
        参数:
            model:           Transformer 模型
            train_loader:    训练数据加载器
            val_loader:      验证数据加载器
            test_loader:     测试数据加载器
            learning_rate:   基础学习率
            num_epochs:      训练轮数
            device:          训练设备
            pad_idx:         填充 token 索引
            warmup_steps:    warmup 步数 (用于学习率调度)
            label_smoothing: 标签平滑系数
            clip_grad_norm:  梯度裁剪阈值
            log_interval:    日志打印间隔 (步数)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.pad_idx = pad_idx
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.label_smoothing = label_smoothing

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
        )

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self._lr_lambda(step),
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_ppl": [],
            "val_ppl": [],
            "learning_rate": [],
        }

        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _lr_lambda(self, step: int) -> float:
        """
        ============================================================
        学习率 warmup 调度 (Noam 调度)
        ============================================================
        论文公式:
          lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

        前 warmup_steps 步线性增长, 之后按 step^{-0.5} 衰减。

        参数:
            step: 当前训练步数

        返回:
            学习率缩放因子
        """
        d_model = self.model.d_model
        step = max(step, 1)
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (d_model ** (-0.5)) * min(arg1, arg2)

    def train(self) -> Dict[str, List[float]]:
        """
        ============================================================
        执行完整训练循环
        ============================================================
        每个 epoch:
          1. 训练阶段: 遍历训练集, 前向传播 → 损失计算 → 反向传播 → 参数更新
          2. 验证阶段: 在验证集上评估模型性能 (loss, accuracy, perplexity)
          3. 记录指标并保存最佳模型

        返回:
            history: 训练历史记录
        """
        print("=" * 60)
        print(f"开始训练 | 设备: {self.device} | Epochs: {self.num_epochs}")
        print(f"模型参数量: {self._count_parameters():,}")
        print("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc, train_ppl = self._train_one_epoch(epoch)

            if self.val_loader is not None:
                val_loss, val_acc, val_ppl = self._validate()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_ppl"].append(val_ppl)

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch

                status = " *" if is_best else ""
            else:
                val_loss = val_acc = val_ppl = 0.0
                status = ""

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_ppl"].append(train_ppl)

            epoch_time = time.time() - epoch_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val PPL: {val_ppl:.2f}{status}"
            )

        print("=" * 60)
        print(f"训练完成! 最佳验证 Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
        print("=" * 60)
        return self.history

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """
        训练一个 epoch。

        返回:
            avg_loss:       平均损失
            avg_accuracy:   平均准确率
            avg_perplexity: 平均困惑度
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.train_loader)

        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            src_mask = create_encoder_mask(src, self.pad_idx).to(self.device)
            tgt_mask = create_combined_mask(tgt, self.pad_idx).to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()

            logits = self.model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                non_pad_mask = (tgt_output != self.pad_idx)
                correct = (pred == tgt_output) & non_pad_mask
                total_correct += correct.sum().item()
                total_tokens += non_pad_mask.sum().item()

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx:4d}/{num_batches}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        avg_accuracy = total_correct / max(total_tokens, 1)
        avg_perplexity = np.exp(avg_loss)

        return avg_loss, avg_accuracy, avg_perplexity

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        """
        验证/评估模型。

        返回:
            avg_loss:       平均损失
            avg_accuracy:   平均准确率
            avg_perplexity: 平均困惑度
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.val_loader)

        for src, tgt in self.val_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            src_mask = create_encoder_mask(src, self.pad_idx).to(self.device)
            tgt_mask = create_combined_mask(tgt, self.pad_idx).to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = self.model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            pred = logits.argmax(dim=-1)
            non_pad_mask = (tgt_output != self.pad_idx)
            correct = (pred == tgt_output) & non_pad_mask
            total_correct += correct.sum().item()
            total_tokens += non_pad_mask.sum().item()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        avg_accuracy = total_correct / max(total_tokens, 1)
        avg_perplexity = np.exp(avg_loss)

        return avg_loss, avg_accuracy, avg_perplexity

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """
        ============================================================
        测试模型
        ============================================================
        在测试集上评估模型性能, 返回详细的指标。

        返回:
            results: 包含 loss, accuracy, perplexity 的字典
        """
        if self.test_loader is None:
            print("未提供测试集, 跳过测试。")
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = len(self.test_loader)

        for src, tgt in self.test_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            src_mask = create_encoder_mask(src, self.pad_idx).to(self.device)
            tgt_mask = create_combined_mask(tgt, self.pad_idx).to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = self.model(src, tgt_input, src_mask, tgt_mask[:, :, :-1, :-1])

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            pred = logits.argmax(dim=-1)
            non_pad_mask = (tgt_output != self.pad_idx)
            correct = (pred == tgt_output) & non_pad_mask
            total_correct += correct.sum().item()
            total_tokens += non_pad_mask.sum().item()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        avg_accuracy = total_correct / max(total_tokens, 1)
        avg_perplexity = np.exp(avg_loss)

        results = {
            "test_loss": avg_loss,
            "test_accuracy": avg_accuracy,
            "test_perplexity": avg_perplexity,
        }

        print("=" * 60)
        print("测试结果:")
        print(f"  Loss:        {avg_loss:.4f}")
        print(f"  Accuracy:    {avg_accuracy:.4f}")
        print(f"  Perplexity:  {avg_perplexity:.2f}")
        print("=" * 60)

        return results

    def save_model(self, filepath: str) -> None:
        """
        ============================================================
        保存模型
        ============================================================
        保存模型参数、优化器状态、训练历史和配置信息。

        参数:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }

        torch.save(checkpoint, filepath)
        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        ============================================================
        加载模型
        ============================================================
        恢复模型参数、优化器状态和训练历史。

        参数:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        if "best_epoch" in checkpoint:
            self.best_epoch = checkpoint["best_epoch"]

        print(f"模型已从 {filepath} 加载 (最佳 Epoch: {self.best_epoch})")

    def plot_curves(self, save_path: str = "training_curves.png") -> None:
        """
        ============================================================
        可视化训练曲线
        ============================================================
        绘制:
          1. 训练和验证损失曲线
          2. 训练和验证准确率曲线
          3. 学习率变化曲线

        参数:
            save_path: 图片保存路径
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
        except ImportError:
            print("matplotlib 未安装, 跳过绘图。")
            return

        try:
            fm.findfont("SimHei")
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        except Exception:
            pass
        plt.rcParams["axes.unicode_minus"] = False

        epochs = range(1, len(self.history["train_loss"]) + 1)
        has_val = len(self.history.get("val_loss", [])) > 0

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Transformer 训练曲线", fontsize=16, fontweight="bold")

        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
        if has_val:
            ax1.plot(epochs, self.history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("损失曲线 (Loss)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
        if has_val:
            ax2.plot(epochs, self.history["val_acc"], "r-", label="Val Acc", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("准确率曲线 (Accuracy)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history["train_ppl"], "b-", label="Train PPL", linewidth=1.5)
        if has_val:
            ax3.plot(epochs, self.history["val_ppl"], "r-", label="Val PPL", linewidth=1.5)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Perplexity")
        ax3.set_title("困惑度曲线 (Perplexity)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.plot(epochs, self.history["learning_rate"], "g-", linewidth=1.5)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("学习率变化曲线")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"训练曲线已保存至: {save_path}")

    def _count_parameters(self) -> int:
        """计算模型的可训练参数总数。"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def export_history(self, filepath: str = "training_history.json") -> None:
        """
        导出训练历史为 JSON 文件。

        参数:
            filepath: JSON 文件路径
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"训练历史已导出至: {filepath}")