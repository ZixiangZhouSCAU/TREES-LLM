"""
两阶段训练器 - TwoStageTrainer

Stage 1（描述对齐）：Caption 任务
  - 只更新三个 FeatureProjector（Object/Relationship/Scene）
  - 目标：让投影特征能驱动 LLM 生成正确描述

Stage 2（分析微调）：Analysis / Planning 任务
  - 更新三个 Projector + 可选的 LoRA adapter
  - 目标：让投影特征支持专业林业分析

用法：
    # 训练 projector（Stage 1）
    trainer = TwoStageTrainer(
        model_path="data/tree_training_data.jsonl",
        output_dir="outputs/stage1",
    )
    trainer.train_stage1(epochs=10, lr=1e-3, batch_size=8)

    # 微调（Stage 2）
    trainer.train_stage2(epochs=5, lr=5e-4, batch_size=4)

    # 加载已训练权重
    trainer.load("outputs/stage1/best_model.pt")
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

from src.training.dataset import TreeDataset


class TwoStageTrainer:
    """
    两阶段训练器

    Stage 1: 特征对齐（Feature Alignment）
      - 数据：ObjectCaption
      - 损失：Caption Loss（交叉熵）
      - 参数：只更新 3 个 Projector（~6M 参数）

    Stage 2: 指令微调（Instruction Tuning）
      - 数据：SceneAnalysis + ScenePlanning
      - 损失：Caption Loss
      - 参数：Projector + LoRA（可选）
    """

    def __init__(
        self,
        tree_encoder_path: str = "src/models/tree_encoder.py",
        training_data_path: str = "data/training/tree_training_data.jsonl",
        output_dir: str = "outputs",
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.training_data_path = training_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 延迟初始化模型
        self.encoder = None
        self.optimizer = None

        # 日志
        self.train_history: List[Dict] = []

        print(f"[TwoStageTrainer] Initialized on {self.device}")

    def _init_model(self):
        """延迟加载 TreeEncoder"""
        if self.encoder is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.models.tree_encoder import TreeEncoder

            self.encoder = TreeEncoder(device=str(self.device), use_pretrained=False)
            self.encoder.to(self.device)
            print(f"[TwoStageTrainer] TreeEncoder loaded: {self.encoder.num_params():,} trainable params")

    def _init_optimizer(self, lr: float = 1e-3):
        """初始化优化器（只优化 projector 参数）"""
        self._init_model()
        params = self.encoder.get_trainable_parameters()
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=0.01)
        print(f"[TwoStageTrainer] Optimizer initialized: {len(params)} param groups, lr={lr}")

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 caption loss（简化的交叉熵）

        注意：这里用 token-level CE 模拟真实训练
        实际训练时，Projector 输出接 LLM 的 embedding 层，
        整体作为 language model 计算 loss
        """
        # 展平
        pred_flat = predictions.view(-1)
        tgt_flat = targets.view(-1)

        # 忽略 pad token（id=0）
        mask = tgt_flat != 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        loss = nn.functional.cross_entropy(
            pred_flat[mask].unsqueeze(0),
            tgt_flat[mask],
            reduction="mean",
        )
        return loss

    def train_stage1(
        self,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 8,
        val_split: float = 0.1,
        log_every: int = 10,
    ):
        """
        Stage 1：特征对齐训练（ObjectCaption 任务）

        目标：让三个 projector 学习将点云特征映射到文本描述空间
        """
        print("\n" + "=" * 50)
        print("Stage 1: Feature Alignment (ObjectCaption)")
        print("=" * 50)

        self._init_optimizer(lr=lr)

        # 加载数据集
        dataset = TreeDataset(
            self.training_data_path,
            task="object_caption",
        )
        if len(dataset) == 0:
            print("[WARN] No object_caption samples found, skipping Stage 1")
            return

        # 划分训练/验证
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        indices = np.random.permutation(len(dataset))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_loader = DataLoader(
            [dataset[i] for i in train_indices],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            [dataset[i] for i in val_indices],
            batch_size=batch_size,
            shuffle=False,
        )

        print(f"Train: {n_train}, Val: {n_val}")

        best_loss = float("inf")

        for epoch in range(epochs):
            self.encoder.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                # Forward（这里用随机 embedding 模拟 projector 输出）
                # 真实训练：encoder 输出 → LLM → loss
                batch_size_actual = len(batch)

                # 模拟 projector 输出 logits（vocab_size=5000）
                logits = torch.randn(
                    batch_size_actual, 64, 5000,
                    device=self.device,
                )
                labels = torch.randint(
                    1, 5000,
                    (batch_size_actual, 64),
                    device=self.device,
                )

                loss = self._compute_loss(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.get_trainable_parameters(), max_norm=1.0
                )
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                if n_batches % log_every == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {n_batches}, "
                          f"Loss: {loss.item():.4f}")

            avg_train_loss = total_loss / max(n_batches, 1)

            # 验证
            val_loss = self._eval(val_loader)
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 记录
            self.train_history.append({
                "stage": "stage1",
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            })

            # 保存 best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save("best_model.pt")
                print(f"  [Saved] best_model.pt (val_loss={val_loss:.4f})")

        print(f"\nStage 1 完成！Best Val Loss: {best_loss:.4f}")

    def train_stage2(
        self,
        epochs: int = 5,
        lr: float = 5e-4,
        batch_size: int = 4,
        log_every: int = 10,
    ):
        """
        Stage 2：指令微调（SceneAnalysis + ScenePlanning）

        目标：训练 projector 支持专业林业分析和管理规划
        """
        print("\n" + "=" * 50)
        print("Stage 2: Instruction Tuning (Analysis + Planning)")
        print("=" * 50)

        # 加载 Stage 1 权重
        self._init_model()
        self.load("best_model.pt")

        # 降低学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # 加载 SceneAnalysis + ScenePlanning 数据
        dataset = TreeDataset(self.training_data_path, task=None)
        scene_data = [d for d in dataset if d.get("task") in ("scene_analysis", "scene_planning")]

        if not scene_data:
            print("[WARN] No scene_analysis/scene_planning samples found, skipping Stage 2")
            return

        print(f"Scene samples: {len(scene_data)}")

        loader = DataLoader(scene_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.encoder.train()
            total_loss = 0.0

            for i, batch in enumerate(loader):
                logits = torch.randn(len(batch), 64, 5000, device=self.device)
                labels = torch.randint(1, 5000, (len(batch), 64), device=self.device)
                loss = self._compute_loss(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (i + 1) % log_every == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / max(len(loader), 1)
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

            self.train_history.append({
                "stage": "stage2",
                "epoch": epoch + 1,
                "train_loss": avg_loss,
            })

            # 保存
            self.save(f"stage2_epoch{epoch+1}.pt")

        print(f"\nStage 2 完成！")

    @torch.no_grad()
    def _eval(self, val_loader) -> float:
        """验证"""
        self.encoder.eval()
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            logits = torch.randn(len(batch), 64, 5000, device=self.device)
            labels = torch.randint(1, 5000, (len(batch), 64), device=self.device)
            loss = self._compute_loss(logits, labels)
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def save(self, filename: str = "best_model.pt"):
        """保存 projector 权重"""
        if self.encoder is None:
            return
        save_path = self.output_dir / filename
        torch.save(
            {
                "object_proj": self.encoder.object_proj.state_dict(),
                "relationship_proj": self.encoder.relationship_proj.state_dict(),
                "scene_proj": self.encoder.scene_proj.state_dict(),
                "train_history": self.train_history,
            },
            save_path,
        )
        print(f"[TwoStageTrainer] Saved to {save_path}")

    def load(self, filename: str = "best_model.pt"):
        """加载 projector 权重"""
        self._init_model()
        load_path = self.output_dir / filename
        if not load_path.exists():
            print(f"[WARN] Checkpoint not found: {load_path}, skipping load")
            return

        ckpt = torch.load(load_path, map_location=self.device)
        self.encoder.object_proj.load_state_dict(ckpt["object_proj"])
        self.encoder.relationship_proj.load_state_dict(ckpt["relationship_proj"])
        self.encoder.scene_proj.load_state_dict(ckpt["scene_proj"])
        print(f"[TwoStageTrainer] Loaded from {load_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="data/training/tree_training_data.jsonl",
                        help="训练数据路径")
    parser.add_argument("--output", type=str,
                        default="outputs",
                        help="输出目录")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3],
                        default=3,
                        help="训练阶段: 1=Stage1, 2=Stage2, 3=both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    trainer = TwoStageTrainer(
        training_data_path=args.data,
        output_dir=args.output,
    )

    if args.stage == 1:
        trainer.train_stage1(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    elif args.stage == 2:
        trainer.train_stage2(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    else:
        trainer.train_stage1(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        trainer.train_stage2(epochs=max(3, args.epochs // 2),
                             lr=args.lr / 2, batch_size=args.batch_size)

    # 保存训练历史
    history_path = Path(args.output) / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.train_history, f, ensure_ascii=False, indent=2)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()