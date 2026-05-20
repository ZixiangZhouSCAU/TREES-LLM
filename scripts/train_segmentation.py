"""
PointNet++ 单木分割模型训练脚本

功能：
  - 从 TreeLearn TLS 数据集训练语义分割模型
  - Stage 1: 语义分割（CrossEntropy）
  - 保存 best_model.pt，支持在 service.py 中加载

使用：
  # 下载 TreeLearn 数据
  python scripts/download_treelearn.py  # 需要手动下载

  # 训练（语义分割）
  python scripts/train_segmentation.py --epochs 30 --lr 1e-3 --batch-size 4

  # 验证
  python scripts/train_segmentation.py --mode eval --checkpoint outputs/best_segmentation_model.pt

  # 测试推理
  python scripts/train_segmentation.py --mode demo --checkpoint outputs/best_segmentation_model.pt --input test.ply
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tree_segmentation import TreeSegmentationModel
from src.training.treelearn_dataset import TreeLearnDataset, collate_fn


# ============ 训练配置 ============

def get_args():
    parser = argparse.ArgumentParser(description="PointNet++ 单木分割训练")
    parser.add_argument("--data-root", type=str, default="data/TreeLearn/data/train/forests",
                        help="TreeLearn 数据集根目录（含 .laz 文件）")
    parser.add_argument("--output-dir", type=str, default="outputs/segmentation",
                        help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="加载已有 checkpoint 继续训练")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=8192,
                        help="每样本点数")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-interval", type=int, default=5,
                        help="每 N 个 epoch 验证一次")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="每 N 个 epoch 保存一次")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "demo"],
                        help="train=训练, eval=验证, demo=演示")
    parser.add_argument("--input", type=str, default=None,
                        help="demo 模式输入点云文件")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


# ============ 损失函数 ============

class SegmentationLoss(nn.Module):
    """
    语义分割损失：带类别权重
    ground(0) vs tree(1) 二分类
    """

    def __init__(self, num_classes: int = 2, ignore_label: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.register_buffer("weight", torch.tensor([1.0, 1.5]))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, N, C) logits
        target: (B, N)  label
        """
        B, N, C = pred.shape
        pred = pred.reshape(B * N, C)
        target = target.reshape(B * N)

        # 忽略特殊标签
        mask = (target != self.ignore_label) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]

        if mask.sum() == 0:
            return pred.sum() * 0

        return F.cross_entropy(pred, target, weight=self.weight.to(pred.device))


# ============ 评估指标 ============

def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> dict:
    """计算每类 IoU 和平均 IoU"""
    ious = {}
    for c in range(num_classes):
        pred_c = pred == c
        tgt_c = target == c
        intersection = (pred_c & tgt_c).sum()
        union = (pred_c | tgt_c).sum()
        ious[c] = intersection / max(union, 1)

    ious["mean"] = sum(v for k, v in ious.items() if k != "mean") / num_classes
    return ious


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device, num_classes: int = 2) -> dict:
    """在验证集上评估"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = SegmentationLoss(num_classes=num_classes)

    with torch.no_grad():
        for batch in val_loader:
            points = batch["points"].to(device)          # (B, N, 3)
            semantic = batch["semantic"].to(device)      # (B, N)

            out = model(points)
            logits = out["semantic_logits"]              # (B, N, 3)

            loss = criterion(logits, semantic)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)                # (B, N)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(semantic.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算 IoU（忽略 ignore_label=-1）
    valid_mask = all_targets >= 0
    ious = compute_iou(all_preds[valid_mask], all_targets[valid_mask], num_classes)

    n_samples = len(val_loader)
    avg_loss = total_loss / n_samples

    return {
        "val_loss": avg_loss,
        "ious": ious,
        "accuracy": (all_preds[valid_mask] == all_targets[valid_mask]).mean(),
    }


# ============ 训练循环 ============

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """单 epoch 训练"""
    model.train()
    total_loss = 0
    n_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        points = batch["points"].to(device)          # (B, N, 3)
        semantic = batch["semantic"].to(device)      # (B, N)

        optimizer.zero_grad()
        out = model(points)
        logits = out["semantic_logits"]              # (B, N, 3)

        loss = criterion(logits, semantic)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{n_batches}] Loss: {loss.item():.4f}")

    return total_loss / n_batches


def train(args):
    """主训练流程"""
    device = torch.device(args.device) if args.device != "auto" else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Train] Device: {device}")
    print(f"[Train] Data root: {args.data_root}")
    print(f"[Train] Output: {args.output_dir}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查数据
    if not Path(args.data_root).exists():
        print(f"[ERROR] TreeLearn dataset not found at {args.data_root}")
        print("Download from: https://github.com/Weizheng-NY/TreeLearn")
        print("Tip: Set --data-root to your TreeLearn folder path")
        return

    # 数据集
    try:
        full_dataset = TreeLearnDataset(
            root=args.data_root,
            split="train",
            num_points=args.num_points,
            use_augmentation=True,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # 简单划分：80% train, 20% val
    n_samples = len(full_dataset)
    n_train = int(n_samples * 0.8)
    indices = np.random.RandomState(42).permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    class SubDataset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.dataset[self.indices[i]]

    train_dataset = SubDataset(full_dataset, train_idx)
    val_dataset = SubDataset(full_dataset, val_idx)

    print(f"[Train] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 模型（2类：ground vs tree）
    model = TreeSegmentationModel(num_classes=2).to(device)
    print(f"[Model] Parameters: {model.num_params():,}")

    # 加载 checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch = state.get("epoch", 0)
        best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"[Checkpoint] Loaded from {args.checkpoint}, epoch {start_epoch}")

    # 优化器 + 学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = SegmentationLoss(num_classes=2)

    # 训练循环
    print(f"[Train] Starting from epoch {start_epoch}, total {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 验证
        if (epoch + 1) % args.val_interval == 0:
            metrics = evaluate_model(model, val_loader, device)
            print(f"  Val Loss: {metrics['val_loss']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}, "
                  f"mIoU: {metrics['ious']['mean']:.4f}")
            print(f"  IoU: ground={metrics['ious'][0]:.3f}, "
                  f"tree={metrics['ious'][1]:.3f}")

            # 保存 best
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_path = output_dir / "best_segmentation_model.pt"
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "config": {
                        "num_classes": 2,
                        "instance_feat_dim": 32,
                        "num_points": args.num_points,
                    }
                }, best_path)
                print(f"  [Saved] best_segmentation_model.pt (val_loss={best_val_loss:.4f})")

        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, ckpt_path)

    print(f"[Done] Training complete. Best val_loss: {best_val_loss:.4f}")


def eval_model(args):
    """验证模式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    if not args.checkpoint or not Path(args.checkpoint).exists():
        print("[ERROR] --checkpoint required for eval mode")
        return

    model = TreeSegmentationModel(num_classes=2).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"[Eval] Loaded: {args.checkpoint}")

    # 数据集
    if not Path(args.data_root).exists():
        print(f"[ERROR] Dataset not found: {args.data_root}")
        return

    dataset = TreeLearnDataset(args.data_root, split="train", num_points=args.num_points, use_augmentation=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    metrics = evaluate_model(model, loader, device, num_classes=2)
    print(f"\n[Eval Results]")
    print(f"Val Loss: {metrics['val_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"IoU ground:  {metrics['ious'][0]:.4f}")
    print(f"IoU tree:     {metrics['ious'][1]:.4f}")


def demo_segmentation(args):
    """演示模式：对单个点云进行分割"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Demo] Device: {device}")

    if not args.checkpoint or not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("Train first: python scripts/train_segmentation.py --epochs 30")
        return

    model = TreeSegmentationModel(num_classes=4).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[Demo] Model loaded from {args.checkpoint}")

    # 加载点云
    if not args.input:
        # 使用默认测试文件
        test_ply = Path("H:/1reserch/02lidarsplatting/data/tree2/colmap_runbasicsfm/sparse/0/points3D.ply")
        if test_ply.exists():
            args.input = str(test_ply)
        else:
            print("[ERROR] --input required")
            return
    else:
        test_ply = Path(args.input)

    if not test_ply.exists():
        print(f"[ERROR] Input file not found: {test_ply}")
        return

    print(f"[Demo] Loading: {test_ply}")

    # 解析 PLY（简单版，只取 xyz）
    points = []
    with open(test_ply, "r") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            data_start = i + 1
            break

    has_rgb = any("uchar r" in l for l in lines[:data_start])
    has_normal = any("float nx" in l for l in lines[:data_start])

    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])
        except:
            continue

    points = np.array(points, dtype=np.float32)
    print(f"[Demo] Loaded {len(points)} points")

    # 语义分割
    with torch.no_grad():
        points_tensor = torch.from_numpy(points).unsqueeze(0).to(device)
        semantic = model.predict_semantic(points_tensor, device=str(device))
        semantic = semantic[0]  # (N,)

    unique, counts = np.unique(semantic, return_counts=True)
    print(f"[Semantic] Predicted classes:")
    class_names = ["ground", "trunk", "crown", "other"]
    for u, c in zip(unique, counts):
        name = class_names[u] if u < 4 else f"class_{u}"
        print(f"  {name}: {c} points ({100*c/len(points):.1f}%)")

    # 实例分割
    trees = model.predict_instances(points, eps=0.3, min_samples=20, device=str(device))
    print(f"\n[Instance] Detected {len(trees)} trees")
    for i, tree in enumerate(trees[:5]):
        print(f"  Tree {i}: {len(tree)} points")

    # 保存可视化标签
    output_npy = test_ply.with_suffix(".labels.npy")
    np.save(output_npy, semantic)
    print(f"[Demo] Labels saved to {output_npy}")


if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_model(args)
    elif args.mode == "demo":
        demo_segmentation(args)
