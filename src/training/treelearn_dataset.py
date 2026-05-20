"""
TreeLearn TLS 森林点云数据集加载器
https://github.com/Weizheng-NY/TreeLearn

支持：
  - PLY 格式（含 semantic + instance 标签）
  - 训练时随机裁剪 / 数据增强
  - 语义分割（3类）+ 实例分割标签

标签说明（TreeLearn）：
  semantic:
    0 = ground   地面
    1 = trunk    树干
    2 = crown    树冠
    3 = other    其他（辅助类别）

  instance_id: 每棵树的唯一ID（0 = ground，不属于任何树）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random


def parse_ply_with_labels(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析带标签的 PLY 文件（TreeLearn 格式）

    Returns:
        points: (N, 3) xyz 坐标
        semantic: (N,) 语义标签 0=ground 1=trunk 2=crown 3=other
        instance: (N,) 实例标签（每棵树唯一ID）
    """
    points_list = []
    semantic_list = []
    instance_list = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    # 找到 face/vertex 数据起始位置
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break

    # 解析 header 确定列索引
    header = lines[:header_end]
    data_lines = lines[header_end:]

    # 确定 property 顺序
    has_rgb = any("property uchar r" in l or "property uchar red" in l for l in header)
    has_normal = any("property float nx" in l for l in header)
    has_semantic = any("property uchar semantic" in l or "property uint semantic" in l for l in header)
    has_instance = any("property uint instance_id" in l or "property uchar instance_id" in l for l in header)

    data_format = "binary" if any("binary" in l for l in header) else "ascii"

    if data_format == "ascii":
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points_list.append([x, y, z])

            offset = 3
            if has_rgb:
                offset += 3
            if has_normal:
                offset += 3

            if has_semantic and has_instance:
                sem = int(parts[offset])
                ins = int(parts[offset + 1])
                semantic_list.append(sem)
                instance_list.append(ins)
    else:
        import struct
        # binary little endian
        # 计算格式
        fmt_items = []
        for line in header:
            line = line.strip()
            if line.startswith("property"):
                parts = line.split()
                ptype = parts[1]
                if ptype == "float":
                    fmt_items.append("f")
                elif ptype == "double":
                    fmt_items.append("d")
                elif ptype == "int" or ptype == "uint" or ptype == "uchar":
                    fmt_items.append("B" if ptype == "uchar" else "I")
                elif ptype == "uchar":
                    fmt_items.append("B")
        fmt_str = "<" + "".join(fmt_items)
        item_size = struct.calcsize(fmt_str)

        for line in data_lines[:]:
            if line.strip() == "":
                continue
            # binary 模式：跳过文字行
            if not line.startswith(" " .encode()) and line[0] not in b" \t\n":
                # 找第一个数字的位置
                continue
            try:
                data = line.strip()
                vals = list(struct.unpack(fmt_str, data[:item_size]))
                points_list.append([vals[0], vals[1], vals[2]])

                offset = 3
                if has_rgb:
                    offset += 3
                if has_normal:
                    offset += 3

                if has_semantic and has_instance and len(vals) > offset + 1:
                    semantic_list.append(vals[offset])
                    instance_list.append(vals[offset + 1])
            except Exception:
                continue

    points = np.array(points_list, dtype=np.float32)
    semantic = np.array(semantic_list, dtype=np.int64) if semantic_list else np.zeros(len(points), dtype=np.int64)
    instance = np.array(instance_list, dtype=np.int64) if instance_list else np.zeros(len(points), dtype=np.int64)

    return points, semantic, instance


class TreeLearnDataset(Dataset):
    """
    TreeLearn 数据集加载器

    支持模式：
      - semantic: 语义分割（3分类：ground/trunk/crown）
      - instance: 实例分割（tree_id）
      - both: 同时返回

    数据增强（训练模式）：
      - 随机裁剪（固定点数）
      - 随机旋转（绕 Z 轴）
      - 随机缩放
      - 随机平移
    """

    CLASSES = ["ground", "trunk", "crown", "other"]
    NUM_CLASSES = 4

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_points: int = 8192,
        use_augmentation: bool = True,
        transform_mode: str = "semantic",  # "semantic" | "instance" | "both"
        cache_in_memory: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.use_augmentation = use_augmentation and (split == "train")
        self.transform_mode = transform_mode
        self.cache_in_memory = cache_in_memory

        # 加载文件列表
        self.samples: List[Dict] = []
        self._load_file_list()

        # 内存缓存
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _load_file_list(self):
        """扫描 root 目录下的所有 PLY 文件"""
        if not self.root.exists():
            raise FileNotFoundError(f"TreeLearn root not found: {self.root}")

        ply_files = sorted(self.root.rglob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No .ply files found in {self.root}")

        for ply_file in ply_files:
            self.samples.append({
                "path": str(ply_file),
                "name": ply_file.stem,
            })

        print(f"[TreeLearnDataset] Found {len(self.samples)} PLY files in {self.root}")

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载单个 PLY 样本"""
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        path = sample["path"]

        try:
            points, semantic, instance = parse_ply_with_labels(path)
        except Exception as e:
            print(f"[TreeLearnDataset] Failed to parse {path}: {e}")
            points = np.zeros((100, 3), dtype=np.float32)
            semantic = np.zeros(100, dtype=np.int64)
            instance = np.zeros(100, dtype=np.int64)

        if self.cache_in_memory:
            self._cache[idx] = (points, semantic, instance)

        return points, semantic, instance

    def _subsample(self, points: np.ndarray, semantic: np.ndarray, instance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """随机/均匀降采样到 num_points"""
        n = len(points)
        if n == self.num_points:
            return points, semantic, instance

        if n > self.num_points:
            # 优先保留所有 trunk 点，再均匀降采样其他点
            trunk_mask = semantic == 1
            trunk_pts = points[trunk_mask]
            trunk_sem = semantic[trunk_mask]
            trunk_ins = instance[trunk_mask]

            other_mask = ~trunk_mask
            other_pts = points[other_mask]
            other_sem = semantic[other_mask]
            other_ins = instance[other_mask]

            # trunk 点全保留，最多保留 num_points // 4
            max_trunk = self.num_points // 4
            n_trunk = min(len(trunk_pts), max_trunk)
            if n_trunk > 0 and len(trunk_pts) > n_trunk:
                chosen = np.random.choice(len(trunk_pts), n_trunk, replace=False)
                trunk_pts, trunk_sem, trunk_ins = trunk_pts[chosen], trunk_sem[chosen], trunk_ins[chosen]

            # 其余给 other
            remaining = self.num_points - n_trunk
            if len(other_pts) > remaining:
                chosen = np.random.choice(len(other_pts), remaining, replace=False)
                other_pts, other_sem, other_ins = other_pts[chosen], other_sem[chosen], other_ins[chosen]

            points = np.concatenate([trunk_pts, other_pts], axis=0)
            semantic = np.concatenate([trunk_sem, other_sem], axis=0)
            instance = np.concatenate([trunk_ins, other_ins], axis=0)
            return points, semantic, instance
        else:
            # 重复采样补齐
            indices = np.random.choice(n, self.num_points, replace=True)
            return points[indices], semantic[indices], instance[indices]

    def _augment(self, points: np.ndarray) -> np.ndarray:
        """数据增强：旋转 + 缩放 + 平移"""
        # 随机旋转（绕 Z 轴）
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
        points = points @ rot.T

        # 随机缩放 [0.8, 1.2]
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        # 随机平移（小幅）
        shift = np.random.randn(3).astype(np.float32) * 0.05
        points = points + shift

        return points

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """质心归一化 + 单位球归一化"""
        centroid = points.mean(axis=0, keepdims=True)
        pts = points - centroid
        scale = np.linalg.norm(pts, axis=1).max()
        scale = max(scale, 1e-8)
        pts = pts / scale
        return pts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        points, semantic, instance = self._load_sample(idx)

        # 预处理
        points = self._normalize(points.astype(np.float32))
        points, semantic, instance = self._subsample(points, semantic, instance)

        # 数据增强
        if self.use_augmentation:
            points = self._augment(points)

        # 标签映射：ground=0, trunk=1, crown=2, other→3
        # TreeLearn 原始标签：0=ground, 1=trunk, 2=crown, 3=other（已是目标格式）

        result = {
            "points": torch.from_numpy(points.astype(np.float32)),          # (N, 3)
            "semantic": torch.from_numpy(semantic).long(),                 # (N,)
            "instance": torch.from_numpy(instance).long(),                  # (N,)
        }

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """自定义 batch 收集函数"""
    return {
        "points": torch.stack([b["points"] for b in batch]),      # (B, N, 3)
        "semantic": torch.stack([b["semantic"] for b in batch]), # (B, N)
        "instance": torch.stack([b["instance"] for b in batch]), # (B, N)
    }


def test_dataset():
    """快速测试"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/TreeLearn")
    args = parser.parse_args()

    try:
        ds = TreeLearnDataset(args.root, split="train", num_points=8192)
        print(f"Samples: {len(ds)}")

        sample = ds[0]
        print(f"Points: {sample['points'].shape}")
        print(f"Semantic: {sample['semantic'].shape}, unique={sample['semantic'].unique().tolist()}")
        print(f"Instance: {sample['instance'].shape}, unique count={sample['instance'].unique().nelement()}")

        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"Batch points: {batch['points'].shape}")
        print(f"Batch semantic: {batch['semantic'].shape}")
        print("[OK] TreeLearnDataset test passed")
    except FileNotFoundError as e:
        print(f"[SKIP] TreeLearn dataset not found: {e}")
        print("Download from: https://github.com/Weizheng-NY/TreeLearn")
        print("Place in: data/TreeLearn/")


if __name__ == "__main__":
    test_dataset()
