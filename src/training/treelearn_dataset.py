"""
TreeLearn TLS 森林点云数据集加载器
https://github.com/ecker-lab/TreeLearn

支持：
  - .laz / .las 格式（TreeLearn 官方格式，Lidar360 标注）
  - PLY 格式（含 semantic + instance 标签，第三方格式）
  - 训练时随机裁剪 / 数据增强
  - 语义分割（3类）+ 实例分割标签

LAZ/LAS 标签说明（Lidar360）：
  classification:
    2 = ground   地面
    3 = trunk    树干（TLS 主干）
    4 = crown    树冠
  treeID: 每棵树的唯一ID（0 = ground，不属于任何树）

PLY 标签说明（第三方格式）：
  semantic:
    0 = ground   地面
    1 = trunk    树干
    2 = crown    树冠
    3 = other    其他
  instance_id: 每棵树的唯一ID（0 = ground，不属于任何树）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random


def parse_npy_with_labels(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析预处理的 NPY 文件（由 preprocess_treelearn.py 生成）

    NPY 格式: (N, 5) float32 array
      col 0-2: xyz coordinates
      col 3: semantic label (0=ground, 1=trunk, 2=crown)
      col 4: instance ID (treeID)

    Returns:
        points: (N, 3) xyz
        semantic: (N,) int64
        instance: (N,) int64
    """
    arr = np.load(filepath, allow_pickle=False)
    points = arr[:, :3].astype(np.float32)
    semantic = arr[:, 3].astype(np.int64)
    instance = arr[:, 4].astype(np.int64)
    return points, semantic, instance


def parse_las_with_labels(filepath: str, max_points: int = 200000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析 LAZ/LAS 文件（TreeLearn 官方 Lidar360 标注格式）
    对大文件做在线降采样，控制内存占用

    Lidar360 标签映射：
      classification:
        2 = ground  (地面)
        3 = trunk   (树干)
        4 = crown  (树冠)
      treeID: 每棵树唯一ID（0 = ground，不属于任何树）

    Returns:
        points: (N, 3) xyz 坐标（已降采样到 max_points）
        semantic: (N,) 语义标签 0=ground 1=trunk 2=crown
        instance: (N,) 实例标签（每棵树唯一ID）
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy required for LAZ/LAS files: pip install laspy")

    las = laspy.read(filepath)
    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    n_total = len(x)

    # 大文件在线降采样：随机采样 max_points 个点
    if n_total > max_points:
        idx = np.random.RandomState().choice(n_total, max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        cls_raw = np.array(las.classification, dtype=np.int32)[idx]
        try:
            instance = np.array(las.treeID, dtype=np.int64)[idx]
        except Exception:
            instance = np.zeros(max_points, dtype=np.int64)
    else:
        cls_raw = np.array(las.classification, dtype=np.int32)
        try:
            instance = np.array(las.treeID, dtype=np.int64)
        except Exception:
            instance = np.zeros(n_total, dtype=np.int64)

    points = np.stack([x, y, z], axis=1)

    # Lidar360 → 标准语义标签
    mapping = {2: 0, 3: 1, 4: 2}
    semantic = np.zeros(len(cls_raw), dtype=np.int64)
    for orig, new in mapping.items():
        semantic[cls_raw == orig] = new

    return points, semantic, instance


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
        cache_in_memory: bool = False,
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
        """扫描 root 目录下的所有 .npy / .laz / .las / .ply 文件"""
        if not self.root.exists():
            raise FileNotFoundError(f"TreeLearn root not found: {self.root}")

        # NPY 优先（预处理过的，加载最快）
        npy_files = sorted(self.root.rglob("*.npy"))
        laz_files = sorted(self.root.rglob("*.laz"))
        las_files = sorted(self.root.rglob("*.las"))
        ply_files = sorted(self.root.rglob("*.ply"))

        for f in npy_files:
            self.samples.append({"path": str(f), "name": f.stem, "fmt": "npy"})
        for f in laz_files + las_files:
            if not any(s["name"] == f.stem for s in self.samples):
                self.samples.append({"path": str(f), "name": f.stem, "fmt": "las"})
        for f in ply_files:
            if not any(s["name"] == f.stem for s in self.samples):
                self.samples.append({"path": str(f), "name": f.stem, "fmt": "ply"})

        print(f"[TreeLearnDataset] Found {len(self.samples)} point cloud files in {self.root}")

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载单个样本（支持 NPY / LAZ/LAS / PLY 格式）"""
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        path = sample["path"]

        try:
            fmt = sample.get("fmt", "")
            if fmt == "npy":
                points, semantic, instance = parse_npy_with_labels(path)
            elif fmt == "las":
                points, semantic, instance = parse_las_with_labels(path)
            else:
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

        # 标签映射：
        #   TreeLearn 自动标注数据只有 ground(0) 和 crown(2)，没有 trunk
        #   将 crown(2) 映射为 tree(1)，用于 2 类分割（ground vs tree）
        semantic[semantic == 2] = 1

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
    parser.add_argument("--root", type=str, default="data/TreeLearn/data/train/forests")
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
        print("Download from: https://github.com/ecker-lab/TreeLearn")
        print("Place .laz files in: data/TreeLearn/data/train/forests/")


if __name__ == "__main__":
    test_dataset()
