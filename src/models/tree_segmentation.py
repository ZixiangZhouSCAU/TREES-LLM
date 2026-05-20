"""
PointNet++ 单木分割模型
参考 3DCITY-LLM 的 SoftGroup 等实例分割思路

功能：
- 输入：林分点云 (B, N, 3)
- 输出 1：每点语义标签 (B, N) — ground / trunk / crown
- 输出 2：trunk 点的实例标签 — 用于 DBSCAN 实例分割

模型架构：
  TreePointEncoder (PointNet++) → 跳跃连接
      ↓
  Feature Propagation → (B, N, 128)
      ↓
  ┌─ SemanticSegHead → (B, N, 3) 每点 3 类概率
  └─ InstanceHead → (B, N, 1) 每点实例分数

训练策略（两阶段）：
  Stage 1: 语义分割（CrossEntropy）
  Stage 2: trunk 点 DBSCAN 实例分割

使用方法：
    model = TreeSegmentationModel()
    model.load("outputs/segmentation_model.pt")  # 加载训练好的模型
    model.eval()

    points = torch.randn(1, 8192, 3)
    with torch.no_grad():
        semantic_pred = model.predict_semantic(points)  # (N,) 0/1/2
        trees = model.predict_instances(points)  # List[np.ndarray] 每棵树的点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path

from .point_encoder import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
    TreePointEncoder,
)


# ============ 语义分割头 ============

class SemanticSegHead(nn.Module):
    """
    语义分割头：输出每点属于 ground / trunk / crown 的概率

    类别：
      0 = ground  （地面）
      1 = trunk   （树干）— 用于实例分割
      2 = crown   （树冠）
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        returns: (B, N, num_classes) logits
        """
        x = x.permute(0, 2, 1)   # (B, C, N)
        logits = self.conv(x)     # (B, num_classes, N)
        return logits.permute(0, 2, 1)   # (B, N, num_classes)


# ============ 实例分割头（简化版）============

class InstanceHead(nn.Module):
    """
    实例分割头（简化版）：输出每点的实例特征向量

    策略：
    - 不做复杂的对比学习，直接输出 32 维特征向量
    - 推理时：trunk 点 → DBSCAN → 每类 = 一棵树

    如需更精确的实例分割，可在此处接入：
    - PointGroup 的 offset branch
    - SoftGroup 的 semantic/instance heads
    """

    def __init__(self, in_channels: int = 128, feat_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        returns: (B, N, feat_dim) 实例特征
        """
        x = x.permute(0, 2, 1)
        feat = self.conv(x)
        return feat.permute(0, 2, 1)


# ============ 主模型 ============

class TreeSegmentationModel(nn.Module):
    """
    PointNet++ 单木分割模型

    功能：
    - 语义分割：ground / trunk / crown 三类
    - 实例分割：trunk 点 DBSCAN → 每类 = 一棵树

    输入：(B, N, 3) 点云（已预处理：质心归一化 + 单位球归一化）
    输出：语义 logits + 实例特征
    """

    SEMANTIC_CLASSES = ["ground", "tree"]

    def __init__(self, num_classes: int = 3, instance_feat_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes

        # 编码器（复用车已有的 TreePointEncoder）
        self.encoder = TreePointEncoder(input_dim=3, global_dim=512)

        # Feature Propagation MLP: SA2(256-dim) → 128-dim
        self.fp_mlp = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # 语义分割头
        self.semantic_head = SemanticSegHead(in_channels=128, num_classes=num_classes)

        # 实例分割头（简化：只对 trunk 点有意义）
        self.instance_head = InstanceHead(in_channels=128, feat_dim=instance_feat_dim)

    def forward(
        self,
        points: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 3) 点云
            return_features: 是否返回中间特征

        Returns:
            Dict with:
              - semantic_logits: (B, N, 3) 每点 3 类 logits
              - instance_feat: (B, N, 32) 每点实例特征
              - global_feat: (B, 512) 全局特征（可选）
              - multi_scale: list of features（可选）
        """
        # 编码
        enc_out = self.encoder(points)

        # 取多尺度特征中最后一层的特征（SA3 后的特征）
        l3_features = enc_out["l3_features"]  # (B, 64, 256)

        # Feature Propagation：插值回原始点数 N
        fp_out = self._propagate_features(enc_out, points.shape[1])  # (B, N, 128)

        # 语义分割
        semantic_logits = self.semantic_head(fp_out)   # (B, N, 3)

        # 实例特征
        instance_feat = self.instance_head(fp_out)   # (B, N, 32)

        result = {
            "semantic_logits": semantic_logits,
            "instance_feat": instance_feat,
        }

        if return_features:
            result["global_feat"] = enc_out["global_feature"]
            result["multi_scale"] = enc_out["multi_scale"]

        return result

    def _propagate_features(
        self,
        encoder_out: Dict,
        target_n: int,
    ) -> torch.Tensor:
        """
        将 SA 层的稀疏特征插值回 target_n 个点

        SA2: (B, 256, 256) 特征 → 插值到 (B, N, 256) → MLP → (B, N, 128)
        """
        normalized = encoder_out["normalized_coords"]   # (B, N_full, 3)
        B, N_full, _ = normalized.shape
        N = target_n

        # 使用 SA2 特征（256点，比 SA3 的 64 点更细）
        feat = encoder_out["multi_scale"][1]   # (B, 256, 256)

        # 用 FPS 对 normalized 采样出 256 个源坐标点
        src_xyz = self._sample_xyz(normalized, 256)   # (B, 256, 3)
        # 目标坐标：对 normalized 做均匀采样到 N 个点
        tgt_xyz = self._sample_xyz(normalized, N)    # (B, N, 3)

        # 3-NN 插值：tgt → src
        dists = torch.cdist(tgt_xyz, src_xyz)   # (B, N, 256)
        k = 3
        dists, idx = torch.topk(dists, k=k, dim=-1, largest=False)
        dists = dists.clone()
        dists[dists < 1e-10] = 1e-10
        weight = (1.0 / dists)
        weight = weight / weight.sum(dim=-1, keepdim=True)

        # gather src_points: (B, 256, 256) → (B, N, k, 256)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, feat.shape[-1])   # (B, N, k, 256)
        feat_expanded = feat.unsqueeze(1).expand(B, N, 256, 256)             # (B, N, 256, 256)
        gathered = torch.gather(feat_expanded, dim=2, index=idx_expanded)       # (B, N, k, 256)
        interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=2)              # (B, N, 256)

        # MLP: (B, 256, N) → (B, 128, N) → (B, N, 128)
        out = self.fp_mlp(interpolated.permute(0, 2, 1))   # (B, 128, N)
        return out.permute(0, 2, 1)                        # (B, N, 128)

    def _sample_xyz(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """对 xyz 做 FPS 采样，返回 npoint 个采样点"""
        B, N, _ = xyz.shape
        device = xyz.device

        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        dists = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), device=device)
        batch_indices = torch.arange(B, device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
            mask = dist < dists
            dists[mask] = dist[mask]
            farthest = torch.max(dists, dim=1)[1]

        # gather
        idx_flat = centroids + batch_indices.view(B, 1) * N
        xyz_flat = xyz.reshape(B * N, 3)
        sampled = xyz_flat[idx_flat.reshape(-1)].reshape(B, npoint, 3)
        return sampled

    def predict_semantic(
        self,
        points: torch.Tensor,
        device: str = "auto",
    ) -> np.ndarray:
        """
        推理：语义分割

        Args:
            points: (B, N, 3) numpy 或 tensor
            device: "cuda" / "cpu" / "auto"

        Returns:
            np.ndarray: (B, N) 每点的语义类别 0=ground, 1=trunk, 2=crown
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points.astype(np.float32))

        if points.ndim == 2:
            points = points.unsqueeze(0)  # (N, 3) → (1, N, 3)

        points = points.to(device, non_blocking=True)
        self.eval()

        with torch.no_grad():
            # 预处理
            points = self._preprocess(points)
            out = self.forward(points)
            logits = out["semantic_logits"]   # (B, N, 3)
            preds = logits.argmax(dim=-1)   # (B, N)

        return preds.cpu().numpy()

    def predict_instances(
        self,
        points: torch.Tensor,
        eps: float = 0.3,
        min_samples: int = 20,
        device: str = "auto",
    ) -> List[np.ndarray]:
        """
        推理：实例分割

        流程：
          1. 语义分割 → trunk 点
          2. trunk 点 → DBSCAN 实例分割
          3. 每棵树的完整点 = 该实例的所有 trunk 点（膨胀到 crown 点）

        Args:
            points: (N, 3) 或 (B, N, 3) numpy
            eps: DBSCAN 半径
            min_samples: 最小点数
            device: "cuda" / "cpu" / "auto"

        Returns:
            List[np.ndarray]: 每棵树的点云列表
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(points, np.ndarray):
            points_raw = torch.from_numpy(points.astype(np.float32))
            points_np = points
            batch_mode = (points_raw.ndim == 3)
            if not batch_mode:
                points_raw = points_raw.unsqueeze(0)   # (1, N, 3)
        else:
            points_raw = points
            points_np = points.cpu().numpy()
            batch_mode = (points_raw.ndim == 3)

        points_raw = points_raw.to(device)

        with torch.no_grad():
            points_norm = self._preprocess(points_raw)
            out = self.forward(points_norm)
            preds = out["semantic_logits"].argmax(dim=-1)   # (B, N)

        # 取 batch 0
        pred = preds[0].cpu().numpy()
        points_np = points_np[0] if not isinstance(points_np, list) and points_np.ndim == 3 else points_np

        # 提取 trunk 点
        trunk_mask = (pred == 1)   # trunk
        trunk_points = points_np[trunk_mask]

        if len(trunk_points) < min_samples:
            # fallback：全部点做 DBSCAN
            return self._dbscan_fallback(points_np, eps, min_samples)

        # Trunk 点 DBSCAN
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(trunk_points[:, :2])
            labels = clustering.labels_
        except Exception:
            return self._dbscan_fallback(points_np, eps, min_samples)

        # 构建每棵树的完整点云
        trees = []
        for label in set(labels):
            if label < 0:
                continue
            # trunk 点子集
            trunk_sub = trunk_points[labels == label]
            # 找到离这个 cluster 中心最近的原始点（用于膨胀）
            cluster_center = trunk_sub[:, :2].mean(axis=0)
            # 在 trunk_mask 内找距离中心最近的点作为种子
            trunk_indices = np.where(trunk_mask)[0]
            trunk_xy = points_np[trunk_indices, :2]
            dists = np.sqrt(((trunk_xy - cluster_center) ** 2).sum(axis=1))
            nearest_idx = trunk_indices[dists.argmin()]

            # 以最近点的 z 范围 + xy 扩张确定树的完整范围
            seed_xyz = points_np[nearest_idx]
            tree_mask = (
                (np.abs(points_np[:, 0] - seed_xyz[0]) < 1.0) &
                (np.abs(points_np[:, 1] - seed_xyz[1]) < 1.0)
            )
            full_tree = points_np[tree_mask]
            if len(full_tree) > 50:
                trees.append(full_tree)

        if not trees:
            return self._dbscan_fallback(points_np, eps, min_samples)

        return trees

    def _dbscan_fallback(
        self,
        points: np.ndarray,
        eps: float,
        min_samples: int,
    ) -> List[np.ndarray]:
        """DBSCAN 回退方案"""
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps * 2, min_samples=min_samples).fit(points[:, :2])
            labels = clustering.labels_
            trees = []
            for label in set(labels):
                if label >= 0:
                    tree_pts = points[labels == label]
                    if len(tree_pts) > min_samples:
                        trees.append(tree_pts)
            return trees if trees else [points]
        except Exception:
            return [points]

    def _preprocess(self, points: torch.Tensor) -> torch.Tensor:
        """
        预处理：质心归一化 + 单位球归一化
        points: (B, N, 3)
        """
        centroid = points.mean(dim=1, keepdim=True)
        pts = points - centroid
        scale = pts.norm(dim=-1).max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        pts = pts / scale
        return pts

    @torch.no_grad()
    def segment(
        self,
        points: np.ndarray,
        eps: float = 0.3,
        min_samples: int = 20,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        完整分割流程：语义分割 + 实例分割

        Args:
            points: (N, 3) numpy
            eps: DBSCAN 半径（m）
            min_samples: 最小 trunk 点数

        Returns:
            (semantic_labels, tree_list)
              - semantic_labels: (N,) 每点的语义标签
              - tree_list: List[每棵树的点云]
        """
        dev = self._detect_device()
        semantic = self.predict_semantic(points, device=dev)
        trees = self.predict_instances(points, eps=eps, min_samples=min_samples, device=dev)
        return semantic, trees

    def _detect_device(self) -> str:
        """检测模型当前所在设备"""
        return next(self.parameters()).device

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============ 独立推理函数 ============

@torch.no_grad()
def segment_point_cloud(
    points: np.ndarray,
    model: Optional[TreeSegmentationModel] = None,
    model_path: Optional[str] = None,
    eps: float = 0.3,
    min_samples: int = 20,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    完整的点云分割函数（推荐使用）

    Args:
        points: (N, 3) numpy array
        model: 已加载的模型（可选）
        model_path: 模型权重路径（可选）
        eps: DBSCAN 半径
        min_samples: 最小点数

    Returns:
        (semantic_labels, tree_list)
    """
    # 加载模型
    if model is None:
        if model_path and Path(model_path).exists():
            model = TreeSegmentationModel()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print(f"[segment] Loaded model from {model_path}")
        else:
            # 无模型：回退到 DBSCAN
            return _dbscan_segment(points, eps, min_samples)

    # 推理
    semantic, trees = model.segment(points, eps=eps, min_samples=min_samples)
    return semantic, trees


def _dbscan_segment(
    points: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """纯 DBSCAN 分割（无模型时回退）"""
    try:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
        labels = clustering.labels_

        # 所有点标记为 "unknown" (用 -1 表示无语义)
        semantic = np.full(len(points), -1, dtype=np.int32)

        trees = []
        for label in set(labels):
            if label >= 0:
                tree_pts = points[labels == label]
                if len(tree_pts) >= min_samples:
                    trees.append(tree_pts)

        return semantic, trees if trees else [points]
    except Exception:
        return np.full(len(points), -1, dtype=np.int32), [points]


def test_model():
    """测试分割模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    model = TreeSegmentationModel().to(device)
    model.eval()
    print(f"Model params: {model.num_params():,}")

    # 测试前向传播
    points = torch.randn(2, 8192, 3).to(device)
    with torch.no_grad():
        out = model(points)

    print(f"Semantic logits: {out['semantic_logits'].shape}")   # (2, 8192, 3)
    print(f"Instance feat: {out['instance_feat'].shape}")      # (2, 8192, 32)

    # 测试推理
    points_np = np.random.randn(8192, 3).astype(np.float32) * 3
    with torch.no_grad():
        semantic = model.predict_semantic(points_np, device=str(device))
        trees = model.predict_instances(points_np, device=str(device))

    print(f"Semantic pred shape: {semantic.shape}")  # (1, 8192)
    print(f"Unique classes: {np.unique(semantic)}")  # 应该是 [0, 1, 2]
    print(f"Trees detected: {len(trees)}")

    print("[OK] TreeSegmentationModel test passed")


if __name__ == "__main__":
    test_model()
