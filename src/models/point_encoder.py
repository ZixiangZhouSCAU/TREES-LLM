"""
PointLLM-style Point Encoder for Tree Point Clouds
基于PointNet++的3D点云编码器，直接处理原始点云
PointLLM (ECCV 2024, ByteDance/OpenGVLab)风格：点云 → 语义特征序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction: FPS采样 + Ball Query分组 + PointNet"""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list, group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz: (B, N, 3)  坐标
        points: (B, N, C) 可选特征，为None时只用坐标
        返回: (new_xyz, new_points)  new_xyz: (B, S, 3), new_points: (B, ?, S)
        """
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)          # (B, 1, 3)
            if points is not None:
                feat = points.mean(dim=1, keepdim=True)     # (B, 1, C)
            else:
                feat = new_xyz                               # (B, 1, 3)
            new_points = torch.cat([new_xyz, feat], dim=-1)  # (B, 1, 3+C)
            new_points = new_points.permute(0, 2, 1).unsqueeze(-1)  # (B, 3+C, 1, 1)
        else:
            new_xyz = self._sample_point(xyz, self.npoint)  # (B, S, 3)
            grouped_xyz, idx = self._group_points(xyz, new_xyz, self.radius, self.nsample)
            # grouped_xyz: (B, C=3, S, K)

            if points is not None:
                # Index the per-point features (B, N, C_feat) → (B, S, K, C_feat)
                interp_points = self._index_points(points, idx)   # (B, S, K, C_feat)
                interp_points = interp_points.permute(0, 3, 1, 2)  # (B, C_feat, S, K)
                # Concatenate xyz features + point features along channel dim
                new_points = torch.cat([grouped_xyz, interp_points], dim=1)  # (B, 3+C_feat, S, K)
            else:
                new_points = grouped_xyz  # (B, 3, S, K)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]  # (B, last_channel, S)
        new_points = new_points.permute(0, 2, 1)        # (B, S, last_channel)
        return new_xyz, new_points

    def _sample_point(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        device = xyz.device
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        dists = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), device=device)
        batch_indices = torch.arange(B, device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid_xyz) ** 2, -1)
            mask = dist < dists
            dists[mask] = dist[mask]
            farthest = torch.max(dists, dim=1)[1]

        return xyz[batch_indices, centroids, :]

    def _group_points(self, xyz: torch.Tensor, new_xyz: torch.Tensor,
                      radius: float, nsample: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B = xyz.shape[0]
        new_xyz_expanded = new_xyz.unsqueeze(2)
        xyz_expanded = xyz.unsqueeze(1)
        dists = torch.sum((xyz_expanded - new_xyz_expanded) ** 2, -1)
        idx = torch.argsort(dists, dim=-1)[:, :, :nsample]
        grouped_xyz = self._index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
        return grouped_xyz_norm.permute(0, 3, 1, 2), idx

    @staticmethod
    def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather points by index. points: (B, N, C), idx: (B, S, K) → (B, S, K, C)"""
        B, N, C = points.shape
        _, S, K = idx.shape
        # Flatten index: offset each batch
        idx_flat = idx + torch.arange(B, device=points.device).view(B, 1, 1) * N  # (B, S, K)
        idx_flat = idx_flat.reshape(-1)  # (B*S*K,)
        points_flat = points.reshape(B * N, C)  # (B*N, C)
        gathered = points_flat[idx_flat]  # (B*S*K, C)
        return gathered.view(B, S, K, C)


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation: 从稀疏点插值到密集点"""

    def __init__(self, in_channel: int, mlp: list):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: torch.Tensor, points2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Interpolate features from sparse xyz1 to dense xyz2.
        xyz1: (B, S, 3) source coords, xyz2: (B, N, 3) target coords
        points1: (B, S, C1) source features, returns (B, N, C_out)
        """
        B, N, _ = xyz2.shape
        S = xyz1.shape[1]
        k = min(3, S)

        if S == 1:
            # Single source point: replicate
            interpolated = points1.expand(B, N, -1)
        else:
            dists = self._pairwise_distance(xyz2, xyz1)  # (B, N, S)
            dists, idx = torch.topk(dists, k=k, dim=-1)
            dists[dists < 1e-10] = 1e-10
            weight = (1.0 / dists)
            weight = weight / weight.sum(dim=-1, keepdim=True)

            idx_flat = idx.reshape(B, N * k)
            pts_flat = points1.reshape(B * S, -1)
            gathered = pts_flat[idx_flat].reshape(B, N, k, -1)
            interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=2)

        if points2 is not None:
            p2 = points2.expand(B, N, -1) if points2.shape[1] != N else points2
            new_points = torch.cat([interpolated, p2], dim=-1)
        else:
            new_points = interpolated

        # MLP: new_points is (B, N, C_in), need (B, C_in, N)
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)

    @staticmethod
    def _pairwise_distance(xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz1.shape
        dist = -2 * torch.matmul(xyz1, xyz2.transpose(1, 2))
        dist += torch.sum(xyz1 ** 2, -1).view(B, N, 1)
        dist += torch.sum(xyz2 ** 2, -1).view(B, 1, xyz2.shape[1])
        return torch.abs(dist)

    @staticmethod
    def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        repeat_shape[1] = 1
        indices = idx.view(view_shape).expand(repeat_shape + [points.shape[-1]])
        return torch.gather(points, dim=1, index=indices)


class TreePointEncoder(nn.Module):
    """
    树木点云专用PointNet++编码器 (PointLLM风格)
    层次化提取：局部几何 → 中层结构 → 全局面貌

    输出: 全局特征 + 多尺度特征序列，可直接送入tokenizer或LLM
    """

    def __init__(self, input_dim: int = 3, global_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.global_dim = global_dim

        # SA层1: 局部几何 (N -> 1024)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_dim, [64, 64, 128])
        # SA层2: 中层结构 (1024 -> 256)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 64, 128 + 3, [128, 128, 256])
        # SA层3: 全局面貌 (256 -> 64)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 64, 256 + 3, [256, 256, global_dim])
        # 全局聚合
        self.sa4 = PointNetSetAbstraction(None, None, None, global_dim + 3, [512, 1024], group_all=True)

        # 跳跃连接特征传播
        self.fp3 = PointNetFeaturePropagation(global_dim, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256, [128, 128])

    def forward(self, points: torch.Tensor) -> dict:
        """
        points: (B, N, 3) 或 (B, N, 6) 带RGB
        返回: 包含全局特征和多尺度特征的字典
        """
        normalized, mean, scale = self._normalize_points(points)

        l1_xyz, l1_points = self.sa1(normalized, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        _, l4_points = self.sa4(l3_xyz, l3_points)

        global_feat = l4_points.squeeze(1)

        # Skip FP layers, use SA multi-scale features directly
        up_features = self._upsample_to_original(l3_xyz, l3_points, normalized)

        return {
            "global_feature": global_feat,
            "multi_scale": [l1_points, l2_points, l3_points, l4_points],
            "up_features": up_features,
            "normalized_coords": normalized,
            "mean": mean,
            "scale": scale,
            "l3_features": l3_points,
        }

    def _normalize_points(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centroid = torch.mean(points, dim=1, keepdim=True)
        normalized = points - centroid
        scale = torch.max(torch.norm(normalized, dim=-1, keepdim=True), dim=1, keepdim=True)[0]
        scale = torch.clamp(scale, min=1e-8)
        normalized = normalized / scale
        return normalized, centroid.squeeze(1), scale.squeeze(1)

    def _upsample_to_original(self, sparse_xyz: torch.Tensor, sparse_points: torch.Tensor,
                               target_xyz: torch.Tensor) -> torch.Tensor:
        B, N, _ = target_xyz.shape
        S = sparse_xyz.shape[1]
        C = sparse_points.shape[2]
        k = min(3, S)
        dists = self._pairwise_distance(target_xyz, sparse_xyz)
        dists, idx = torch.topk(dists, k=k, dim=-1)
        dists = dists.clone()
        dists[dists < 1e-10] = 1e-10
        weight = (1.0 / dists)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        gathered = torch.gather(
            sparse_points.unsqueeze(1).expand(B, N, S, C),
            dim=2,
            index=idx.unsqueeze(-1).expand(B, N, k, C)
        )
        interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=2)
        proj = nn.Linear(C, 512, device=interpolated.device)
        return proj(interpolated)

    @staticmethod
    def _pairwise_distance(xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz1.shape
        dist = -2 * torch.matmul(xyz1, xyz2.transpose(1, 2))
        dist += torch.sum(xyz1 ** 2, -1).view(B, N, 1)
        dist += torch.sum(xyz2 ** 2, -1).view(B, 1, xyz2.shape[1])
        return torch.abs(dist)


def test_encoder():
    """测试编码器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TreePointEncoder(input_dim=3).to(device)
    points = torch.randn(2, 4096, 3).to(device)

    with torch.no_grad():
        out = encoder(points)

    print("=== TreePointEncoder 测试 ===")
    print(f"输入: {points.shape}")
    print(f"全局特征: {out['global_feature'].shape}")
    print(f"上采样特征: {out['up_features'].shape}")
    print(f"多尺度: {[p.shape for p in out['multi_scale']]}")
    print("✓ 编码器测试通过")


if __name__ == "__main__":
    test_encoder()