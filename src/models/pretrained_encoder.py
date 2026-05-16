"""
Pretrained ULIP-2 PointBERT Encoder
 wrapping the Salesforce ULIP-2 pretrained PointBERT encoder.
 PointBERT: ViT-style transformer on grouped point cloud patches.
 Output: 768-dim global feature (pretrained on Objaverse with point-text alignment).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Utility: Pure-Python FPS (no CUDA extensions required)
# ──────────────────────────────────────────────────────────────────────────────

def fps_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling on CPU/GPU — pure PyTorch implementation.
    xyz: (B, N, 3)
    npoint: number of centroids to sample
    returns: (B, npoint, 3) sampled coordinates
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dists = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[torch.arange(B, device=device), farthest, :]  # (B, 3)
        centroid_xyz = centroid_xyz.view(B, 1, 3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)  # (B, N)
        mask = dist < dists
        dists[mask] = dist[mask]
        farthest = torch.max(dists, dim=1)[1]

    # gather sampled point coordinates via flat indexing
    idx_base = torch.arange(B, device=device).view(B, 1) * N     # (B, 1)
    idx_flat = (centroids + idx_base).reshape(-1)                 # (B*npoint,)
    xyz_flat = xyz.reshape(-1, 3)                                 # (B*N, 3)
    return xyz_flat[idx_flat].reshape(B, npoint, 3)


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    KNN on pure PyTorch (CPU/GPU compatible).
    xyz: (B, N, 3)  — all points
    new_xyz: (B, S, 3)  — query points
    returns: (B, S, k) indices of k nearest neighbours
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    # (B, S, 1, 3) - (B, 1, N, 3) → (B, S, N)
    dists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    _, idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=False)
    return idx   # (B, S, k)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder: PointNet-style local feature extractor
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Local encoder: per-group Conv1D → 256-dim local features. input_dim=3 (xyz only)"""

    def __init__(self, encoder_channel: int = 256, input_dim: int = 3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """
        point_groups: (B, G, M, C)  where M=group_size, C=input_dim(3)
        returns: (B, G, encoder_channel) local features per group
        """
        bs, g, m, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, m, c)
        # (B*G, C, M)
        point_groups = point_groups.transpose(2, 1).contiguous()

        feature = self.first_conv(point_groups)                        # (B*G, 256, M)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]   # (B*G, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, m), feature], dim=1)  # (B*G, 512, M)
        feature = self.second_conv(feature)                             # (B*G, encoder_channel, M)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]   # (B*G, encoder_channel)

        return feature_global.reshape(bs, g, self.encoder_channel)     # (B, G, C)


# ──────────────────────────────────────────────────────────────────────────────
# Grouping: FPS + KNN ball query (pure PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class Group(nn.Module):
    """FPS + KNN grouping: (B,N,3) → (B,G,M,3) + (B,G,3) centroids"""

    def __init__(self, num_group: int = 512, group_size: int = 32):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz: (B, N, 3)
        returns:
          neighborhood: (B, G, M, 3) — normalized relative coords
          center: (B, G, 3) — FPS centroids
        """
        B, N, C = xyz.shape

        # FPS centroids
        center = fps_pytorch(xyz, self.num_group)                   # (B, G, 3)

        # KNN (k=group_size) around centroids
        idx = knn_point(self.group_size, xyz, center)                # (B, G, M)

        idx_base = torch.arange(B, device=xyz.device).view(B, 1, 1) * N
        idx_flat = (idx + idx_base).reshape(-1)                     # (B*G*M,)
        neighborhood = xyz.reshape(B * N, C)[idx_flat]              # (B*G*M, 3)
        neighborhood = neighborhood.view(B, self.num_group, self.group_size, C)

        # Normalize: relative to centroid
        neighborhood = neighborhood - center.unsqueeze(2)            # (B, G, M, 3)

        return neighborhood, center


# ──────────────────────────────────────────────────────────────────────────────
# Transformer components (standard ViT blocks)
# ──────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, act_layer=nn.GELU, drop=drop)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Main PointBERT Encoder
# ──────────────────────────────────────────────────────────────────────────────

class PointTransformer(nn.Module):
    """
    ULIP-2 PointBERT encoder (xyz-only, 8k pretrained).

    Config (from PointTransformer_8192point.yaml):
      trans_dim=384, depth=12, num_heads=6, drop_path_rate=0.1,
      group_size=32, num_group=512, encoder_dims=256

    Output: 768-dim global feature = concat(cls_token, max_pool(group_tokens))
    """

    def __init__(
        self,
        trans_dim: int = 384,
        depth: int = 12,
        drop_path_rate: float = 0.1,
        num_heads: int = 6,
        group_size: int = 32,
        num_group: int = 512,
        encoder_dims: int = 256,
    ):
        super().__init__()

        self.group_size = group_size
        self.num_group = num_group

        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dims, input_dim=3)
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos   = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(trans_dim, num_heads, qkv_bias=True, drop=dpr[i], attn_drop=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(trans_dim)

        # Init cls token / cls pos
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def _load_from_ulip_checkpoint(self, state_dict: dict):
        """
        Load pretrained weights from ULIP-2 PointBERT-8k-xyz checkpoint.

        Key remapping needed:
          checkpoint: module.point_encoder.blocks.blocks.{i}.xxx
          our model:  blocks.{i}.xxx  (we store nn.ModuleList directly)

          checkpoint: module.point_encoder.encoder.first_conv.xxx
          our model:  encoder.first_conv.xxx  (self.encoder is Encoder, child of Group is also self.encoder)
        """
        new_sd = {}
        for k, v in state_dict.items():
            # Strip DDP prefix
            k = k.replace("module.", "")
            # Keep only point_encoder keys
            if not k.startswith("point_encoder."):
                continue
            # Strip point_encoder. prefix
            k = k[len("point_encoder."):]
            # Fix double "blocks.blocks." → "blocks."
            k = k.replace("blocks.blocks.", "blocks.")
            new_sd[k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"[ULIP-2 Encoder] missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"[ULIP-2 Encoder] unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        loaded = len(new_sd) - len(unexpected)
        print(f"[ULIP-2 Encoder] Loaded {loaded}/{len(new_sd)} pretrained parameters")

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, 3) — raw point cloud
        returns: (B, 768) global feature vector
        """
        # Group into (B, G, M, 3) patches
        neighborhood, center = self.group_divider(pts)           # (B,G,M,3), (B,G,3)

        # Local encoder → (B, G, 256)
        group_tokens = self.encoder(neighborhood)

        # Project 256 → 384
        group_tokens = self.reduce_dim(group_tokens)             # (B, G, 384)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)   # (B, 1, 384)
        cls_pos    = self.cls_pos.expand(group_tokens.size(0), -1, -1)    # (B, 1, 384)

        # Positional embedding from group centroids
        pos_emb = self.pos_embed(center)                        # (B, G, 384)

        # Input: [CLS] + group tokens, positions = [CLS_pos] + pos_emb
        x = torch.cat([cls_tokens, group_tokens], dim=1)        # (B, G+1, 384)
        pos = torch.cat([cls_pos, pos_emb], dim=1)             # (B, G+1, 384)

        # Transformer blocks
        for block in self.blocks:
            x = block(x + pos)

        x = self.norm(x)

        # Global feature: concat([CLS], max_pool(group_features)) → 768
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(dim=1)[0]], dim=-1)   # (B, 768)
        return concat_f


# ──────────────────────────────────────────────────────────────────────────────
# Public interface: PretrainedPointEncoder
# ──────────────────────────────────────────────────────────────────────────────

class PretrainedPointEncoder(nn.Module):
    """
    Wrapper for ULIP-2 PointBERT pretrained encoder.
    Drop-in replacement for the existing TreePointEncoder (point_encoder.py).

    Input:  (B, N, 3) raw point cloud
    Output: dict with:
      - global_feature: (B, 768) ULIP-2 pretrained global feature
      - raw_feature:    (B, 768) same as global_feature (alias for compatibility)
      - trans_feature:  (B, 384) transformer-level feature (pre-concat)
      - n_groups:       int — number of groups (512)
      - group_size:      int — points per group (32)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        trans_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        group_size: int = 32,
        num_group: int = 512,
        encoder_dims: int = 256,
        load_pretrained: bool = True,
    ):
        super().__init__()

        self.point_transformer = PointTransformer(
            trans_dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
            group_size=group_size,
            num_group=num_group,
            encoder_dims=encoder_dims,
        )

        if load_pretrained and checkpoint_path:
            self.load_pretrained(checkpoint_path)

    def load_pretrained(self, ckpt_path: str):
        """Load ULIP-2 PointBERT-8k-xyz pretrained weights."""
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"[PretrainedPointEncoder] checkpoint not found: {ckpt_path}")

        print(f"[PretrainedPointEncoder] Loading ULIP-2 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state_dict = ckpt.get("state_dict", ckpt)
        self.point_transformer._load_from_ulip_checkpoint(state_dict)

    def forward(self, pts: torch.Tensor) -> dict:
        """
        pts: (B, N, 3) torch float tensor
        returns: dict with pretrained features
        """
        # Ensure float32 and on same device as model
        pts = pts.to(dtype=torch.float32, device=next(self.parameters()).device)

        # Normalize to unit sphere (standard preprocessing)
        centroid = pts.mean(dim=1, keepdim=True)                   # (B, 1, 3)
        pts_norm = pts - centroid
        # scale: (B, N, 1) — max norm per sample
        scale = pts_norm.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0]  # (B, 1, 1)
        scale = scale.clamp(min=1e-8)
        pts_norm = pts_norm / scale                                # (B, N, 3)

        # Transformed features before concat
        trans_feat = self._forward_transformer(pts_norm)            # (B, 384)

        # Global feature (768-dim, used for downstream tasks)
        global_feat = torch.cat([
            trans_feat[:, 0],
            trans_feat[:, 1:].max(dim=1)[0]
        ], dim=-1)                                                  # (B, 768)

        return {
            "global_feature": global_feat,      # (B, 768) — main output
            "raw_feature": global_feat,          # alias
            "trans_feature": trans_feat,         # (B, 384) — pre-concat
            "n_groups": self.point_transformer.num_group,
            "group_size": self.point_transformer.group_size,
            "normalized_points": pts_norm,
            "centroid": centroid.squeeze(1),
            "scale": scale.squeeze(-1),
        }

    @torch.no_grad()
    def _forward_transformer(self, pts: torch.Tensor) -> torch.Tensor:
        """Forward through transformer only, returning (B, G+1, 384) before concat."""
        neighborhood, center = self.point_transformer.group_divider(pts)
        group_tokens = self.point_transformer.encoder(neighborhood)
        group_tokens = self.point_transformer.reduce_dim(group_tokens)

        cls_tokens = self.point_transformer.cls_token.expand(group_tokens.size(0), -1, -1)
        cls_pos    = self.point_transformer.cls_pos.expand(group_tokens.size(0), -1, -1)
        pos_emb    = self.point_transformer.pos_embed(center)

        x = torch.cat([cls_tokens, group_tokens], dim=1)
        pos = torch.cat([cls_pos, pos_emb], dim=1)

        for block in self.point_transformer.blocks:
            x = block(x + pos)

        return self.point_transformer.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────

def test_pretrained_encoder():
    """Test the pretrained encoder with dummy data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "H:/1reserch/03TREES-LLM/pretrained/ULIP-2/pretrained_models/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt"

    encoder = PretrainedPointEncoder(checkpoint_path=ckpt_path).to(device)
    encoder.eval()

    # Dummy input: 8192 points (matching pretraining)
    points = torch.randn(2, 8192, 3).to(device)

    with torch.no_grad():
        out = encoder(points)

    print("=== PretrainedPointEncoder (ULIP-2 PointBERT) Test ===")
    print(f"Input: {points.shape}")
    print(f"global_feature: {out['global_feature'].shape}, mean={out['global_feature'].mean():.4f}, std={out['global_feature'].std():.4f}")
    print(f"trans_feature:  {out['trans_feature'].shape}, mean={out['trans_feature'].mean():.4f}, std={out['trans_feature'].std():.4f}")
    print(f"n_groups: {out['n_groups']}, group_size: {out['group_size']}")
    print("[OK] Pretrained encoder loaded and working!")


if __name__ == "__main__":
    test_pretrained_encoder()