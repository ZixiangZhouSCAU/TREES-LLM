"""
三层编码器 - TreeEncoder
参考 3DCITY-LLM 粗到细（Coarse-to-Fine）编码架构

三层：
  Object Encoding（单树级别）：每棵树 → 512维特征
  Relationship Encoding（邻树关系）：目标树 + K近邻 → 256维特征
  Scene Encoding（林分级）：整块样地 → 512维特征

每个编码分支有独立的 FeatureProjector（可训练，约 2M 参数）
训练时只更新 projector，GLM-4-Flash 冻结
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class FeatureProjector(nn.Module):
    """
    可训练的 FeatureProjector
    将多模态特征投影到 LLM embedding 空间（512维）

    结构：Input(864/960/48) → Linear(512) → ReLU → Linear(512) → Output(512)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim) 或 (B, N, input_dim)
        returns: (B, output_dim) 或 (B, N, output_dim)
        """
        return self.net(x)


class ObjectProjector(nn.Module):
    """
    Object Encoding: 单树特征投影
    输入维度: 864 = 768(PointBERT) + 64(几何) + 32(物种)
    """

    def __init__(self):
        super().__init__()
        self.proj_shape = nn.Linear(768, 256)
        self.proj_geom = nn.Linear(64, 128)
        self.proj_species = nn.Linear(32, 128)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(
        self,
        pointbert_feature: torch.Tensor,   # (B, 768)
        geom_feature: torch.Tensor,         # (B, 64)
        species_feature: torch.Tensor,      # (B, 32)
    ) -> torch.Tensor:
        """
        returns: (B, 512) object embedding
        """
        f_shape = self.proj_shape(pointbert_feature)
        f_geom = self.proj_geom(geom_feature)
        f_species = self.proj_species(species_feature)

        fused = torch.cat([f_shape, f_geom, f_species], dim=-1)   # (B, 416)
        return self.fusion(fused)                                  # (B, 512)


class RelationshipProjector(nn.Module):
    """
    Relationship Encoding: 邻树关系特征投影
    输入维度: 48 = 16(距离) + 16(高差) + 16(竞争指数)
    """

    def __init__(self):
        super().__init__()
        self.proj_dist = nn.Linear(16, 64)
        self.proj_height = nn.Linear(16, 64)
        self.proj_competition = nn.Linear(16, 64)
        self.fusion = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, rel_features: torch.Tensor) -> torch.Tensor:
        """
        rel_features: (B, K, 48) K=5 近邻的关系特征
        returns: (B, 256) relationship embedding
        """
        f_dist = self.proj_dist(rel_features[:, :, :16])       # (B, K, 64)
        f_height = self.proj_height(rel_features[:, :, 16:32])  # (B, K, 64)
        f_comp = self.proj_competition(rel_features[:, :, 32:]) # (B, K, 64)

        fused = torch.cat([f_dist, f_height, f_comp], dim=-1)  # (B, K, 192)
        pooled = fused.mean(dim=1)                             # (B, 192)
        return self.fusion(pooled)                              # (B, 256)


class SceneProjector(nn.Module):
    """
    Scene Encoding: 林分级特征投影
    输入维度: 960 = 768(PointBERT) + 64(密度) + 64(高度分布) + 64(郁闭度)
    """

    def __init__(self):
        super().__init__()
        self.proj_global = nn.Linear(768, 256)
        self.proj_density = nn.Linear(64, 128)
        self.proj_height = nn.Linear(64, 128)
        self.proj_canopy = nn.Linear(64, 128)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(
        self,
        global_feature: torch.Tensor,   # (B, 768)
        density_feature: torch.Tensor,  # (B, 64)
        height_feature: torch.Tensor,   # (B, 64)
        canopy_feature: torch.Tensor,   # (B, 64)
    ) -> torch.Tensor:
        """
        returns: (B, 512) scene embedding
        """
        f_global = self.proj_global(global_feature)     # (B, 256)
        f_density = self.proj_density(density_feature)  # (B, 128)
        f_height = self.proj_height(height_feature)     # (B, 128)
        f_canopy = self.proj_canopy(canopy_feature)    # (B, 128)

        fused = torch.cat([f_global, f_density, f_height, f_canopy], dim=-1)  # (B, 640)
        return self.fusion(fused)                                             # (B, 512)


class TreeEncoder(nn.Module):
    """
    三层编码器主类

    功能：
    - encode_object(tree_points): 单树 Object 层编码 → 512维
    - encode_relationship(target_features, neighbor_features): 邻树 Relationship 层 → 256维
    - encode_scene(all_tree_points): 整块样地 Scene 层编码 → 512维
    - encode_all(trees_points_list): 一次性编码所有树 + 场景

    用法：
        encoder = TreeEncoder(device="cuda")
        E_obj = encoder.encode_object(tree_points, dbh=35.2, height=12.3, species="玉兰")
        E_rel = encoder.encode_relationship(target_idx=0, all_tree_features=[...])
        E_scene = encoder.encode_scene(all_tree_points, n_trees=17, area_m2=200)
    """

    def __init__(
        self,
        device: str = "auto",
        use_pretrained: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 加载 PointBERT 编码器
        if use_pretrained:
            try:
                from pathlib import Path
                if pretrained_path is None:
                    project_root = Path(__file__).parent.parent.parent
                    pretrained_path = str(
                        project_root / "pretrained/ULIP-2/pretrained_models"
                        "/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt"
                    )
                from .pretrained_encoder import PretrainedPointEncoder
                self.pointbert = PretrainedPointEncoder(
                    checkpoint_path=pretrained_path,
                    load_pretrained=True,
                ).to(self.device)
                self.pointbert.eval()
                print("[TreeEncoder] Using pretrained ULIP-2 PointBERT")
            except Exception as e:
                print(f"[TreeEncoder] Failed to load pretrained encoder: {e}, using random")
                self.pointbert = self._build_random_encoder()
        else:
            self.pointbert = self._build_random_encoder()

        # 三个可训练的 projector
        self.object_proj = ObjectProjector().to(self.device)
        self.relationship_proj = RelationshipProjector().to(self.device)
        self.scene_proj = SceneProjector().to(self.device)

        self.K = 5  # 近邻数量
        self._encoder_loaded = True
        print(f"[TreeEncoder] Initialized on {self.device}")

    def _build_random_encoder(self):
        """备用：随机初始化的 PointNet++ 编码器"""
        from .point_encoder import TreePointEncoder
        return TreePointEncoder(input_dim=3, global_dim=768).to(self.device)

    @torch.no_grad()
    def _encode_pointbert(self, points: np.ndarray) -> torch.Tensor:
        """用 PointBERT 编码点云，返回 768 维全局特征"""
        pts = torch.from_numpy(points.astype(np.float32)).to(self.device)

        # 预处理：质心归一化 + 单位球归一化
        centroid = pts.mean(dim=0, keepdim=True)
        pts_norm = pts - centroid
        scale = pts_norm.norm(dim=-1).max().clamp(min=1e-8)
        pts_norm = pts_norm / scale
        pts_norm = pts_norm.unsqueeze(0)  # (1, N, 3)

        # 编码
        out = self.pointbert(pts_norm)
        return out["global_feature"]  # (1, 768)

    def encode_object(
        self,
        tree_points: np.ndarray,
        dbh: float = 0.0,
        height: float = 0.0,
        crown_width: float = 0.0,
        species_embedding: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Object Encoding（单树级别）

        Args:
            tree_points: (N, 3) 单棵树点云
            dbh: 胸径(cm)
            height: 树高(m)
            crown_width: 冠幅(m)
            species_embedding: (32,) 树种特征向量（可选，默认零向量）

        Returns:
            torch.Tensor: (1, 512) object embedding
        """
        # 1. PointBERT 形状特征 (768,)
        shape_feat = self._encode_pointbert(tree_points)          # (1, 768)

        # 2. 几何参数特征 (64,)
        # 编码: [dbh/100, height/30, crown_width/10, 体积/1000, 碳储量/1000,
        #        x_range/5, y_range/5, z_std/5, 树点数/10000, ...]
        h = float(height) if height > 0 else 1.0
        geom_feat = torch.tensor([
            dbh / 100,            # 归一化胸径 (0-1)
            height / 30,          # 归一化树高 (0-1)
            crown_width / 10,     # 归一化冠幅 (0-1)
            (dbh / 100) ** 2 * height * 0.5 * 700 / 1000,  # 碳储量归一化
            tree_points.shape[0] / 10000,  # 点数归一化
            # 派生特征
            height / max(dbh / 100, 0.01),  # 高径比
            crown_width / max(height, 0.1),  # 冠高比
            np.std(tree_points[:, 0]) / 5,   # X扩散
            np.std(tree_points[:, 1]) / 5,   # Y扩散
            np.std(tree_points[:, 2]) / height if height > 0 else 0,  # Z标准差
        ] + [0.0] * 54, dtype=torch.float32, device=self.device)  # 补到64维

        # 3. 树种特征 (32,) — 默认零向量
        if species_embedding is None:
            species_feat = torch.zeros(32, device=self.device)
        else:
            species_feat = torch.from_numpy(species_embedding.astype(np.float32)).to(self.device)

        # 4. 投影
        E_obj = self.object_proj(
            shape_feat,      # (1, 768)
            geom_feat.unsqueeze(0),   # (1, 64)
            species_feat.unsqueeze(0), # (1, 32)
        )  # (1, 512)

        return E_obj

    def encode_relationship(
        self,
        target_features: torch.Tensor,     # (1, 512) 目标树 object embedding
        neighbor_features: List[torch.Tensor],  # list of (1, 512) K个近邻的 object embedding
        target_params: Dict,               # 目标树几何参数
        neighbor_params_list: List[Dict],  # 近邻树几何参数
    ) -> torch.Tensor:
        """
        Relationship Encoding（邻树关系）

        构建 K 个近邻的关系特征：
          - 距离特征：欧氏距离（归一化）
          - 高差特征：树高差（归一化）
          - 竞争指数：冠幅重叠率近似

        Args:
            target_features: (1, 512) 目标树 embedding
            neighbor_features: K 个近邻的 (1, 512) embeddings
            target_params: 目标树几何参数字典
            neighbor_params_list: K 个近邻参数字典列表

        Returns:
            torch.Tensor: (1, 256) relationship embedding
        """
        K = min(len(neighbor_features), self.K)

        if K == 0:
            # 无近邻，返回零向量
            return torch.zeros(1, 256, device=self.device)

        rel_features = []
        for i in range(K):
            nb_feat = neighbor_features[i]  # (1, 512)
            nb_params = neighbor_params_list[i]

            # 计算距离（用质心XY坐标的欧氏距离）
            tgt_xy = target_params.get("center_xy", [0, 0])
            nb_xy = nb_params.get("center_xy", [0, 0])
            dist = np.sqrt((tgt_xy[0] - nb_xy[0]) ** 2 + (tgt_xy[1] - nb_xy[1]) ** 2)
            dist_norm = min(dist / 10, 1.0)  # 归一化到 [0, 1]

            # 计算高差
            tgt_h = target_params.get("height", 0)
            nb_h = nb_params.get("height", 0)
            h_diff = abs(tgt_h - nb_h)
            h_diff_norm = min(h_diff / 20, 1.0)  # 归一化

            # 计算竞争指数（冠幅重叠近似）
            tgt_cw = target_params.get("crown_width", 0)
            nb_cw = nb_params.get("crown_width", 0)
            # 用冠幅比来估算重叠：min(cw1, cw2) / max(cw1, cw2)
            if tgt_cw > 0 and nb_cw > 0:
                overlap = min(tgt_cw, nb_cw) / max(tgt_cw, nb_cw)
            else:
                overlap = 0.0

            # 距离大的竞争小，距离小的竞争大
            competition = (1 - dist_norm) * overlap

            rel_feat = [
                dist_norm,        # 距离特征
                h_diff_norm,      # 高差特征
                competition,      # 竞争指数
            ]

            # 扩展到 16 维（填充零）
            rel_feat = rel_feat + [0.0] * 13
            rel_features.append(rel_feat)

        # 填充到 K=5
        while len(rel_features) < self.K:
            rel_features.append([0.0] * 16)

        rel_tensor = torch.tensor(rel_features[:self.K], dtype=torch.float32, device=self.device)  # (K, 16)
        rel_features_padded = torch.zeros(self.K, 48, device=self.device)
        rel_features_padded[:, :16] = rel_tensor

        # 投影
        E_rel = self.relationship_proj(rel_features_padded.unsqueeze(0))  # (1, 256)
        return E_rel

    @torch.no_grad()
    def encode_scene(
        self,
        all_trees_points: List[np.ndarray],
        n_trees: int,
        area_m2: float = 100.0,
        tree_params_list: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """
        Scene Encoding（林分级）

        Args:
            all_trees_points: 所有树的点云列表
            n_trees: 树木数量
            area_m2: 样地面积（平方米）
            tree_params_list: 每棵树的参数字典列表

        Returns:
            torch.Tensor: (1, 512) scene embedding
        """
        if not all_trees_points:
            return torch.zeros(1, 512, device=self.device)

        # 1. 全局 PointBERT 特征：将所有点合并编码
        all_points = np.vstack(all_trees_points) if len(all_trees_points) > 1 else all_trees_points[0]
        # 限制点数不超过 8192（PointBERT 最大输入）
        if len(all_points) > 8192:
            idx = np.random.choice(len(all_points), 8192, replace=False)
            all_points = all_points[idx]
        global_feat = self._encode_pointbert(all_points)  # (1, 768)

        # 2. 密度特征 (64,)
        density = n_trees / max(area_m2, 1.0) * 10000  # 棵/公顷
        density_norm = min(density / 2000, 1.0)

        # 3. 高度分布特征 (64,)
        if tree_params_list:
            heights = [p.get("height", 0) for p in tree_params_list]
            dbhs = [p.get("dbh", 0) for p in tree_params_list]
            heights = [h for h in heights if h > 0]
            h_mean = np.mean(heights) if heights else 0
            h_std = np.std(heights) if heights else 0
            h_max = max(heights) if heights else 0
            d_mean = np.mean(dbhs) if dbhs else 0

            height_feat = [
                h_mean / 30,
                h_std / 30,
                h_max / 30,
                d_mean / 100,
                density_norm,
            ] + [0.0] * 59
        else:
            height_feat = [0.0] * 64

        # 4. 郁闭度特征 (64,) — 用地面点比例近似
        ground_ratio = 0.15  # 假设 15% 是地面点
        canopy_ratio = 1 - ground_ratio
        canopy_feat = [
            canopy_ratio,
            density_norm,
            h_mean / 30 if heights else 0,
        ] + [0.0] * 61

        global_feat_t = global_feat  # (1, 768)
        density_t = torch.tensor([height_feat[:64]], dtype=torch.float32, device=self.device)
        height_t = torch.tensor([height_feat], dtype=torch.float32, device=self.device)
        canopy_t = torch.tensor([canopy_feat], dtype=torch.float32, device=self.device)

        E_scene = self.scene_proj(global_feat_t, density_t, height_t, canopy_t)  # (1, 512)
        return E_scene

    def encode_all(
        self,
        trees_points: List[np.ndarray],
        trees_params: List[Dict],
        area_m2: float = 100.0,
    ) -> Dict[str, torch.Tensor]:
        """
        一次性编码所有树 + 场景

        Args:
            trees_points: 所有树的点云列表
            trees_params: 所有树的参数字典列表（必须包含 center_xy, height, crown_width）
            area_m2: 样地面积

        Returns:
            Dict with:
              - object_features: List[Tensor] 每棵树的 512 维 embedding
              - relationship_features: List[Tensor] 每棵树的邻树关系 embedding
              - scene_features: Tensor (1, 512) 场景 embedding
              - tree_count: int
        """
        n = len(trees_points)

        # Object 层：编码所有树
        object_features = []
        for i, (pts, params) in enumerate(zip(trees_points, trees_params)):
            E = self.encode_object(
                pts,
                dbh=params.get("dbh", 0),
                height=params.get("height", 0),
                crown_width=params.get("crown_width", 0),
            )
            object_features.append(E)

        # Relationship 层：对每棵树计算邻树关系
        relationship_features = []
        for i in range(n):
            # 找 K=5 近邻（按 XY 距离）
            distances = []
            tgt_xy = trees_params[i].get("center_xy", [0, 0])
            for j in range(n):
                if i == j:
                    continue
                nb_xy = trees_params[j].get("center_xy", [0, 0])
                d = np.sqrt((tgt_xy[0] - nb_xy[0]) ** 2 + (tgt_xy[1] - nb_xy[1]) ** 2)
                distances.append((j, d))
            distances.sort(key=lambda x: x[1])
            neighbor_indices = [d[0] for d in distances[:self.K]]

            neighbor_features = [object_features[idx] for idx in neighbor_indices]
            neighbor_params = [trees_params[idx] for idx in neighbor_indices]

            E_rel = self.encode_relationship(
                target_features=object_features[i],
                neighbor_features=neighbor_features,
                target_params=trees_params[i],
                neighbor_params_list=neighbor_params,
            )
            relationship_features.append(E_rel)

        # Scene 层：整块样地
        E_scene = self.encode_scene(
            all_trees_points=trees_points,
            n_trees=n,
            area_m2=area_m2,
            tree_params_list=trees_params,
        )

        return {
            "object_features": object_features,      # list of (1, 512)
            "relationship_features": relationship_features,  # list of (1, 256)
            "scene_features": E_scene,              # (1, 512)
            "tree_count": n,
        }

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取所有可训练参数（用于优化器）"""
        return list(self.object_proj.parameters()) + \
               list(self.relationship_proj.parameters()) + \
               list(self.scene_proj.parameters())

    def num_params(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_tree_encoder():
    """测试三层编码器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing TreeEncoder on {device}")

    encoder = TreeEncoder(device=str(device), use_pretrained=False)

    # Test 1: Object 层
    tree_points = np.random.randn(2000, 3).astype(np.float32) * 3
    E_obj = encoder.encode_object(tree_points, dbh=35.2, height=12.3, crown_width=4.5)
    print(f"Object encoding: {E_obj.shape}")  # (1, 512)

    # Test 2: Relationship 层
    neighbor_pts = [np.random.randn(1500, 3).astype(np.float32) * 3 for _ in range(5)]
    neighbor_features = []
    neighbor_params = []
    for j, pts in enumerate(neighbor_pts):
        f = encoder.encode_object(pts, dbh=30 + j * 2, height=10 + j, crown_width=4)
        neighbor_features.append(f)
        neighbor_params.append({
            "center_xy": [j * 2.0, j * 1.5],
            "height": 10 + j,
            "crown_width": 4,
        })

    E_rel = encoder.encode_relationship(
        target_features=E_obj,
        neighbor_features=neighbor_features,
        target_params={"center_xy": [0.0, 0.0], "height": 12.3, "crown_width": 4.5},
        neighbor_params_list=neighbor_params,
    )
    print(f"Relationship encoding: {E_rel.shape}")  # (1, 256)

    # Test 3: Scene 层
    all_trees = [tree_points] + neighbor_pts
    E_scene = encoder.encode_scene(all_trees, n_trees=6, area_m2=200)
    print(f"Scene encoding: {E_scene.shape}")  # (1, 512)

    # Test 4: encode_all
    params_list = [
        {"center_xy": [0.0, 0.0], "height": 12.3, "crown_width": 4.5, "dbh": 35.2},
    ] + [
        {"center_xy": [j * 2.0, j * 1.5], "height": 10 + j, "crown_width": 4, "dbh": 30 + j * 2}
        for j in range(5)
    ]
    result = encoder.encode_all(all_trees, params_list, area_m2=200)
    print(f"encode_all: {result['tree_count']} trees, "
          f"object={[f.shape for f in result['object_features']]}, "
          f"rel={[f.shape for f in result['relationship_features']]}, "
          f"scene={result['scene_features'].shape}")

    print(f"Trainable params: {encoder.num_params():,}")
    print("[OK] TreeEncoder test passed")


if __name__ == "__main__":
    test_tree_encoder()