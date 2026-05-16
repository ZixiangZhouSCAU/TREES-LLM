"""
PointLLM for Trees - 主模型封装
端到端Pipeline: 点云 → Encoder → Tokenizer → Projector → GLM

PointLLM (ECCV 2024, ByteDance/OpenGVLab)风格：
- 直接处理原始3D点云，不渲染为图像
- VQ-VAE离散化为token序列
- 通过投影层与LLM语义空间对齐
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from .point_encoder import TreePointEncoder
from .tokenizer import TreePointTokenizer
from .llm_projector import LLMProjector, TreeDescriptionGenerator


class PointLLMConfig:
    """PointLLM for Trees配置"""
    num_points: int = 4096
    encoder_dim: int = 512
    token_dim: int = 128
    num_tokens: int = 2048
    llm_dim: int = 4096
    num_queries: int = 256
    commitment_cost: float = 0.25
    input_dim: int = 3
    # Path to ULIP-2 PointBERT pretrained checkpoint (relative to project root)
    pretrained_encoder_path: str = "pretrained/ULIP-2/pretrained_models/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt"


class PointLLMForTrees:
    """
    PointLLM主模型（无训练版）
    提供点云编码 + token化 + 描述生成
    不做LLM微调，直接调用GLM API

    支持两种encoder模式：
    - pretrained: 使用ULIP-2 PointBERT（推荐，从Objaverse预训练）
    - random: 使用随机初始化的TreePointEncoder
    """

    def __init__(self, config: Optional[PointLLMConfig] = None, device: str = "auto",
                 use_pretrained: bool = True):
        if config is None:
            config = PointLLMConfig()
        self.config = config

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 选择编码器：优先使用预训练的ULIP-2 PointBERT
        if use_pretrained:
            try:
                import os as _os
                from pathlib import Path as _P
                # Resolve checkpoint path relative to project root (3 levels up from this file)
                project_root = _P(__file__).parent.parent.parent
                ckpt_path = project_root / config.pretrained_encoder_path
                if ckpt_path.exists():
                    from .pretrained_encoder import PretrainedPointEncoder
                    self.encoder = PretrainedPointEncoder(
                        checkpoint_path=str(ckpt_path),
                        load_pretrained=True,
                    ).to(self.device)
                    self.encoder.eval()
                    self._encoder_type = "ulip2_pointbert"
                    print(f"[PointLLMForTrees] Using pretrained ULIP-2 PointBERT encoder")
                else:
                    print(f"[PointLLMForTrees] Pretrained checkpoint not found at {ckpt_path}, falling back to random encoder")
                    self._encoder_type = "random"
                    self.encoder = TreePointEncoder(
                        input_dim=config.input_dim,
                        global_dim=config.encoder_dim,
                    ).to(self.device)
                    self.encoder.eval()
            except Exception as e:
                print(f"[PointLLMForTrees] Failed to load pretrained encoder: {e}, using random encoder")
                self._encoder_type = "random"
                self.encoder = TreePointEncoder(
                    input_dim=config.input_dim,
                    global_dim=config.encoder_dim,
                ).to(self.device)
                self.encoder.eval()
        else:
            self._encoder_type = "random"
            self.encoder = TreePointEncoder(
                input_dim=config.input_dim,
                global_dim=config.encoder_dim,
            ).to(self.device)
            self.encoder.eval()

        self.tokenizer = TreePointTokenizer(
            input_dim=config.encoder_dim,
            token_dim=config.token_dim,
            num_tokens=config.num_tokens,
            commitment_cost=config.commitment_cost,
        ).to(self.device)

        self.projector = LLMProjector(
            token_dim=config.token_dim,
            llm_dim=config.llm_dim,
            num_queries=config.num_queries,
        ).to(self.device)

        self.desc_gen = TreeDescriptionGenerator()
        self.tokenizer.eval()
        self.projector.eval()

    def encode(self, points: np.ndarray) -> Dict:
        """
        对点云进行完整编码
        points: (N, 3) numpy array

        返回: 分析结果字典（含token序列 + 几何参数）
        """
        # 归一化到单位球，采样到固定点数
        pts_tensor = self._preprocess(points)   # (1, N, 3)

        # Encoder
        enc_out = self.encoder(pts_tensor)
        global_feat = enc_out["global_feature"]   # (1, encoder_dim)

        if self._encoder_type == "ulip2_pointbert":
            # 预训练encoder: 768-dim global feature + trans_feature
            analysis = self._encode_pretrained(pts_tensor, enc_out)
        else:
            # 随机encoder: 使用up_features进行VQ
            analysis = self._encode_random(pts_tensor, enc_out)

        return analysis

    def _encode_pretrained(self, pts_tensor: torch.Tensor, enc_out: Dict) -> Dict:
        """使用预训练ULIP-2 PointBERT编码器"""
        global_feat = enc_out["global_feature"]    # (1, 768)
        trans_feat = enc_out["trans_feature"]      # (1, 513, 384)

        z = pts_tensor[0, :, 2].cpu().numpy()

        # 预训练特征统计
        cls_token = trans_feat[0, 0]               # (384,)
        group_tokens = trans_feat[0, 1:]           # (512, 384)
        feat_norm = global_feat[0].norm().item()
        cls_norm = cls_token.norm().item()
        group_std = group_tokens.std().item()
        group_mean = group_tokens.mean().item()

        # VQ量化: 将768维global_feat投影到tokenizer的input_dim (512)，再展开到per-point
        proj_768_to_512 = nn.Linear(768, self.config.encoder_dim, device=self.device)
        projected_global = proj_768_to_512(global_feat)  # (1, 512)
        # 展开到per-point特征用于VQ
        feat_for_vq = projected_global.unsqueeze(1).expand(-1, pts_tensor.shape[1], -1)  # (1, N, 512)
        quantized, info = self.tokenizer(feat_for_vq)

        analysis = {
            "geometry": {
                "n_points": int(pts_tensor.shape[1]),
                "z_min": float(z.min()),
                "z_max": float(z.max()),
                "height": float(z.max() - z.min()),
                "centroid": pts_tensor[0].mean(axis=0).cpu().numpy().tolist(),
            },
            "tokens": {
                "unique_tokens": int(info["perplexity"].cpu().item()),
                "total_tokens": int(pts_tensor.shape[1]),
                "token_indices": info["token_indices"].cpu().numpy().tolist(),
                "vq_loss": float(info["vq_loss"].cpu().item()),
            },
            "pretrained_features": {
                "global_norm": feat_norm,
                "cls_token_norm": cls_norm,
                "group_std": group_std,
                "group_mean": group_mean,
                "encoder_type": "ULIP-2 PointBERT (pretrained on Objaverse)",
            },
        }
        return analysis

    def _encode_random(self, pts_tensor: torch.Tensor, enc_out: Dict) -> Dict:
        """使用随机初始化的TreePointEncoder"""
        up_feats = enc_out["up_features"]          # (1, N, encoder_dim)
        quantized, info = self.tokenizer(up_feats)  # (1, N, token_dim)

        z = pts_tensor[0, :, 2].cpu().numpy()
        analysis = {
            "geometry": {
                "n_points": int(pts_tensor.shape[1]),
                "z_min": float(z.min()),
                "z_max": float(z.max()),
                "height": float(z.max() - z.min()),
                "centroid": pts_tensor[0].mean(axis=0).cpu().numpy().tolist(),
            },
            "tokens": {
                "unique_tokens": int(info["perplexity"].cpu().item()),
                "total_tokens": int(pts_tensor.shape[1]),
                "token_indices": info["token_indices"].cpu().numpy().tolist(),
                "vq_loss": float(info["vq_loss"].cpu().item()),
            },
        }
        return analysis

    def build_prompt(self, analysis: dict, question: str) -> str:
        """构建发送给GLM的prompt"""
        description = self.desc_gen.from_analysis(analysis)

        prompt = f"""你是一位资深林业专家，擅长分析LiDAR点云数据。
你收到了PointLLM树木点云分析结果，包含：
- 几何结构：高度范围、质心位置、点数
- Token序列：点云特征的离散化编码（含VQ困惑度）

{description}

用户问题: {question}

请基于以上3D几何分析回答，适当引用体素/分层/密度数据支撑你的分析。用专业但易懂的语言。"""

        return prompt

    def _preprocess(self, points: np.ndarray) -> torch.Tensor:
        """归一化 + 采样"""
        pts = torch.from_numpy(points.astype(np.float32))

        # Centroid normalize
        centroid = pts.mean(dim=0, keepdim=True)
        pts = pts - centroid

        # Scale to unit sphere
        scale = pts.norm(dim=-1).max()
        scale = max(scale.item(), 1e-8)
        pts = pts / scale

        # Sample to fixed number
        N = pts.shape[0]
        if N > self.config.num_points:
            idx = np.random.choice(N, self.config.num_points, replace=False)
            pts = pts[idx]
        elif N < self.config.num_points:
            pad = torch.zeros(self.config.num_points - N, 3)
            pts = torch.cat([pts, pad], dim=0)

        return pts.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def analyze(self, points: np.ndarray) -> Dict:
        """完整分析：编码 + 参数提取 + 描述"""
        analysis = self.encode(points)

        # 添加高度分层统计
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        height = z_max - z_min

        trunk_thresh = z_min + height * 0.3
        crown_thresh = z_max - height * 0.2

        trunk_pts = (z < trunk_thresh).sum()
        mid_pts = ((z >= trunk_thresh) & (z < crown_thresh)).sum()
        crown_pts = (z >= crown_thresh).sum()
        total = trunk_pts + mid_pts + crown_pts

        analysis["layers"] = {
            "trunk(0-30%)": trunk_pts / total * 100 if total > 0 else 0,
            "transition(30-80%)": mid_pts / total * 100 if total > 0 else 0,
            "crown(80-100%)": crown_pts / total * 100 if total > 0 else 0,
        }
        return analysis

    def to_description(self, analysis: dict) -> str:
        return self.desc_gen.from_analysis(analysis)


def test_pointllm():
    """测试完整pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointLLMForTrees(device=str(device))
    points = np.random.randn(4096, 3).astype(np.float32) * 5

    analysis = model.analyze(points)
    print("=== PointLLMForTrees 测试 ===")
    print(f"几何: {analysis['geometry']}")
    print(f"分层: {analysis['layers']}")
    print(f"Token: {analysis['tokens']}")

    prompt = model.build_prompt(analysis, "这棵树有多高？")
    print(f"\nPrompt预览: {prompt[:200]}...")
    print("✓ 完整pipeline测试通过")


if __name__ == "__main__":
    test_pointllm()
