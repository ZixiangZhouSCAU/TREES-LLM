"""
PointLLM LLM Projector - 投影层：3D token特征 → LLM语义空间
将VQ量化后的点云token映射到GLM的embedding空间
"""

import torch
import torch.nn as nn
from typing import Optional


class LLMProjector(nn.Module):
    """
    Linear + LayerNorm投影层：将token_dim的点云特征映射到llm_dim
    PointLLM风格：轻量级映射，不做复杂变换
    """

    def __init__(self, token_dim: int = 128, llm_dim: int = 4096, num_queries: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries

        self.proj = nn.Sequential(
            nn.Linear(token_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, token_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads=8, batch_first=True)
        self.query_norm = nn.LayerNorm(token_dim)

    def forward(self, token_features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B = token_features.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        queries = self.query_norm(queries)

        key_padding_mask = (~mask.bool()) if mask is not None else None

        aggregated, _ = self.cross_attn(
            query=queries,
            key=token_features,
            value=token_features,
            key_padding_mask=key_padding_mask,
        )
        llm_embeds = self.proj(aggregated)
        return llm_embeds

    def project_direct(self, token_features: torch.Tensor):
        return self.proj(token_features)


class TreeDescriptionGenerator:
    """将点云分析结果转为LLM可读的文本描述"""

    @staticmethod
    def from_analysis(analysis: dict) -> str:
        parts = []
        if "geometry" in analysis:
            g = analysis["geometry"]
            parts.append("【几何结构】")
            parts.append(f"  总点数: {g.get('n_points', 'N/A')}")
            zmin = g.get('z_min', 0)
            zmax = g.get('z_max', 0)
            height = g.get('height', 0)
            parts.append(f"  高度范围: {zmin:.2f}m ~ {zmax:.2f}m (高度{height:.2f}m)")
            if "centroid" in g:
                parts.append(f"  质心位置: ({g['centroid'][0]:.2f}, {g['centroid'][1]:.2f})")
        if "tokens" in analysis:
            t = analysis["tokens"]
            parts.append("【Token分析】")
            parts.append(f"  唯一token数: {t.get('unique_tokens', 'N/A')}")
            parts.append(f"  总token数: {t.get('total_tokens', 'N/A')}")
            if "top_tokens" in t:
                parts.append(f"  高频token: {t['top_tokens']}")
        if "params" in analysis:
            p = analysis["params"]
            parts.append("【树木参数】")
            parts.append(f"  树高: {p.get('height', 0):.2f}m")
            parts.append(f"  胸径(DBH): {p.get('dbh_estimate', 0)*100:.1f}cm")
            parts.append(f"  冠幅: {p.get('crown_width', 0):.2f}m")
        if "layers" in analysis:
            l = analysis["layers"]
            parts.append("【垂直分层】")
            for layer_name, pct in l.items():
                parts.append(f"  {layer_name}: {pct:.1f}%")
        return "\n".join(parts)


def test_projector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector = LLMProjector(token_dim=128, llm_dim=4096, num_queries=256).to(device)
    token_features = torch.randn(2, 1024, 128).to(device)
    llm_embeds = projector(token_features)
    print("=== LLMProjector 测试 ===")
    print(f"输入: {token_features.shape}")
    print(f"输出(LLM embeddings): {llm_embeds.shape}")
    print("✓ Projector测试通过")


if __name__ == "__main__":
    test_projector()
