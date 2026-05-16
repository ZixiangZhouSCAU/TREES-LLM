"""
PointLLM-style VQ-VAE Tokenizer for Tree Point Clouds
将点云特征离散化为token序列，供LLM理解3D结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VectorQuantizer(nn.Module):
    """
    VQ-VAE离散码本：特征向量 → 离散token索引
    代码本学习：EMA更新或梯度下降
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # 码本: (num_embeddings, embedding_dim)
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))
        self.ema_w.copy_(self.embedding)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        z: (B, N, D) 编码特征
        返回: (quantized, info_dict)
            quantized: (B, N, D) 量化为最近邻码字
            info: {token_indices, latent_loss, commitment_loss, perplexity}
        """
        B, N, D = z.shape
        assert D == self.embedding_dim, f"特征维度{D} != 码本维度{self.embedding_dim}"

        # flatten: (B*N, D)
        z_flat = z.view(-1, D)
        # 计算与所有码字距离: (B*N, num_embeddings)
        d = torch.sum(z_flat ** 2, dim=-1, keepdim=True) \
            + torch.sum(self.embedding ** 2, dim=-1) \
            - 2 * torch.matmul(z_flat, self.embedding.t())

        # 最近邻索引: (B*N,)
        token_indices = torch.argmin(d, dim=-1)
        token_indices = token_indices.view(B, N)

        # 取码字: (B, N, D)
        quantized = self._get_codebook(token_indices)

        # VQ损失
        latent_loss = F.mse_loss(z.detach(), quantized)
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = latent_loss + self.commitment_cost * commitment_loss

        # 梯度直通（straight-through estimator）
        quantized = z + (quantized - z).detach()

        # EMA码本更新
        if self.training:
            self._update_ema(z_flat, token_indices)

        # 困惑度
        avg_probs = torch.softmax(d.float(), dim=-1).mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.epsilon)))

        info = {
            "token_indices": token_indices,
            "perplexity": perplexity,
            "commitment_loss": commitment_loss,
            "vq_loss": vq_loss,
            "latent_loss": latent_loss,
        }
        return quantized, info

    def _get_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """根据索引获取码字"""
        B, N = indices.shape
        indices_flat = indices.view(-1)
        z_q = F.embedding(indices_flat, self.embedding)
        return z_q.view(B, N, self.embedding_dim)

    def _update_ema(self, z_flat: torch.Tensor, token_indices: torch.Tensor):
        """EMA更新码本"""
        B_N = z_flat.shape[0]
        batch_size = 1  # 一个batch内更新

        # 统计每个码字分配数量
        encodings = torch.zeros(B_N, self.num_embeddings, device=z_flat.device)
        encodings.scatter_(1, token_indices.unsqueeze(-1), 1.0)

        # EMA cluster size
        self.ema_cluster_size = self.decay * self.ema_cluster_size \
            + (1 - self.decay) * encodings.sum(0)

        # n: 每类样本数
        n = self.ema_cluster_size.sum()
        n = torch.clamp(n, min=self.epsilon)
        cluster_size = (self.ema_cluster_size >= self.epsilon) * self.ema_cluster_size + (self.ema_cluster_size < self.epsilon) * 1.0

        # 更新码字
        dw = torch.matmul(encodings.t(), z_flat)
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
        self.embedding = self.ema_w / cluster_size.unsqueeze(-1)

    def get_token_count(self, token_indices: torch.Tensor) -> Dict[str, int]:
        """统计token分布"""
        unique, counts = torch.unique(token_indices.view(-1), return_counts=True)
        return {"unique_tokens": len(unique), "total_tokens": token_indices.numel()}


class TreePointTokenizer(nn.Module):
    """
    树木点云专用VQ-VAE Tokenizer
    架构: 编码特征 → VQ离散化 → token序列

    与PointLLM风格一致：点云 → 离散token → LLM
    """

    def __init__(self, input_dim: int = 512, token_dim: int = 128,
                 num_tokens: int = 2048, commitment_cost: float = 0.25):
        super().__init__()
        self.input_dim = input_dim
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        # 压缩层: (B*N, 512) -> (B*N, token_dim)
        self.encoder_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, token_dim),
        )

        # VQ码本
        self.vq = VectorQuantizer(
            num_embeddings=num_tokens,
            embedding_dim=token_dim,
            commitment_cost=commitment_cost,
        )

        # 用于从token恢复特征（推理时可选）
        self.decoder_proj = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        features: (B, N, input_dim) 来自TreePointEncoder的特征
        返回:
            quantized: (B, N, token_dim) 量化的token特征
            info: {token_indices, perplexity, vq_loss, ...}
        """
        # 投影到token_dim
        z = self.encoder_proj(features)

        # VQ离散化
        quantized, info = self.vq(z)

        return quantized, info

    def encode(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """纯编码：返回token索引"""
        z = self.encoder_proj(features)
        _, info = self.vq(z)
        return info["token_indices"], info

    def decode(self, token_indices: torch.Tensor) -> torch.Tensor:
        """从token索引恢复特征"""
        B, N = token_indices.shape
        z_q = F.embedding(token_indices.view(-1), self.vq.embedding)
        z_q = z_q.view(B, N, self.token_dim)
        return self.decoder_proj(z_q)

    def get_token_text_description(self, token_indices: torch.Tensor) -> str:
        """将token索引序列转为文本描述（用于GLM）"""
        unique, counts = torch.unique(token_indices.view(-1), return_counts=True)
        top_k = torch.topk(counts, min(20, len(counts)))
        total = counts.sum().item()
        desc_parts = []
        for idx, cnt in zip(unique[top_k.indices], top_k.values):
            pct = cnt.item() / total * 100
            desc_parts.append(f"token#{idx.item()}({pct:.1f}%)")
        return f"共{len(unique)}种token，前20高频: {', '.join(desc_parts)}"


def test_tokenizer():
    """测试tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟encoder输出特征
    B, N = 2, 1024
    input_dim, token_dim, num_tokens = 512, 128, 2048

    tokenizer = TreePointTokenizer(input_dim, token_dim, num_tokens).to(device)
    features = torch.randn(B, N, input_dim).to(device)

    with torch.no_grad():
        quantized, info = tokenizer(features)
        tokens, _ = tokenizer.encode(features)
        recovered = tokenizer.decode(tokens)

    print("=== TreePointTokenizer 测试 ===")
    print(f"输入特征: {features.shape}")
    print(f"量化特征: {quantized.shape}")
    print(f"Token索引: {tokens.shape}, 范围[{tokens.min()}, {tokens.max()}]")
    print(f"Perplexity: {info['perplexity']:.2f}")
    print(f"VQ Loss: {info['vq_loss']:.4f}")
    print(f"恢复特征: {recovered.shape}, diff={torch.mean((features - recovered)**2).item():.4f}")
    print("✓ Tokenizer测试通过")


if __name__ == "__main__":
    test_tokenizer()