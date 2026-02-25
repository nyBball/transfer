"""
Model for KV Cache Prediction — Scheme 3: Low-Rank Decomposition
Key changes vs v5:
  - The largest linear layer Linear(1024, 8192) is replaced by
    low-rank factorization: Linear(1024, r) → Linear(r, 8192)
  - Per-layer INDEPENDENT models are kept (no weight sharing)
  - Residual learning: pred = reuse + delta

With hidden_dim=512, rank=32, expected params: ~40M  (vs 325M original)
With hidden_dim=512, rank=16, expected params: ~30M
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class HiddenStateEncoder(nn.Module):
    """
    Encode hidden states (B, 256, 4096) -> (B, 256, embed_dim)
    Two-layer MLP with LayerNorm.
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024, embed_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, 256, 4096) float32
        returns: (B, 256, embed_dim)
        """
        return self.encoder(hidden_states)


class LowRankLinear(nn.Module):
    """
    Low-rank linear layer: W ≈ A @ B
    A: (in_features, rank), B: (rank, out_features)
    Params: in_features * rank + rank * out_features  (vs in_features * out_features)
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class PerLayerKVPredictorLR(nn.Module):
    """
    Independent KV delta predictor for a single transformer layer.
    Uses low-rank decomposition on the output layer to reduce parameters.

    Flow:
    1. Mean-pool reuse_kv across heads → (B, 2, 256, 128) → flatten → (B, 256, 256)
    2. Project reuse features → (B, 256, reuse_proj_dim)
    3. Concat [hidden_embed, reuse_proj] → fused
    4. MLP with low-rank output → delta (B, 256, 2*32*128)
    5. pred = reuse + delta (residual)
    """

    def __init__(
        self,
        num_heads: int = 32,
        kv_dim: int = 128,
        embed_dim: int = 256,
        reuse_proj_dim: int = 256,
        hidden_dim: int = 512,
        output_rank: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_dim = kv_dim
        self.output_dim = 2 * num_heads * kv_dim  # 8192

        # Reuse KV projector: (B, 256, 2*128) → (B, 256, reuse_proj_dim)
        self.reuse_proj = nn.Sequential(
            nn.Linear(2 * kv_dim, reuse_proj_dim),
            nn.LayerNorm(reuse_proj_dim),
            nn.GELU(),
        )

        # Delta MLP with low-rank output layer
        fused_dim = embed_dim + reuse_proj_dim
        self.delta_mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Low-rank output: Linear(hidden_dim, rank) → Linear(rank, output_dim)
        self.output_head = LowRankLinear(hidden_dim, self.output_dim, rank=output_rank)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Low-rank output: small init for stable residual start
        nn.init.normal_(self.output_head.up.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_head.up.bias)

    def forward(
        self,
        hidden_embed: torch.Tensor,
        reuse_layer: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_embed: (B, 256, embed_dim) — shared hidden state encoding
        reuse_layer:  (B, 2, 32, 256, 128) — reuse KV for this layer
        returns:      (B, 2, 32, 256, 128) — predicted KV for this layer
        """
        B = hidden_embed.shape[0]

        # Mean-pool across heads: (B, 2, 32, 256, 128) → (B, 2, 256, 128)
        reuse_mean = reuse_layer.float().mean(dim=2)

        # Reshape to (B, 256, 2*128=256)
        reuse_flat = reuse_mean.permute(0, 2, 1, 3).reshape(B, 256, -1)

        # Project reuse features
        reuse_proj = self.reuse_proj(reuse_flat)      # (B, 256, reuse_proj_dim)

        # Fuse: [hidden_embed, reuse_proj]
        fused = torch.cat([hidden_embed, reuse_proj], dim=-1)

        # MLP body → low-rank output
        h = self.delta_mlp(fused)                     # (B, 256, hidden_dim)
        delta = self.output_head(h)                   # (B, 256, 8192)

        # Reshape: (B, 256, 2, 32, 128) → (B, 2, 32, 256, 128)
        delta = delta.reshape(B, 256, 2, self.num_heads, self.kv_dim)
        delta = delta.permute(0, 2, 3, 1, 4).contiguous()

        # Residual: pred = reuse + delta
        pred = reuse_layer.float() + delta
        return pred


class KVCacheModelLR(nn.Module):
    """
    Low-Rank model: HiddenStateEncoder + 32 independent PerLayerKVPredictorLR.
    Each layer has its own parameters, but the output projection uses low-rank
    decomposition to significantly reduce parameter count.
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        kv_dim: int = 128,
        hidden_state_dim: int = 4096,
        embed_dim: int = 256,
        reuse_proj_dim: int = 256,
        hidden_dim: int = 512,
        output_rank: int = 32,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Shared hidden state encoder
        self.hidden_encoder = HiddenStateEncoder(
            input_dim=hidden_state_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )

        # Per-layer INDEPENDENT low-rank predictors
        self.layer_predictors = nn.ModuleList([
            PerLayerKVPredictorLR(
                num_heads=num_heads,
                kv_dim=kv_dim,
                embed_dim=embed_dim,
                reuse_proj_dim=reuse_proj_dim,
                hidden_dim=hidden_dim,
                output_rank=output_rank,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        reuse_kv: torch.Tensor,
        stable_mask: torch.Tensor,
        target_kv: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        hidden_states: (B, 256, 4096) float32
        reuse_kv:      (B, 32, 2, 32, 256, 128) bfloat16
        stable_mask:   (B, 256) bool
        target_kv:     (B, 32, 2, 32, 256, 128) bfloat16 (optional)
        """
        # Shared encoding (computed once, reused for all layers)
        hidden_embed = self.hidden_encoder(hidden_states)  # (B, 256, embed_dim)

        # Per-layer prediction
        pred_kv_list = []
        for layer_idx in range(self.num_layers):
            reuse_layer = reuse_kv[:, layer_idx]           # (B, 2, 32, 256, 128)
            pred_layer = self.layer_predictors[layer_idx](hidden_embed, reuse_layer)
            pred_kv_list.append(pred_layer)

        pred_kv = torch.stack(pred_kv_list, dim=1)         # (B, 32, 2, 32, 256, 128)

        result = {"pred_kv": pred_kv}
        if target_kv is not None:
            # Masked loss: only compute on stable patch positions
            mask = stable_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(-1)

            pred_float = pred_kv
            target_float = target_kv.float()

            diff_sq = (pred_float - target_float) ** 2
            masked_diff_sq = diff_sq * mask.float()

            num_stable = stable_mask.sum().float().clamp(min=1.0)
            elements_per_patch = self.num_layers * 2 * 32 * 128
            loss = masked_diff_sq.sum() / (num_stable * elements_per_patch)

            result["loss"] = loss
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
