"""
Model for KV Cache Prediction — Scheme 1: Weight Sharing
Key changes vs v5:
  - 32 PerLayerKVPredictors share ONE set of weights
  - A learnable layer_embedding(32, embed_dim) distinguishes layers
  - HiddenStateEncoder is unchanged (shared)
  - Residual learning: pred = reuse + delta

Expected params: ~14.5M  (vs 325M original)
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


class SharedKVPredictor(nn.Module):
    """
    Shared KV delta predictor for ALL transformer layers.
    A layer_embedding is added to the fused features to distinguish layers.

    Flow:
    1. Mean-pool reuse_kv across heads → (B, 2, 256, 128) → flatten → (B, 256, 256)
    2. Project reuse features → (B, 256, reuse_proj_dim)
    3. Concat [hidden_embed, reuse_proj, layer_embed] → fused
    4. MLP → delta (B, 256, 2*32*128)
    5. pred = reuse + delta (residual)
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        kv_dim: int = 128,
        embed_dim: int = 256,
        reuse_proj_dim: int = 256,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_dim = kv_dim
        self.output_dim = 2 * num_heads * kv_dim  # 8192

        # Learnable layer embedding to distinguish layers
        self.layer_embedding = nn.Embedding(num_layers, embed_dim)

        # Reuse KV projector: (B, 256, 2*128) → (B, 256, reuse_proj_dim)
        self.reuse_proj = nn.Sequential(
            nn.Linear(2 * kv_dim, reuse_proj_dim),
            nn.LayerNorm(reuse_proj_dim),
            nn.GELU(),
        )

        # Delta MLP: fused features → delta KV
        # Input = hidden_embed + reuse_proj + layer_embed
        fused_dim = embed_dim + reuse_proj_dim + embed_dim
        self.delta_mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Last layer: small std for stable residual learning start
        last_linear = self.delta_mlp[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last_linear.bias)

        # Layer embedding: small init
        nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_embed: torch.Tensor,
        reuse_layer: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        hidden_embed: (B, 256, embed_dim) — shared hidden state encoding
        reuse_layer:  (B, 2, 32, 256, 128) — reuse KV for this layer
        layer_idx:    int — which layer (0-31)
        returns:      (B, 2, 32, 256, 128) — predicted KV for this layer
        """
        B = hidden_embed.shape[0]

        # Mean-pool across heads: (B, 2, 32, 256, 128) → (B, 2, 256, 128)
        reuse_mean = reuse_layer.float().mean(dim=2)

        # Reshape to (B, 256, 2*128=256)
        reuse_flat = reuse_mean.permute(0, 2, 1, 3).reshape(B, 256, -1)

        # Project reuse features
        reuse_proj = self.reuse_proj(reuse_flat)      # (B, 256, reuse_proj_dim)

        # Layer embedding: (embed_dim,) → (1, 1, embed_dim) → broadcast to (B, 256, embed_dim)
        layer_emb = self.layer_embedding(
            torch.tensor(layer_idx, device=hidden_embed.device)
        )
        layer_emb = layer_emb.unsqueeze(0).unsqueeze(0).expand(B, 256, -1)

        # Fuse: [hidden_embed, reuse_proj, layer_emb]
        fused = torch.cat([hidden_embed, reuse_proj, layer_emb], dim=-1)

        # Predict delta
        delta = self.delta_mlp(fused)                 # (B, 256, 8192)

        # Reshape: (B, 256, 2, 32, 128) → (B, 2, 32, 256, 128)
        delta = delta.reshape(B, 256, 2, self.num_heads, self.kv_dim)
        delta = delta.permute(0, 2, 3, 1, 4).contiguous()

        # Residual: pred = reuse + delta
        pred = reuse_layer.float() + delta
        return pred


class KVCacheModelWS(nn.Module):
    """
    Weight-Sharing model: HiddenStateEncoder + 1 SharedKVPredictor.
    All 32 layers share the same predictor, distinguished by layer_embedding.
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        kv_dim: int = 128,
        hidden_state_dim: int = 4096,
        embed_dim: int = 256,
        reuse_proj_dim: int = 256,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Shared hidden state encoder
        self.hidden_encoder = HiddenStateEncoder(
            input_dim=hidden_state_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )

        # Single shared predictor for all layers
        self.shared_predictor = SharedKVPredictor(
            num_layers=num_layers,
            num_heads=num_heads,
            kv_dim=kv_dim,
            embed_dim=embed_dim,
            reuse_proj_dim=reuse_proj_dim,
            hidden_dim=hidden_dim,
        )

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

        # Per-layer prediction using shared predictor
        pred_kv_list = []
        for layer_idx in range(self.num_layers):
            reuse_layer = reuse_kv[:, layer_idx]           # (B, 2, 32, 256, 128)
            pred_layer = self.shared_predictor(hidden_embed, reuse_layer, layer_idx)
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
