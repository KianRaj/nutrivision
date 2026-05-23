"""
Ingredient-Aware Mask-based Prediction Head (MPH) — Enhanced from OmniFood8k Section 4.3.

Enhancement: The MPH's channel selection and gating is conditioned on
ingredient embeddings, so different foods activate different channels.

Architecture:
1. Concatenate: fused_feat (256) + cls_rgb (512) + cls_depth (512) + ingr_embed (512)
2. Ingredient-conditioned channel gate
3. Dynamic channel mask (top-k)
4. Regression MLP → 5 nutrition values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IngredientAwareMPH(nn.Module):
    def __init__(
        self,
        fused_dim: int = 256,
        cls_dim: int = 512,
        clip_dim: int = 512,
        hidden_dim: int = 512,
        num_tasks: int = 5,
        channel_mask_ratio: float = 0.75,
        drop: float = 0.1,
    ):
        super().__init__()
        # Total input dim: fused + cls_rgb + cls_depth
        in_dim = fused_dim + cls_dim + cls_dim

        # Ingredient-conditioned gate: ingredient embedding → gate weights for input channels
        self.ingr_gate = nn.Sequential(
            nn.Linear(clip_dim, in_dim),
            nn.Sigmoid(),
        )

        self.channel_mask_ratio = channel_mask_ratio

        # Cross-attention: input features (Q) attend to ingredient (K,V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=4, batch_first=True, dropout=drop,
        )
        self.cross_norm = nn.LayerNorm(in_dim)

        # Gated fusion after cross-attention
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid(),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
            nn.Linear(hidden_dim // 2, num_tasks),
        )

    def forward(
        self,
        fused_feat: torch.Tensor,
        cls_rgb: torch.Tensor,
        cls_depth: torch.Tensor,
        ingr_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_feat: (B, fused_dim) — from hierarchical FAFM
            cls_rgb: (B, cls_dim) — global RGB feature
            cls_depth: (B, cls_dim) — global depth feature
            ingr_embed: (B, clip_dim) — ingredient embedding
        Returns:
            pred: (B, num_tasks) — [cal, mass, fat, carb, protein]
        """
        # Concatenate multi-source features
        x = torch.cat([fused_feat, cls_rgb, cls_depth], dim=1)  # (B, in_dim)

        # 1. Ingredient-conditioned gating
        gate = self.ingr_gate(ingr_embed)  # (B, in_dim)
        x = x * gate

        # 2. Dynamic channel mask (keep top-k by magnitude)
        if self.training:
            k = int(x.size(1) * self.channel_mask_ratio)
            _, topk_idx = x.abs().topk(k, dim=1)
            mask = torch.zeros_like(x)
            mask.scatter_(1, topk_idx, 1.0)
            x = x * mask

        # 3. Cross-attention with ingredient embedding
        x_seq = x.unsqueeze(1)             # (B, 1, in_dim)
        ingr_seq = ingr_embed.unsqueeze(1)  # (B, 1, clip_dim)
        # Pad ingredient to match in_dim for cross-attention
        pad_dim = x.size(1) - ingr_embed.size(1)
        if pad_dim > 0:
            ingr_seq = F.pad(ingr_seq, (0, pad_dim))
        attn_out, _ = self.cross_attn(x_seq, ingr_seq, ingr_seq)
        x = self.cross_norm(x + attn_out.squeeze(1))

        # 4. Gated fusion
        g = self.gate_mlp(x)
        x = x * g

        # 5. Regression
        pred = self.head(x)
        return pred
