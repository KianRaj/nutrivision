"""
Ingredient-Conditioned Hierarchical FAFM — Our Novel Contribution.

Combines OmniFood8k's Frequency-Aligned Fusion Module with IGSMNet's
ingredient-guided cross-attention.

Key idea: Inject ingredient semantic information INTO the frequency fusion
process at each scale, so the fusion knows WHAT food to look for.

Architecture per scale:
1. FAFM: FFT decomposition → low/high freq fusion (OmniFood8k Eq. 4-7)
2. Ingredient Cross-Attention: ingredient embedding guides fused features
   (IGSMNet Eq. 5-6)
3. Residual connection

Final: Multi-scale pooling → gated fusion → output vector

Also computes inter-modal alignment loss (OmniFood8k Eq. 8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List


def _make_low_freq_mask(H: int, W: int, ratio: float, device: torch.device) -> torch.Tensor:
    """Create circular low-frequency mask in the FFT frequency domain."""
    yy = torch.arange(H, device=device).view(H, 1).float()
    xx = torch.arange(W, device=device).view(1, W).float()
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = torch.sqrt(torch.tensor(cy**2 + cx**2, device=device))
    mask = (dist <= (ratio * max_r)).float()
    return mask[None, None, :, :]


class FrequencyAlignedFusion(nn.Module):
    """Single-scale FAFM: OmniFood8k Eq. 4-7."""

    def __init__(self, c: int = 256, tau_freq: float = 0.20):
        super().__init__()
        self.c = c
        self.tau_freq = tau_freq
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self._mask_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _get_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = _make_low_freq_mask(H, W, self.tau_freq, device)
        return self._mask_cache[key]

    def forward(self, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = r.shape
        if d.shape[-2:] != r.shape[-2:]:
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        ML = self._get_mask(H, W, r.device)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            r_f, d_f = r.float(), d.float()
            Rf = torch.fft.fftshift(torch.fft.fft2(r_f), dim=(-2, -1))
            Df = torch.fft.fftshift(torch.fft.fft2(d_f), dim=(-2, -1))

            RL = torch.fft.ifft2(torch.fft.ifftshift(Rf * ML, dim=(-2, -1))).real
            RH = torch.fft.ifft2(torch.fft.ifftshift(Rf * (1 - ML), dim=(-2, -1))).real
            DL = torch.fft.ifft2(torch.fft.ifftshift(Df * ML, dim=(-2, -1))).real
            DH = torch.fft.ifft2(torch.fft.ifftshift(Df * (1 - ML), dim=(-2, -1))).real

        FH = RH + DH
        FL = RL + DL
        return self.fuse(torch.cat([FH, FL], dim=1))


class IngredientCrossAttention(nn.Module):
    """
    Cross-attention: ingredient embedding (query) attends to fused visual
    features (key, value). Adapted from IGSMNet Eq. 5-6.

    Q = ingredient (B, 1, C), K = V = visual (B, HW, C)
    Output broadcast to all spatial locations + residual.
    """

    def __init__(self, visual_c: int = 256, clip_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.visual_c = visual_c

        # Project CLIP dim to visual channel dim
        self.clip_proj = nn.Linear(clip_dim, visual_c)
        self.ln_q = nn.LayerNorm(visual_c)
        self.ln_kv = nn.LayerNorm(visual_c)

        self.W_Q = nn.Linear(visual_c, visual_c)
        self.W_K = nn.Linear(visual_c, visual_c)
        self.W_V = nn.Linear(visual_c, visual_c)
        self.out_proj = nn.Linear(visual_c, visual_c)

    def forward(self, fused_feat: torch.Tensor, ingr_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_feat: (B, C, H, W) — FAFM output
            ingr_embed: (B, clip_dim) — ingredient embedding
        Returns:
            guided_feat: (B, C, H, W)
        """
        B, C, H, W = fused_feat.shape

        # Prepare query from ingredient
        ingr = self.clip_proj(ingr_embed).unsqueeze(1)  # (B, 1, C)
        ingr = self.ln_q(ingr)

        # Prepare key/value from visual features
        vis = fused_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        vis = self.ln_kv(vis)

        Q = self.W_Q(ingr)   # (B, 1, C)
        K = self.W_K(vis)    # (B, HW, C)
        V = self.W_V(vis)    # (B, HW, C)

        # Multi-head attention
        head_dim = C // self.num_heads
        Q = Q.view(B, 1, self.num_heads, head_dim).transpose(1, 2)         # (B, H, 1, d)
        K = K.view(B, H * W, self.num_heads, head_dim).transpose(1, 2)     # (B, H, HW, d)
        V = V.view(B, H * W, self.num_heads, head_dim).transpose(1, 2)     # (B, H, HW, d)

        attn = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, 1, C)
        out = self.out_proj(out)  # (B, 1, C)

        # Broadcast to spatial dims + residual (Eq. 6)
        guided = fused_feat + out.permute(0, 2, 1).unsqueeze(-1).expand_as(fused_feat)
        return guided


class IngredientConditionedHierarchicalFAFM(nn.Module):
    """
    4-scale hierarchical fusion with ingredient conditioning.
    At each scale: FAFM → Ingredient Cross-Attention → pool.
    Final: concat all scales → gated fusion → output.
    """

    def __init__(
        self,
        in_c: int = 256,
        out_dim: int = 256,
        n_scales: int = 4,
        clip_dim: int = 512,
        num_heads: int = 4,
        tau_freq: float = 0.20,
    ):
        super().__init__()
        self.n_scales = n_scales

        # Per-scale FAFM
        self.fafms = nn.ModuleList([
            FrequencyAlignedFusion(c=in_c, tau_freq=tau_freq)
            for _ in range(n_scales)
        ])

        # Per-scale ingredient cross-attention
        self.ingr_attn = nn.ModuleList([
            IngredientCrossAttention(visual_c=in_c, clip_dim=clip_dim, num_heads=num_heads)
            for _ in range(n_scales)
        ])

        # Per-scale pooling projections
        self.scale_pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(in_c, out_dim), nn.ReLU(inplace=True),
            ) for _ in range(n_scales)
        ])

        # Learnable gates for each scale
        self.scale_gates = nn.Parameter(torch.ones(n_scales))

        # Channel mask: top-k selection
        self.channel_mask_ratio = 0.75

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(n_scales * out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
        )

        # Alignment projectors for info-NCE loss (OmniFood8k Eq. 8)
        self.align_proj_r = nn.ModuleList([
            nn.Sequential(nn.Linear(in_c, 128), nn.ReLU(True), nn.Linear(128, 128))
            for _ in range(n_scales)
        ])
        self.align_proj_d = nn.ModuleList([
            nn.Sequential(nn.Linear(in_c, 128), nn.ReLU(True), nn.Linear(128, 128))
            for _ in range(n_scales)
        ])

    def _info_nce(self, fr: torch.Tensor, fd: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        fr = F.normalize(fr, dim=1)
        fd = F.normalize(fd, dim=1)
        logits = (fr @ fd.t()) / tau
        labels = torch.arange(fr.size(0), device=fr.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(
        self,
        rgb_scales: List[torch.Tensor],
        depth_scales: List[torch.Tensor],
        ingr_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_scales: list of 4 (B, C, H_i, W_i)
            depth_scales: list of 4 (B, C, H_i, W_i)
            ingr_embed: (B, clip_dim)
        Returns:
            fused_feat: (B, out_dim)
            align_loss: scalar
        """
        scale_feats = []
        align_loss = torch.tensor(0.0, device=rgb_scales[0].device)

        for i in range(self.n_scales):
            # 1. Frequency-aligned fusion
            fused = self.fafms[i](rgb_scales[i], depth_scales[i])

            # 2. Ingredient-guided cross-attention
            guided = self.ingr_attn[i](fused, ingr_embed)

            # 3. Pool this scale
            pooled = self.scale_pools[i](guided)  # (B, out_dim)
            scale_feats.append(pooled)

            # 4. Alignment loss
            r_pool = F.adaptive_avg_pool2d(rgb_scales[i], 1).flatten(1)
            d_pool = F.adaptive_avg_pool2d(depth_scales[i], 1).flatten(1)
            fr = self.align_proj_r[i](r_pool)
            fd = self.align_proj_d[i](d_pool)
            align_loss = align_loss + self._info_nce(fr, fd)

        align_loss = align_loss / self.n_scales

        # Gated scale fusion
        gates = torch.sigmoid(self.scale_gates)  # (n_scales,)
        gated = [scale_feats[i] * gates[i] for i in range(self.n_scales)]

        # Channel mask: keep top-k channels by magnitude
        concat = torch.cat(gated, dim=1)  # (B, n_scales * out_dim)
        if self.training:
            k = int(concat.size(1) * self.channel_mask_ratio)
            topk_vals, topk_idx = concat.abs().topk(k, dim=1)
            mask = torch.zeros_like(concat)
            mask.scatter_(1, topk_idx, 1.0)
            concat = concat * mask

        # Final fusion
        fused_feat = self.fusion_mlp(concat)  # (B, out_dim)
        return fused_feat, align_loss
