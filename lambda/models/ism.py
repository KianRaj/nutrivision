"""
Internal Semantic Modeling (ISM) — from IGSMNet Section 3.3.

Contains:
1. Dynamic Position Encoding (DPE) — learnable MLP-based relative position bias
2. Fine-Grained Modeling (FGM) — window-based multi-head self-attention
3. ISM block: LN → FGM(+DPE) → residual → LN → MLP → residual

Applied at each pyramid level independently to refine fused features
before the prediction head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPositionEncoding(nn.Module):
    """DPE (Eq. 8): b_ij = MLP(delta_x_ij, delta_y_ij)"""

    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = max(embed_dim // 4, 8)
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LayerNorm(hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(True),
            nn.Linear(hidden, 1),
        )
        self._cache = {}

    def forward(self, window_size: int) -> torch.Tensor:
        G = window_size
        if G not in self._cache or self._cache[G].device != next(self.parameters()).device:
            coords = torch.arange(G, device=next(self.parameters()).device)
            grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
            coords_flat = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1).float()
            deltas = coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)
            self._cache[G] = deltas
        deltas = self._cache[G].to(next(self.parameters()).device)
        G2 = G * G
        return self.net(deltas.view(-1, 2)).view(G2, G2)


class FineGrainedModeling(nn.Module):
    """Window-based MHSA with DPE bias (Eq. 7)."""

    def __init__(self, embed_dim: int, num_heads: int = 4, window_size: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dpe = DynamicPositionEncoding(embed_dim)

    def _window_partition(self, x, G):
        B, H, W, C = x.shape
        pad_h = (G - H % G) % G
        pad_w = (G - W % G) % G
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape
        nH, nW = Hp // G, Wp // G
        x = x.view(B, nH, G, nW, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B * nH * nW, G, G, C), (Hp, Wp, nH, nW, pad_h, pad_w)

    def _window_unpartition(self, x, info, B):
        Hp, Wp, nH, nW, pad_h, pad_w = info
        G = self.window_size
        C = x.shape[-1]
        x = x.view(B, nH, nW, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :Hp - pad_h, :Wp - pad_w, :].contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        G = self.window_size
        windows, info = self._window_partition(x, G)
        nW = windows.shape[0]
        G2 = G * G
        tokens = windows.view(nW, G2, C)
        qkv = self.qkv(tokens).reshape(nW, G2, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn + self.dpe(G).unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(nW, G2, C)
        out = self.proj(out).view(nW, G, G, C)
        return self._window_unpartition(out, info, B)


class ISMBlock(nn.Module):
    """LN → FGM → residual → LN → MLP → residual (Eq. 11)"""

    def __init__(self, embed_dim: int, num_heads: int = 4, window_size: int = 7,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.fgm = FineGrainedModeling(embed_dim, num_heads, window_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fgm(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class InternalSemanticModeling(nn.Module):
    """ISM applied at each pyramid level. L stacked blocks per level."""

    def __init__(self, channels: list, num_blocks: int = 2, num_heads: int = 4,
                 window_size: int = 7, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.levels = nn.ModuleList()
        for c in channels:
            blocks = nn.Sequential(*[
                ISMBlock(c, num_heads, window_size, mlp_ratio, drop)
                for _ in range(num_blocks)
            ])
            self.levels.append(blocks)

    def forward(self, feats: list) -> list:
        """
        Args:
            feats: list of 4 (B, C_i, H_i, W_i) — NCHW format
        Returns:
            refined: list of 4 (B, C_i, H_i, W_i)
        """
        out = []
        for i, feat in enumerate(feats):
            B, C, H, W = feat.shape
            x = feat.permute(0, 2, 3, 1)  # → (B, H, W, C)
            x = self.levels[i](x)
            x = x.permute(0, 3, 1, 2)     # → (B, C, H, W)
            out.append(x)
        return out
