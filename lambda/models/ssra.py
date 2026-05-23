"""
Scale-Shift Residual Adapter (SSRA) — OmniFood8k Section 4.1, Eq. 1-3.

Refines monocular depth estimation with:
1. Global scale-shift: d_global = α·d_mono + β  (Eq. 1)
2. Local residual refinement: d_res = f_θ(d_global)  (Eq. 2)
3. Final: d_out = d_global + d_res  (Eq. 3)

Since we use single RGB (no category ID at test time), this version
learns a single global α/β instead of per-category embeddings.
"""

import torch
import torch.nn as nn


class SSRA(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # Global learnable scale and shift (Eq. 1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

        # Local residual refinement CNN (Eq. 2-3)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )

    def forward(self, depth_mono: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_mono: (B, 1, H, W) raw monocular depth
        Returns:
            depth_refined: (B, 1, H, W)
        """
        d_global = self.alpha * depth_mono + self.beta
        d_res = self.refine(d_global)
        return d_global + d_res
