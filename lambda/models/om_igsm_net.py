"""
Om-IGSMNet: Ingredient-Aware Hierarchical Frequency Fusion Network.

Combines the best of OmniFood8k and IGSMNet:
- Dual ConvNeXt-Base backbone (true multi-scale, from OmniFood8k)
- SSRA for monocular depth refinement (OmniFood8k Eq. 1-3)
- ISM for fine-grained feature refinement (IGSMNet Section 3.3)
- Ingredient-Conditioned Hierarchical FAFM (our novel contribution)
- Ingredient-Aware MPH (our novel enhanced prediction head)

Input: Single RGB image + ingredient text (CLIP-encoded).
Depth estimated from RGB via DepthAnythingV2 (precomputed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DualConvNeXtBackbone
from .ssra import SSRA
from .ism import InternalSemanticModeling
from .ingredient_fafm import IngredientConditionedHierarchicalFAFM
from .mph import IngredientAwareMPH


class OmIGSMNet(nn.Module):
    def __init__(
        self,
        # Backbone
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k",
        out_c: int = 256,
        cls_dim: int = 512,
        unfreeze_stages: int = 2,
        # ISM
        ism_blocks: int = 2,
        ism_heads: int = 4,
        ism_window: int = 7,
        # CLIP
        clip_dim: int = 512,
        # FAFM
        tau_freq: float = 0.20,
        fafm_heads: int = 4,
        # MPH
        mph_hidden: int = 512,
        num_tasks: int = 5,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        # 1. SSRA for depth refinement
        self.ssra = SSRA(in_channels=1)

        # 2. Dual ConvNeXt backbone
        self.backbone = DualConvNeXtBackbone(
            out_c=out_c, cls_dim=cls_dim,
            unfreeze_stages=unfreeze_stages,
            model_name=backbone_name,
        )

        # 3. ISM for fine-grained feature refinement
        self.ism_rgb = InternalSemanticModeling(
            channels=[out_c] * 4, num_blocks=ism_blocks,
            num_heads=ism_heads, window_size=ism_window,
            mlp_ratio=4.0, drop=drop_rate,
        )
        self.ism_depth = InternalSemanticModeling(
            channels=[out_c] * 4, num_blocks=ism_blocks,
            num_heads=ism_heads, window_size=ism_window,
            mlp_ratio=4.0, drop=drop_rate,
        )

        # 4. Ingredient-Conditioned Hierarchical FAFM
        self.hier_fafm = IngredientConditionedHierarchicalFAFM(
            in_c=out_c, out_dim=out_c, n_scales=4,
            clip_dim=clip_dim, num_heads=fafm_heads,
            tau_freq=tau_freq,
        )

        # 5. Ingredient-Aware MPH
        self.mph = IngredientAwareMPH(
            fused_dim=out_c, cls_dim=cls_dim, clip_dim=clip_dim,
            hidden_dim=mph_hidden, num_tasks=num_tasks,
            drop=drop_rate,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth_mono: torch.Tensor,
        ingr_embed: torch.Tensor,
    ):
        """
        Args:
            rgb: (B, 3, 224, 224) — input RGB image
            depth_mono: (B, 1, H, W) — monocular depth (precomputed)
            ingr_embed: (B, clip_dim) — precomputed CLIP ingredient embedding
        Returns:
            pred: (B, num_tasks) — [cal, mass, fat, carb, protein]
            losses: dict with auxiliary losses
        """
        # Step 1: Refine depth with SSRA
        depth_refined = self.ssra(depth_mono)

        # Step 2: Extract multi-scale features
        rgb_scales, depth_scales, cls_rgb, cls_depth = self.backbone(rgb, depth_refined)

        # Step 3: ISM refinement on each modality
        rgb_refined = self.ism_rgb(rgb_scales)
        depth_refined_scales = self.ism_depth(depth_scales)

        # Step 4: Ingredient-Conditioned Hierarchical FAFM
        fused_feat, align_loss = self.hier_fafm(
            rgb_refined, depth_refined_scales, ingr_embed
        )

        # Step 5: Ingredient-Aware MPH prediction
        pred = self.mph(fused_feat, cls_rgb, cls_depth, ingr_embed)

        losses = {"align_loss": align_loss}
        return pred, losses

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
