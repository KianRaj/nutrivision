"""
Dual ConvNeXt-Base backbone for RGB and estimated depth.
Produces 4-scale feature pyramids from each modality.
ConvNeXt-Base stages: [128, 256, 512, 1024] at strides [4, 8, 16, 32].
All scales projected to uniform out_c=256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DualConvNeXtBackbone(nn.Module):
    STAGE_CHANNELS = [128, 256, 512, 1024]  # ConvNeXt-Base

    def __init__(
        self,
        out_c: int = 256,
        cls_dim: int = 512,
        unfreeze_stages: int = 2,
        model_name: str = "convnext_base.fb_in22k_ft_in1k",
    ):
        super().__init__()
        self.out_c = out_c

        # RGB backbone
        self.rgb_backbone = timm.create_model(
            model_name, pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # Depth backbone (separate weights, like OmniFood8k)
        self.depth_backbone = timm.create_model(
            model_name, pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # Depth input adapter: 1ch -> 3ch (Eq. 1-3 in OmniFood8k)
        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.constant_(self.depth_adapter.weight, 1.0 / 3.0)
        nn.init.zeros_(self.depth_adapter.bias)

        # Freeze early stages
        self._freeze_stages(self.rgb_backbone, unfreeze_stages)
        self._freeze_stages(self.depth_backbone, unfreeze_stages)

        # Per-stage 1x1 projection to uniform channel count
        self.rgb_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ) for ch in self.STAGE_CHANNELS
        ])
        self.depth_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ) for ch in self.STAGE_CHANNELS
        ])

        # CLS projections (global avg pool of last stage)
        self.rgb_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.STAGE_CHANNELS[-1], cls_dim),
            nn.ReLU(inplace=True),
        )
        self.depth_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.STAGE_CHANNELS[-1], cls_dim),
            nn.ReLU(inplace=True),
        )

    def _freeze_stages(self, backbone, unfreeze_stages: int):
        for p in backbone.parameters():
            p.requires_grad = False
        if unfreeze_stages <= 0:
            return
        stage_patterns = []
        for i in range(4 - unfreeze_stages, 4):
            stage_patterns.extend([f"stages_{i}", f"stages.{i}"])
        for name, param in backbone.named_parameters():
            for pat in stage_patterns:
                if pat in name:
                    param.requires_grad = True
                    break

    def forward(self, rgb: torch.Tensor, depth_1ch: torch.Tensor):
        """
        Args:
            rgb: (B, 3, 224, 224)
            depth_1ch: (B, 1, H, W) — SSRA-refined monocular depth
        Returns:
            rgb_scales: list of 4 (B, out_c, H_i, W_i)
            depth_scales: list of 4 (B, out_c, H_i, W_i)
            cls_rgb: (B, cls_dim)
            cls_depth: (B, cls_dim)
        """
        rgb_feats = self.rgb_backbone(rgb)
        rgb_scales = [self.rgb_projs[i](rgb_feats[i]) for i in range(4)]
        cls_rgb = self.rgb_cls(rgb_feats[-1])

        depth_224 = F.interpolate(depth_1ch, size=(224, 224), mode="bilinear", align_corners=False)
        depth_3ch = self.depth_adapter(depth_224)
        depth_feats = self.depth_backbone(depth_3ch)
        depth_scales = [self.depth_projs[i](depth_feats[i]) for i in range(4)]
        cls_depth = self.depth_cls(depth_feats[-1])

        return rgb_scales, depth_scales, cls_rgb, cls_depth
