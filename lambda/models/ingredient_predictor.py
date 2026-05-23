"""
Ingredient Predictor — Our Key Novelty.

At training time: uses ground-truth CLIP ingredient embeddings (like IGSMNet).
At test time: predicts ingredient embedding from RGB features (unlike IGSMNet).

This makes the model practical for real-world use (single RGB, no ingredient
labels needed) while still leveraging ingredient semantics during training.

Architecture:
1. Lightweight classifier on pooled RGB features → ingredient class probabilities
2. Learnable ingredient embedding matrix (num_classes × clip_dim)
3. Predicted embedding = weighted sum of class embeddings (soft attention)
4. Training loss: CE(predicted_class, gt_class) + MSE(pred_embed, gt_clip_embed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IngredientPredictor(nn.Module):
    def __init__(
        self,
        visual_dim: int = 512,
        clip_dim: int = 512,
        num_classes: int = 200,
        hidden_dim: int = 256,
    ):
        """
        Args:
            visual_dim: dimension of pooled RGB CLS feature
            clip_dim: CLIP embedding dimension (512 for ViT-B/32)
            num_classes: number of food categories for classification
            hidden_dim: intermediate MLP dimension
        """
        super().__init__()
        self.clip_dim = clip_dim
        self.num_classes = num_classes

        # Classification head: visual features → food category
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

        # Learnable ingredient embedding per class
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes, clip_dim) * 0.02
        )

        # Projection: refine predicted embedding to match CLIP space
        self.embed_proj = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.ReLU(inplace=True),
            nn.Linear(clip_dim, clip_dim),
        )

    def forward(self, cls_rgb: torch.Tensor, gt_clip_embed: torch.Tensor = None):
        """
        Args:
            cls_rgb: (B, visual_dim) — global RGB feature from backbone
            gt_clip_embed: (B, clip_dim) — ground-truth CLIP embedding (training only)
        Returns:
            ingr_embed: (B, clip_dim) — predicted/gt ingredient embedding
            cls_logits: (B, num_classes) — classification logits
            embed_loss: scalar — MSE between predicted and GT CLIP embedding
        """
        # Predict food category
        cls_logits = self.classifier(cls_rgb)  # (B, num_classes)

        # Soft ingredient embedding via class probabilities
        cls_probs = F.softmax(cls_logits, dim=-1)  # (B, num_classes)
        pred_embed = cls_probs @ self.class_embeddings  # (B, clip_dim)
        pred_embed = self.embed_proj(pred_embed)  # refine

        # During training: use GT CLIP embedding for guidance, compute alignment loss
        # During inference: use predicted embedding
        if gt_clip_embed is not None and self.training:
            ingr_embed = gt_clip_embed  # teacher forcing during training
            embed_loss = F.mse_loss(pred_embed, gt_clip_embed.detach())
        else:
            ingr_embed = pred_embed
            embed_loss = torch.tensor(0.0, device=cls_rgb.device)

        return ingr_embed, cls_logits, embed_loss
