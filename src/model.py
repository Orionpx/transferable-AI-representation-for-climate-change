"""Prithvi segmentation model with staged backbone unfreezing."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from terratorch.datasets import HLSBands
from terratorch.registry import BACKBONE_REGISTRY


class PrithviFloodModel(nn.Module):
    """Wrap a Terratorch Prithvi backbone with a compact segmentation decoder.

    The decoder accepts the common Terratorch output forms: a tensor, a list/tuple
    of feature maps, or a dict containing feature maps. It upsamples the final
    feature map to the input spatial resolution, keeping the RTX 3060 footprint low.
    """

    def __init__(
        self,
        backbone_name: str,
        num_input_channels: int = 6,
        num_classes: int = 2,
        decoder_channels: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            bands=[
                HLSBands.BLUE,
                HLSBands.GREEN,
                HLSBands.RED,
                HLSBands.NIR_NARROW,
                HLSBands.SWIR_1,
                HLSBands.SWIR_2,
            ],
        )
        self.decoder = nn.Sequential(
            nn.LazyConv2d(decoder_channels, kernel_size=1),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def _last_feature(self, features: Any) -> torch.Tensor:
        if isinstance(features, dict):
            candidates = list(features.values())
            features = candidates[-1]
        elif isinstance(features, (list, tuple)):
            features = features[-1]
        if not isinstance(features, torch.Tensor):
            raise TypeError("Prithvi backbone must return a tensor or feature collection")
        if features.ndim == 3:
            # ViT backbones commonly return [B, tokens, channels]. Convert the
            # square patch sequence into [B, channels, height, width].
            batch_size, token_count, channel_count = features.shape
            side = math.isqrt(token_count)
            if side * side != token_count:
                side = math.isqrt(token_count - 1)
                if side * side != token_count - 1:
                    raise ValueError(f"Cannot reshape Prithvi token sequence: {features.shape}")
                features = features[:, 1:, :]
            features = features.transpose(1, 2).reshape(batch_size, channel_count, side, side)
        if features.ndim != 4:
            raise TypeError("Prithvi backbone must return a 3D or 4D tensor")
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Input is [B, 6, H, W]: B02, B03, B04, B08, B11, B12.
        features = self._last_feature(self.backbone(images))
        logits = self.decoder(features)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, target)
        probabilities = logits.softmax(dim=1)[:, 1]
        target_float = (target == 1).float()
        intersection = (probabilities * target_float).sum(dim=(1, 2))
        denominator = probabilities.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        dice_loss = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        return (1 - self.dice_weight) * cross_entropy + self.dice_weight * dice_loss
