from __future__ import annotations

from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
from torch import nn


def build_lightweight(num_classes: int, pretrained: bool = True):
    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_expert(num_classes: int, pretrained: bool = True):
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
