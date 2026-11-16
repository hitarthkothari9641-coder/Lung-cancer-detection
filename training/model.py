"""
model.py - 3D Multi-Task DenseNet-121 Architecture for Nodule Detection, Grading & Size Regression.

Features:
- 3D Convolutional Backbone (MONAI DenseNet121 with pure PyTorch fallback).
- Multi-task output heads:
    1. Detection: 1 logit for nodule probability.
    2. Grading: 5 logits for CORAL ordinal malignancy classification (1..5).
    3. Size: 1 scalar for continuous nodule diameter regression in mm.
- Fully compatible with TorchScript tracing and Grad-CAM hooks.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing MONAI DenseNet121, fallback to PyTorch implementation
try:
    from monai.networks.nets import DenseNet121 as MonaiDenseNet121
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class _DenseLayer3D(nn.Module):
    def __init__(self, in_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(in_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers: int, in_features: int, bn_size: int, growth_rate: int, drop_rate: float):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer3D(in_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition3D(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.relu(self.norm(x))))


class NativeDenseNet121_3D(nn.Module):
    """Native PyTorch 3D DenseNet-121 backbone."""

    def __init__(self, in_channels: int = 1, growth_rate: int = 32, block_config: Tuple[int, ...] = (6, 12, 24, 16)):
        super().__init__()
        # Initial convolution
        self.features = nn.Sequential()
        self.features.add_module("conv0", nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module("norm0", nn.BatchNorm3d(64))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        # Dense blocks + transitions
        num_features = 64
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_layers, num_features, bn_size=4, growth_rate=growth_rate, drop_rate=0.0)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition3D(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm_final", nn.BatchNorm3d(num_features))
        self.features.add_module("relu_final", nn.ReLU(inplace=True))
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class LungNoduleMultiTaskNet(nn.Module):
    """
    3D Multi-Task Network for Lung Nodule Detection, Ordinal Grading, and Size Estimation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_grading_classes: int = 5,
        backbone_type: str = "densenet121",
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.num_grading_classes = num_grading_classes

        # Initialize backbone
        if HAS_MONAI and backbone_type == "densenet121":
            try:
                monai_dense = MonaiDenseNet121(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=512,  # temporary, we extract features
                )
                self.backbone = monai_dense.features
                self.num_features = 1024
            except Exception:
                self.backbone = NativeDenseNet121_3D(in_channels=in_channels)
                self.num_features = self.backbone.num_features
        else:
            self.backbone = NativeDenseNet121_3D(in_channels=in_channels)
            self.num_features = self.backbone.num_features

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Task 1: Detection head (1 logit for nodule presence)
        self.detection_head = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

        # Task 2: Grading head (5 logits for CORAL ordinal malignancy 1..5)
        self.grading_head = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_grading_classes),
        )

        # Task 3: Size regression head (1 scalar for continuous diameter in mm)
        self.size_head = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial convolutional feature maps before global pooling."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input CT patch tensor of shape [B, 1, D, H, W]

        Returns:
            Tuple of:
              - det_logits: [B, 1]
              - grade_logits: [B, num_grading_classes]
              - size_pred: [B, 1]
        """
        feat_map = self.extract_features(x)
        pooled = self.global_pool(feat_map)
        flat = torch.flatten(pooled, 1)

        det_logits = self.detection_head(flat)
        grade_logits = self.grading_head(flat)
        size_pred = self.size_head(flat)

        return det_logits, grade_logits, size_pred


def build_model(
    in_channels: int = 1,
    num_grading_classes: int = 5,
    backbone: str = "densenet121",
) -> LungNoduleMultiTaskNet:
    """Factory helper to construct the multi-task network."""
    return LungNoduleMultiTaskNet(
        in_channels=in_channels,
        num_grading_classes=num_grading_classes,
        backbone_type=backbone,
    )
