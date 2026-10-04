"""From-scratch Custom CNN for E0; training stays in the shared pipeline."""
from __future__ import annotations

from torch import nn


class CustomCNN(nn.Module):
    """Four convolution blocks followed by global pooling and raw logits."""

    def __init__(
        self,
        num_classes: int = 30,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    padding=kernel_size // 2))
            if use_batch_norm:
                blocks.append(nn.BatchNorm2d(out_channels))
            blocks.extend((nn.ReLU(), nn.MaxPool2d(2)))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p=dropout), nn.Linear(in_channels, num_classes)
        )

    def forward(self, image):
        return self.classifier(self.avgpool(self.features(image)))


def build_model(model_config: dict) -> nn.Module:
    """Build E0 from its existing config without loading pretrained weights."""
    if model_config.get("pretrained", False):
        raise ValueError("E0 must be trained from scratch without pretrained weights.")
    if model_config.get("pooling", "maxpool") != "maxpool":
        raise ValueError("E0 requires MaxPool2d in each convolution block.")
    if model_config.get("global_pooling", "adaptive_avg") != "adaptive_avg":
        raise ValueError("E0 requires adaptive average pooling.")
    return CustomCNN(
        num_classes=int(model_config.get("num_classes", 30)),
        channels=tuple(model_config.get("channels", (32, 64, 128, 256))),
        kernel_size=int(model_config.get("kernel_size", 3)),
        use_batch_norm=bool(model_config.get("use_batch_norm", True)),
        dropout=float(model_config.get("dropout", 0.3)),
    )
