import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution:

    1. Depthwise 3x3 convolution processes each input channel separately.
    2. Pointwise 1x1 convolution combines the channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.depthwise(x))
        x = self.relu(self.pointwise(x))
        return x


class LightweightCNN(nn.Module):
    """
    Model B: Lightweight CNN for 64x64 RGB EuroSAT images.

    Input shape : (batch_size, 3, 64, 64)
    Output shape: (batch_size, 10)
    Expected trainable parameters: 11,080
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # 3 x 64 x 64 -> 16 x 64 x 64
            DepthwiseSeparableConv(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 16 x 32 x 32 -> 32 x 32 x 32
            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 32 x 16 x 16 -> 64 x 16 x 16
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 64 x 8 x 8 -> 96 x 8 x 8
            DepthwiseSeparableConv(64, 96),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x