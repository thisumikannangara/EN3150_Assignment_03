import torch
from torch import nn


class StandardCNN(nn.Module):
    """
    Model A: Standard CNN for 64x64 RGB EuroSAT images.

    Input shape : (batch_size, 3, 64, 64)
    Output shape: (batch_size, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # 3 x 64 x 64 -> 32 x 64 x 64
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            # 32 x 64 x 64 -> 32 x 32 x 32
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 32 x 32 x 32 -> 64 x 32 x 32
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            # 64 x 32 x 32 -> 64 x 16 x 16
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 64 x 16 x 16 -> 128 x 16 x 16
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            # 128 x 16 x 16 -> 128 x 8 x 8
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 128 x 8 x 8 -> 128 x 1 x 1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x