import torch
from torch import nn
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
    squeezenet1_1,
    SqueezeNet1_1_Weights
)


NUM_CLASSES = 10


def get_mobilenet_v2(num_classes=NUM_CLASSES):
    """
    Load pretrained MobileNetV2 and adapt it for EuroSAT.
    """

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Replace ImageNet classifier (1000 classes)
    # with EuroSAT classifier (10 classes)
    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model


def get_squeezenet(num_classes=NUM_CLASSES):
    """
    Load pretrained SqueezeNet 1.1 and adapt it for EuroSAT.
    """

    weights = SqueezeNet1_1_Weights.DEFAULT
    model = squeezenet1_1(weights=weights)

    # Replace ImageNet classifier with 10-class classifier
    model.classifier[1] = nn.Conv2d(
        512,
        num_classes,
        kernel_size=1
    )

    model.num_classes = num_classes

    return model


def count_parameters(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """Count trainable parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
