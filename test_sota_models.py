import torch

from src.sota_models import (
    get_mobilenet_v2,
    get_squeezenet,
    count_parameters,
    count_trainable_parameters
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

models = {
    "MobileNetV2": get_mobilenet_v2(),
    "SqueezeNet1.1": get_squeezenet()
}


for name, model in models.items():

    model = model.to(device)

    x = torch.randn(4, 3, 64, 64).to(device)

    with torch.no_grad():
        output = model(x)

    print("\n", name)
    print("Output shape:", output.shape)
    print("Total parameters:", count_parameters(model))
    print("Trainable parameters:", count_trainable_parameters(model))
