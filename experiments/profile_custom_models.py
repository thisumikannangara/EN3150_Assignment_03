import io
import torch

from src.models import StandardCNN, LightweightCNN


def profile_model(model_name, model):
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    theoretical_size_kb = trainable_parameters * 4 / 1024

    memory_file = io.BytesIO()
    torch.save(model.state_dict(), memory_file)
    serialized_size_kb = memory_file.getbuffer().nbytes / 1024

    print(model_name)
    print(f"  Trainable parameters : {trainable_parameters:,}")
    print(f"  FP32 weight memory   : {theoretical_size_kb:.2f} KB")
    print(f"  Saved state size     : {serialized_size_kb:.2f} KB")
    print()


def main():
    profile_model("Model A - StandardCNN", StandardCNN())
    profile_model("Model B - LightweightCNN", LightweightCNN())


if __name__ == "__main__":
    main()