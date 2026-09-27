import torch
from torch import nn
from torch.optim import Adam

from src.data_pipeline import get_dataloaders
from src.models import StandardCNN, LightweightCNN


MAX_BATCHES = 3
EPOCHS = 2


def smoke_train(model, model_name, train_loader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        processed_batches = 0

        for batch_number, (images, labels) in enumerate(train_loader):
            if batch_number >= MAX_BATCHES:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1

        average_loss = running_loss / processed_batches
        print(
            f"{model_name} - epoch {epoch + 1}/{EPOCHS} "
            f"- smoke-test loss: {average_loss:.4f}"
        )

    print(f"{model_name} smoke test passed.\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, _, _, info = get_dataloaders(
        batch_size=16,
        augment=False
    )

    print("Classes:", info["num_classes"])
    print("Training images:", info["sizes"]["train"])

    smoke_train(StandardCNN(), "Model A", train_loader, device)
    smoke_train(LightweightCNN(), "Model B", train_loader, device)


if __name__ == "__main__":
    main()