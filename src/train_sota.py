import os
import time
import csv

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from src.data_pipeline import get_dataloaders
from src.sota_models import (
    get_mobilenet_v2,
    get_squeezenet,
    count_parameters,
    count_trainable_parameters
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS_DIR = "results/member4"
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_model(model, train_loader, val_loader, name):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    val_losses = []
    epoch_times = []

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

        start_time = time.time()

        # -------------------------
        # Training
        # -------------------------

        model.train()

        running_train_loss = 0.0
        train_samples = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            running_train_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss = (
            running_train_loss / train_samples
        )

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        running_val_loss = 0.0
        val_samples = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                batch_size = images.size(0)

                running_val_loss += loss.item() * batch_size
                val_samples += batch_size

        val_loss = (
            running_val_loss / val_samples
        )

        epoch_time = time.time() - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"{name} | "
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    # Restore best model
    model.load_state_dict(best_state)

    return (
        model,
        train_losses,
        val_losses,
        epoch_times
    )


# --------------------------------------------------
# Test evaluation
# --------------------------------------------------

def evaluate_model(model, test_loader, name):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            ).cpu()

            all_predictions.extend(
                predictions.numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    # Save confusion matrix
    plt.figure(figsize=(8, 7))

    plt.imshow(cm)

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{name}_confusion_matrix.png"
        ),
        dpi=200
    )

    plt.close()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }


# --------------------------------------------------
# Loss curve
# --------------------------------------------------

def plot_loss(
    train_losses,
    val_losses,
    name
):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"{name} - Training and Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{name}_loss_curve.png"
        ),
        dpi=200
    )

    plt.close()


# --------------------------------------------------
# Model size
# --------------------------------------------------

def get_model_size_mb(model):

    path = os.path.join(
        RESULTS_DIR,
        "temporary_model.pth"
    )

    torch.save(
        model.state_dict(),
        path
    )

    size_bytes = os.path.getsize(path)

    os.remove(path)

    return size_bytes / (1024 * 1024)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Device:", DEVICE)

    # Use SAME dataset and SAME split
    # as Members 1, 2 and 3
    train_loader, val_loader, test_loader, info = \
        get_dataloaders(augment=True)

    models = {

        "MobileNetV2":
            get_mobilenet_v2(),

        "SqueezeNet1.1":
            get_squeezenet()
    }

    final_results = []

    for name, model in models.items():

        print("\n")
        print("#" * 60)
        print(f"Training {name}")
        print("#" * 60)

        total_params = count_parameters(model)

        trainable_params = count_trainable_parameters(model)

        print(
            "Total parameters:",
            total_params
        )

        print(
            "Trainable parameters:",
            trainable_params
        )

        model, train_losses, val_losses, epoch_times = \
            train_model(
                model,
                train_loader,
                val_loader,
                name
            )

        # Save trained model
        model_path = os.path.join(
            RESULTS_DIR,
            f"{name}.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        # Plot loss
        plot_loss(
            train_losses,
            val_losses,
            name
        )

        # Test evaluation
        metrics = evaluate_model(
            model,
            test_loader,
            name
        )

        model_size = get_model_size_mb(model)

        average_epoch_time = sum(epoch_times) / len(epoch_times)

        result = {

            "model": name,

            "total_parameters":
                total_params,

            "trainable_parameters":
                trainable_params,

            "model_size_MB":
                model_size,

            "average_epoch_time_seconds":
                average_epoch_time,

            "test_accuracy":
                metrics["accuracy"],

            "test_precision":
                metrics["precision"],

            "test_recall":
                metrics["recall"]
        }

        final_results.append(result)

    # --------------------------------------------------
    # Save comparison
    # --------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        "sota_comparison.csv"
    )

    fieldnames = final_results[0].keys()

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            final_results
        )

    print("\nSOTA comparison saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()