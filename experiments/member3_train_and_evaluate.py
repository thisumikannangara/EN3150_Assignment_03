
"""
Member 03 - Optimizer Selection, Training and Evaluation

Uses the shared:
    src/data_pipeline.py
    src/models/model_a.py
    src/models/model_b.py

Experiments:
    1. SGD
    2. SGD with Momentum
    3. Adam (selected optimizer)

Both custom CNN models are trained for 20 epochs.

Outputs:
    - training/validation loss curves
    - test accuracy
    - macro precision
    - macro recall
    - confusion matrices
    - training time per epoch
    - comparison CSV
"""

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
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.data_pipeline import get_dataloaders
from src.models.model_a import StandardCNN
from src.models.model_b import LightweightCNN


# ============================================================
# SETTINGS
# ============================================================

EPOCHS = 20

# Learning rates selected for comparison
LR_SGD = 0.01
LR_MOMENTUM = 0.01
LR_ADAM = 0.001

MOMENTUM = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "member3"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_time = time.perf_counter() - start_time

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy, epoch_time


# ============================================================
# VALIDATION
# ============================================================

def evaluate_loss(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(model, loader):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_predictions)

    return (
        accuracy,
        macro_precision,
        macro_recall,
        cm,
    )


# ============================================================
# OPTIMIZER CREATION
# ============================================================

def create_optimizer(name, model):
    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=LR_SGD
        )

    elif name == "SGD_Momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=LR_MOMENTUM,
            momentum=MOMENTUM
        )

    elif name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=LR_ADAM
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# MODEL EXPERIMENT
# ============================================================

def run_experiment(model_name, model_class, optimizer_name, train_loader,
                   val_loader, test_loader, class_names):

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print(f"OPTIMIZER: {optimizer_name}")
    print("=" * 70)

    model = model_class(num_classes=len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(optimizer_name, model)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = evaluate_loss(
            model,
            val_loader,
            criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        epoch_times.append(epoch_time)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    (
        test_accuracy,
        macro_precision,
        macro_recall,
        cm
    ) = evaluate_test(model, test_loader)

    average_epoch_time = sum(epoch_times) / len(epoch_times)
    total_training_time = sum(epoch_times)

    print("\nTest results:")
    print(f"Accuracy       : {test_accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"Avg epoch time : {average_epoch_time:.2f} seconds")
    print(f"Total time     : {total_training_time:.2f} seconds")

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        range(1, EPOCHS + 1),
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - {optimizer_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_{optimizer_name}_loss_curve.png"
    )

    plt.savefig(loss_path, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(9, 9))

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )

    ax.set_title(
        f"{model_name} - {optimizer_name} Confusion Matrix"
    )

    plt.tight_layout()

    cm_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_{optimizer_name}_confusion_matrix.png"
    )

    plt.savefig(cm_path, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Save epoch timing
    # --------------------------------------------------------

    timing_path = os.path.join(
        RESULTS_DIR,
        f"{model_name}_{optimizer_name}_epoch_times.csv"
    )

    with open(timing_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["epoch", "time_seconds"]
        )

        for i, t in enumerate(epoch_times, start=1):
            writer.writerow([i, t])

    return {
        "model": model_name,
        "optimizer": optimizer_name,
        "learning_rate": (
            LR_SGD
            if optimizer_name == "SGD"
            else LR_MOMENTUM
            if optimizer_name == "SGD_Momentum"
            else LR_ADAM
        ),
        "momentum": (
            MOMENTUM
            if optimizer_name == "SGD_Momentum"
            else 0.0
        ),
        "test_accuracy": test_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "average_epoch_time": average_epoch_time,
        "total_training_time": total_training_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEMBER 03 - TRAINING AND EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")

    # --------------------------------------------------------
    # Load shared data pipeline
    # --------------------------------------------------------

    print("\nLoading shared dataset pipeline...")

    train_loader, val_loader, test_loader, info = get_dataloaders(
        augment=True
    )

    class_names = info["classes"]

    print("\nDataset information:")
    print(f"Classes      : {class_names}")
    print(f"Train size   : {info['sizes']['train']}")
    print(f"Validation   : {info['sizes']['val']}")
    print(f"Test size    : {info['sizes']['test']}")

    print(f"\nTrain mean: {info['mean']}")
    print(f"Train std : {info['std']}")

    # --------------------------------------------------------
    # Experiment 1: SGD
    # --------------------------------------------------------

    results = []

    results.append(
        run_experiment(
            "ModelA",
            StandardCNN,
            "SGD",
            train_loader,
            val_loader,
            test_loader,
            class_names
        )
    )

    results.append(
        run_experiment(
            "ModelB",
            LightweightCNN,
            "SGD",
            train_loader,
            val_loader,
            test_loader,
            class_names
        )
    )

    # --------------------------------------------------------
    # Experiment 2: SGD with Momentum
    # --------------------------------------------------------

    results.append(
        run_experiment(
            "ModelA",
            StandardCNN,
            "SGD_Momentum",
            train_loader,
            val_loader,
            test_loader,
            class_names
        )
    )

    results.append(
        run_experiment(
            "ModelB",
            LightweightCNN,
            "SGD_Momentum",
            train_loader,
            val_loader,
            test_loader,
            class_names
        )
    )

    # --------------------------------------------------------
    # Experiment 3: Adam
    # --------------------------------------------------------

    results.append(
        run_experiment(
            "ModelA",
            StandardCNN,
            "Adam",
            train_loader,
            val_loader,
            test_loader,
            class_names
        )
    )

    results.append(
        run_experiment(
            "ModelB",
            LightweightCNN,
            "Adam",
            train_loader,
            val_loader,
            test_loader,
            class_names
        )
    )

    # --------------------------------------------------------
    # Save comparison table
    # --------------------------------------------------------

    comparison_path = os.path.join(
        RESULTS_DIR,
        "model_optimizer_comparison.csv"
    )

    with open(comparison_path, "w", newline="") as f:

        fieldnames = [
            "model",
            "optimizer",
            "learning_rate",
            "momentum",
            "test_accuracy",
            "macro_precision",
            "macro_recall",
            "average_epoch_time",
            "total_training_time",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    # --------------------------------------------------------
    # Print final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)

    for result in results:

        print(
            f"{result['model']:8s} | "
            f"{result['optimizer']:14s} | "
            f"Acc: {result['test_accuracy']:.4f} | "
            f"Precision: {result['macro_precision']:.4f} | "
            f"Recall: {result['macro_recall']:.4f} | "
            f"Epoch: {result['average_epoch_time']:.2f}s"
        )

    print("\nResults saved to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()

