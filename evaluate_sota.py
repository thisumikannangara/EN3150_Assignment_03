import os

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data_pipeline import get_dataloaders
from src.sota_models import (
    get_mobilenet_v2,
    get_squeezenet
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS_DIR = "results/member4"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


# --------------------------------------------------
# Confusion matrix evaluation
# --------------------------------------------------

def create_confusion_matrix(model, test_loader, name):

    model = model.to(DEVICE)
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

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------

    plt.figure(figsize=(10, 8))

    plt.imshow(
        cm,
        cmap="viridis"
    )

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.colorbar()

    # Add numbers inside cells
    threshold = cm.max() * 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            text_color = (
                "white"
                if cm[i, j] < threshold
                else "black"
            )

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=text_color
            )

    # Add EuroSAT class names
    plt.xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        f"{name}_confusion_matrix_annotated.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"{name} confusion matrix saved to:")
    print(output_path)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Device:", DEVICE)

    # Load the SAME test set used during training
    _, _, test_loader, _ = get_dataloaders(
        augment=True
    )

    # --------------------------------------------------
    # MobileNetV2
    # --------------------------------------------------

    print("\nLoading MobileNetV2...")

    mobilenet = get_mobilenet_v2()

    mobilenet_path = os.path.join(
        RESULTS_DIR,
        "MobileNetV2.pth"
    )

    mobilenet.load_state_dict(
        torch.load(
            mobilenet_path,
            map_location=DEVICE
        )
    )

    create_confusion_matrix(
        mobilenet,
        test_loader,
        "MobileNetV2"
    )

    # --------------------------------------------------
    # SqueezeNet
    # --------------------------------------------------

    print("\nLoading SqueezeNet1.1...")

    squeezenet = get_squeezenet()

    squeezenet_path = os.path.join(
        RESULTS_DIR,
        "SqueezeNet1.1.pth"
    )

    squeezenet.load_state_dict(
        torch.load(
            squeezenet_path,
            map_location=DEVICE
        )
    )

    create_confusion_matrix(
        squeezenet,
        test_loader,
        "SqueezeNet1.1"
    )

    print("\nDone! No training was performed.")


if __name__ == "__main__":
    main()