from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from backend.ml_pipeline.dataset import create_datasets
from backend.ml_pipeline.models.efficientnetv2 import create_model
from backend.ml_pipeline.preprocessing.transforms import get_val_transform


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "backend" / "ml_pipeline" / "data" / "combined_tomato"
CHECKPOINT = BASE_DIR / "backend" / "ml_pipeline" / "weights" / "efficientnetv2_best.pth"

RESULTS_DIR = BASE_DIR / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 60)
    print("XAI-PHYTOSENSE - MODEL EVALUATION")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    _, _, test_dataset = create_datasets(
        DATA_DIR,
        get_val_transform(),
        get_val_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
        num_classes=len(test_dataset.classes),
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Loaded checkpoint successfully.")
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print(
        "Best validation accuracy:",
        f"{checkpoint.get('best_val_accuracy', 0) * 100:.2f}%"
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    test_loss = total_loss / total
    test_accuracy = correct / total

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    results = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_accuracy": checkpoint.get("best_val_accuracy"),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classes": test_dataset.classes,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    output_file = RESULTS_DIR / "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to:")
    print(output_file)

    print("=" * 60)


if __name__ == "__main__":
    main()
