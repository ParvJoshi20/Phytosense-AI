"""
PhytoSense AI
Experiment-aware model evaluation pipeline.

Evaluation is intentionally separate from training:
- training uses only train/validation data;
- this module evaluates the frozen best checkpoint on the test split;
- metrics and plots are written into the current experiment directory.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from backend.ml_pipeline.config import (
    ACCURACY_CURVE,
    BATCH_SIZE,
    CLASSIFICATION_REPORT_CSV,
    CLASS_NAMES,
    CONFUSION_MATRIX,
    CONFUSION_MATRIX_NORMALIZED,
    DEVICE_PREFERENCE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    HISTORY_DIR,
    LOSS_CURVE,
    MANIFEST_PATH,
    METRICS_JSON,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    NUM_CLASSES,
    PIN_MEMORY,
    PREDICTIONS_CSV,
    TEST_SPLIT,
    TRAINING_HISTORY_JSON,
    BEST_CHECKPOINT,
    COLOR_ROOT,
)
from backend.ml_pipeline.dataset import create_datasets
from backend.ml_pipeline.models.efficientnetv2 import create_model


def resolve_device():
    """Resolve the configured device without silently hiding CUDA failures."""
    if DEVICE_PREFERENCE == "cpu":
        return torch.device("cpu")

    if DEVICE_PREFERENCE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE_PREFERENCE='cuda' but CUDA is not available."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_training_history():
    """Load the epoch-by-epoch history saved by train.py."""
    if not TRAINING_HISTORY_JSON.exists():
        raise FileNotFoundError(
            f"Training history not found: {TRAINING_HISTORY_JSON}"
        )

    with open(TRAINING_HISTORY_JSON, "r", encoding="utf-8") as file:
        history = json.load(file)

    if isinstance(history, dict) and "history" in history:
        history = history["history"]

    if not isinstance(history, list):
        raise RuntimeError(
            "training_history.json must contain a list of epoch records "
            "or a dictionary with a 'history' list."
        )

    return history


def _history_values(history, key):
    values = []
    for record in history:
        value = record.get(key)
        values.append(np.nan if value is None else float(value))
    return values


def plot_training_curves(history):
    """Generate accuracy and loss curves from the saved training history."""
    epochs = [record.get("epoch", index + 1) for index, record in enumerate(history)]

    train_accuracy = _history_values(history, "train_accuracy")
    val_accuracy = _history_values(history, "val_accuracy")
    train_loss = _history_values(history, "train_loss")
    val_loss = _history_values(history, "val_loss")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACCURACY_CURVE, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE, dpi=200)
    plt.close()


def plot_confusion_matrix(matrix, output_path, normalized=False):
    """Save a confusion-matrix heatmap."""
    matrix_to_plot = matrix.astype(float)

    if normalized:
        row_sums = matrix_to_plot.sum(axis=1, keepdims=True)
        matrix_to_plot = np.divide(
            matrix_to_plot,
            row_sums,
            out=np.zeros_like(matrix_to_plot),
            where=row_sums != 0,
        )

    plt.figure(figsize=(11, 9))
    plt.imshow(matrix_to_plot, interpolation="nearest", cmap="Blues")
    plt.title(
        "Normalized Confusion Matrix"
        if normalized
        else "Confusion Matrix"
    )
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=90)
    plt.yticks(tick_marks, CLASS_NAMES)

    threshold = matrix_to_plot.max() / 2.0 if matrix_to_plot.size else 0.0

    for row in range(matrix_to_plot.shape[0]):
        for col in range(matrix_to_plot.shape[1]):
            text = (
                f"{matrix_to_plot[row, col]:.2f}"
                if normalized
                else f"{int(matrix_to_plot[row, col])}"
            )
            plt.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if matrix_to_plot[row, col] > threshold else "black",
                fontsize=7,
            )

    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def write_classification_report_csv(report):
    """Write the sklearn classification report in tabular form."""
    rows = []

    for class_name in CLASS_NAMES:
        values = report[class_name]
        rows.append(
            {
                "class_name": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
                "class_accuracy": values["recall"],
            }
        )

    for summary_name in ("macro avg", "weighted avg"):
        values = report[summary_name]
        rows.append(
            {
                "class_name": summary_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
                "class_accuracy": "",
            }
        )

    with open(
        CLASSIFICATION_REPORT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
                "class_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_predictions_csv(
    dataset,
    labels,
    predictions,
    confidences,
):
    """Write one prediction record per test image."""
    dataframe = dataset.dataframe.reset_index(drop=True)

    with open(
        PREDICTIONS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id",
                "image_path",
                "true_class_index",
                "true_class_name",
                "predicted_class_index",
                "predicted_class_name",
                "confidence",
                "correct",
            ]
        )

        for index, row in dataframe.iterrows():
            true_index = int(labels[index])
            predicted_index = int(predictions[index])

            writer.writerow(
                [
                    row["image_id"] if "image_id" in row else "",
                    row["image_path"],
                    true_index,
                    CLASS_NAMES[true_index],
                    predicted_index,
                    CLASS_NAMES[predicted_index],
                    float(confidences[index]),
                    bool(true_index == predicted_index),
                ]
            )


def main():
    print("=" * 70)
    print("PHYTOSENSE AI — EXPERIMENT EVALUATION")
    print("=" * 70)

    device = resolve_device()

    print(f"Experiment: {EXPERIMENT_ID} — {EXPERIMENT_NAME}")
    print(f"Device: {device}")
    print(f"Checkpoint: {BEST_CHECKPOINT}")

    if not BEST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_CHECKPOINT}"
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    train_dataset, val_dataset, test_dataset = create_datasets(
        MANIFEST_PATH,
        COLOR_ROOT,
    )

    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty.")

    if len(CLASS_NAMES) != NUM_CLASSES:
        raise RuntimeError(
            f"Configuration mismatch: NUM_CLASSES={NUM_CLASSES}, "
            f"but CLASS_NAMES has {len(CLASS_NAMES)} entries."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model: {MODEL_DISPLAY_NAME} ({MODEL_NAME})")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("No test samples were evaluated.")

    test_loss = total_loss / total_samples
    test_accuracy = accuracy_score(all_labels, all_predictions)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    class_accuracy = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=matrix.sum(axis=1) != 0,
    )

    kappa = cohen_kappa_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "dataset": "PlantVillage Tomato v1",
        "split_evaluated": TEST_SPLIT,
        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "checkpoint": str(BEST_CHECKPOINT),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_macro_f1": checkpoint.get("best_val_macro_f1"),
        "test_loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "cohen_kappa": float(kappa),
        "class_metrics": {
            CLASS_NAMES[index]: {
                "class_accuracy": float(class_accuracy[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(NUM_CLASSES)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": (
            matrix
            / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ).tolist(),
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    write_classification_report_csv(report)
    write_predictions_csv(
        test_dataset,
        all_labels,
        all_predictions,
        all_confidences,
    )

    history = load_training_history()
    plot_training_curves(history)
    plot_confusion_matrix(matrix, CONFUSION_MATRIX, normalized=False)
    plot_confusion_matrix(
        matrix,
        CONFUSION_MATRIX_NORMALIZED,
        normalized=True,
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Macro F1:      {report['macro avg']['f1-score']:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    print("\nArtifacts:")
    for path in (
        METRICS_JSON,
        CLASSIFICATION_REPORT_CSV,
        ACCURACY_CURVE,
        LOSS_CURVE,
        CONFUSION_MATRIX,
        CONFUSION_MATRIX_NORMALIZED,
    ):
        print(f"  {path}")

    print("=" * 70)


if __name__ == "__main__":
    main()