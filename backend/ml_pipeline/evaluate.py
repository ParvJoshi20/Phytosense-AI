"""
PhytoSense AI
Experiment-aware model evaluation pipeline.

Evaluation is intentionally separate from training:
- training uses only train/validation data;
- this module evaluates the frozen best checkpoint on the test split;
- metrics and plots are written into the current experiment directory.

This module is designed to work when executed from:
    1. repository root:
       python -m backend.ml_pipeline.evaluate

    2. ML pipeline directory:
       python evaluate.py
"""

# ============================================================================
# IMPORT PATH BOOTSTRAP
# ============================================================================
# The current project contains a mixture of package-style imports
# (backend.ml_pipeline...) and local imports (from config...).
#
# We add both:
#   - repository root
#   - backend/ml_pipeline
#
# to sys.path before importing project modules.
#
# This allows evaluate.py to work without requiring changes to the existing
# dataset.py / model modules.
# ============================================================================

import sys
from pathlib import Path


# evaluate.py is:
#   <repo_root>/backend/ml_pipeline/evaluate.py
#
# Therefore:
#   parents[0] = backend/ml_pipeline
#   parents[1] = backend
#   parents[2] = repo_root

CURRENT_FILE = Path(__file__).resolve()
ML_PIPELINE_ROOT = CURRENT_FILE.parent
BACKEND_ROOT = ML_PIPELINE_ROOT.parent
REPO_ROOT = BACKEND_ROOT.parent

# Add paths only if they are not already present.
for path in (REPO_ROOT, ML_PIPELINE_ROOT):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


# ============================================================================
# STANDARD LIBRARY / THIRD-PARTY IMPORTS
# ============================================================================

import csv
import json

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


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

# config.py is located directly inside backend/ml_pipeline.
# Because ML_PIPELINE_ROOT was added to sys.path above, this works even when
# evaluate.py is launched as:
#
#     python -m backend.ml_pipeline.evaluate
#
# or:
#
#     python evaluate.py

from config import (
    ACCURACY_CURVE,
    BATCH_SIZE,
    CLASSIFICATION_REPORT_CSV,
    CLASS_NAMES,
    CONFUSION_MATRIX,
    CONFUSION_MATRIX_NORMALIZED,
    DEVICE_PREFERENCE,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
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

# These modules may internally use either package-style or local imports.
# The sys.path bootstrap above makes both styles resolvable.
from backend.ml_pipeline.dataset import create_datasets
from backend.ml_pipeline.models.efficientnetv2 import create_model


# ============================================================================
# DEVICE
# ============================================================================

def resolve_device():
    """
    Resolve the configured device without silently hiding CUDA failures.
    """
    if DEVICE_PREFERENCE == "cpu":
        return torch.device("cpu")

    if DEVICE_PREFERENCE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE_PREFERENCE='cuda' but CUDA is not available."
            )
        return torch.device("cuda")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================================
# TRAINING HISTORY
# ============================================================================

def load_training_history():
    """
    Load and normalize the epoch-by-epoch history saved by train.py.

    train.py stores history in column-wise format, for example:

        {
            "epoch": [1, 2, 3, ...],
            "train_loss": [...],
            "train_accuracy": [...],
            "train_macro_f1": [...],
            "val_loss": [...],
            "val_accuracy": [...],
            "val_macro_f1": [...],
            "learning_rate": [...]
        }

    This function converts that format into the row-wise format expected
    by the plotting functions:

        [
            {
                "epoch": 1,
                "train_loss": ...,
                ...
            },
            ...
        ]
    """

    if not TRAINING_HISTORY_JSON.exists():
        raise FileNotFoundError(
            f"Training history not found: {TRAINING_HISTORY_JSON}"
        )

    with open(
        TRAINING_HISTORY_JSON,
        "r",
        encoding="utf-8",
    ) as file:
        history = json.load(file)

    # ---------------------------------------------------------------
    # Case 1: already in row-wise list format
    # ---------------------------------------------------------------

    if isinstance(history, list):

        if not history:
            raise RuntimeError(
                "training_history.json contains an empty history."
            )

        return history

    # ---------------------------------------------------------------
    # Case 2: wrapped row-wise format
    # ---------------------------------------------------------------

    if (
        isinstance(history, dict)
        and isinstance(history.get("history"), list)
    ):

        history = history["history"]

        if not history:
            raise RuntimeError(
                "training_history.json contains an empty history."
            )

        return history

    # ---------------------------------------------------------------
    # Case 3: column-wise format produced by train.py
    # ---------------------------------------------------------------

    if isinstance(history, dict):

        epochs = history.get("epoch")

        if not isinstance(epochs, list):
            raise RuntimeError(
                "training_history.json does not contain a valid "
                "'epoch' list."
            )

        if not epochs:
            raise RuntimeError(
                "training_history.json contains an empty epoch list."
            )

        normalized_history = []

        for index, epoch in enumerate(epochs):

            record = {
                "epoch": epoch
            }

            for key, values in history.items():

                # epoch was already handled above.
                if key == "epoch":
                    continue

                if not isinstance(values, list):
                    continue

                if index < len(values):
                    record[key] = values[index]
                else:
                    record[key] = None

            normalized_history.append(record)

        return normalized_history

    # ---------------------------------------------------------------
    # Unknown format
    # ---------------------------------------------------------------

    raise RuntimeError(
        "Unsupported training_history.json format. "
        "Expected either a list of epoch records or a "
        "column-wise dictionary containing an 'epoch' list."
    )


def _history_values(history, key):
    """
    Extract one metric from the training history.
    """
    values = []

    for record in history:
        value = record.get(key)

        if value is None:
            values.append(np.nan)
        else:
            values.append(float(value))

    return values


# ============================================================================
# TRAINING CURVES
# ============================================================================

def plot_training_curves(history):
    """
    Generate accuracy and loss curves from the saved training history.
    """

    epochs = [
        record.get("epoch", index + 1)
        for index, record in enumerate(history)
    ]

    train_accuracy = _history_values(
        history,
        "train_accuracy",
    )

    val_accuracy = _history_values(
        history,
        "val_accuracy",
    )

    train_loss = _history_values(
        history,
        "train_loss",
    )

    val_loss = _history_values(
        history,
        "val_loss",
    )

    # ------------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        ACCURACY_CURVE,
        dpi=200,
    )

    plt.close()

    # ------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        LOSS_CURVE,
        dpi=200,
    )

    plt.close()


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    matrix,
    output_path,
    normalized=False,
):
    """
    Save a confusion-matrix heatmap.
    """

    matrix_to_plot = matrix.astype(float)

    if normalized:
        row_sums = matrix_to_plot.sum(
            axis=1,
            keepdims=True,
        )

        matrix_to_plot = np.divide(
            matrix_to_plot,
            row_sums,
            out=np.zeros_like(matrix_to_plot),
            where=row_sums != 0,
        )

    plt.figure(figsize=(11, 9))

    plt.imshow(
        matrix_to_plot,
        interpolation="nearest",
        cmap="Blues",
    )

    plt.title(
        "Normalized Confusion Matrix"
        if normalized
        else "Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))

    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=90,
    )

    plt.yticks(
        tick_marks,
        CLASS_NAMES,
    )

    threshold = (
        matrix_to_plot.max() / 2.0
        if matrix_to_plot.size
        else 0.0
    )

    for row in range(matrix_to_plot.shape[0]):
        for col in range(matrix_to_plot.shape[1]):

            if normalized:
                text = f"{matrix_to_plot[row, col]:.2f}"
            else:
                text = f"{int(matrix_to_plot[row, col])}"

            plt.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix_to_plot[row, col] > threshold
                    else "black"
                ),
                fontsize=7,
            )

    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

def write_classification_report_csv(report):
    """
    Write the sklearn classification report in tabular form.
    """

    rows = []

    # Per-class metrics
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

    # Aggregate metrics
    for summary_name in (
        "macro avg",
        "weighted avg",
    ):

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


# ============================================================================
# PREDICTIONS CSV
# ============================================================================

def write_predictions_csv(
    dataset,
    labels,
    predictions,
    confidences,
):
    """
    Write one prediction record per test image.
    """

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
                    (
                        row["image_id"]
                        if "image_id" in row
                        else ""
                    ),
                    row["image_path"],
                    true_index,
                    CLASS_NAMES[true_index],
                    predicted_index,
                    CLASS_NAMES[predicted_index],
                    float(confidences[index]),
                    bool(
                        true_index == predicted_index
                    ),
                ]
            )


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():

    print("=" * 70)
    print("PHYTOSENSE AI — EXPERIMENT EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = resolve_device()

    print(f"Experiment: {EXPERIMENT_ID} — {EXPERIMENT_NAME}")
    print(f"Device: {device}")
    print(f"Checkpoint: {BEST_CHECKPOINT}")

    # ------------------------------------------------------------------------
    # Validate required files
    # ------------------------------------------------------------------------

    if not BEST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_CHECKPOINT}"
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not COLOR_ROOT.exists():
        raise FileNotFoundError(
            f"Image root not found: {COLOR_ROOT}"
        )

    # ------------------------------------------------------------------------
    # Validate class configuration
    # ------------------------------------------------------------------------

    if len(CLASS_NAMES) != NUM_CLASSES:
        raise RuntimeError(
            f"Configuration mismatch: "
            f"NUM_CLASSES={NUM_CLASSES}, "
            f"but CLASS_NAMES has "
            f"{len(CLASS_NAMES)} entries."
        )

    # ------------------------------------------------------------------------
    # Create datasets
    # ------------------------------------------------------------------------

    print("\nCreating datasets...")

    train_dataset, val_dataset, test_dataset = create_datasets(
        MANIFEST_PATH,
        COLOR_ROOT,
    )

    print(f"Train samples:      {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    if len(test_dataset) == 0:
        raise RuntimeError(
            "Test dataset is empty."
        )

    # ------------------------------------------------------------------------
    # Test DataLoader
    # ------------------------------------------------------------------------

    print("\nCreating test DataLoader...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    print("\nCreating model...")

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    # ------------------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------------------

    print("Loading best checkpoint...")

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    print(
        f"Model: {MODEL_DISPLAY_NAME} "
        f"({MODEL_NAME})"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_confidences = []

    print("\nEvaluating test set...")

    with torch.no_grad():

        for batch_index, (
            images,
            labels,
        ) in enumerate(test_loader):

            images = images.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            labels = labels.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            # Forward pass
            logits = model(images)

            # Loss
            loss = criterion(
                logits,
                labels,
            )

            # Probabilities
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            # Predictions
            predictions = probabilities.argmax(
                dim=1
            )

            # Confidence
            confidences = probabilities.max(
                dim=1
            ).values

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_confidences.extend(
                confidences.cpu().tolist()
            )

            # Progress
            if (
                (batch_index + 1) % 10 == 0
                or batch_index == 0
                or (
                    batch_index + 1
                    == len(test_loader)
                )
            ):
                print(
                    f"  Batch "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)}"
                )

    # ------------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------------

    if total_samples == 0:
        raise RuntimeError(
            "No test samples were evaluated."
        )

    # ------------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------------

    test_loss = (
        total_loss
        / total_samples
    )

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # ------------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------------

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    # ------------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------------

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0,
    )

    class_accuracy = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(
            NUM_CLASSES,
            dtype=float,
        ),
        where=(
            matrix.sum(axis=1) != 0
        ),
    )

    # ------------------------------------------------------------------------
    # Cohen's Kappa
    # ------------------------------------------------------------------------

    kappa = cohen_kappa_score(
        all_labels,
        all_predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    # ------------------------------------------------------------------------
    # Normalized confusion matrix
    # ------------------------------------------------------------------------

    normalized_matrix = (
        matrix
        / np.maximum(
            matrix.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )
    )

    # ------------------------------------------------------------------------
    # Metrics object
    # ------------------------------------------------------------------------

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "dataset": "PlantVillage Tomato v1",
        "split_evaluated": TEST_SPLIT,
        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "checkpoint": str(
            BEST_CHECKPOINT
        ),
        "checkpoint_epoch": checkpoint_epoch,
        "best_validation_macro_f1": (
            checkpoint.get(
                "best_val_macro_f1"
            )
        ),
        "test_samples": int(
            total_samples
        ),
        "test_loss": float(
            test_loss
        ),
        "accuracy": float(
            test_accuracy
        ),
        "macro_precision": float(
            report["macro avg"][
                "precision"
            ]
        ),
        "macro_recall": float(
            report["macro avg"][
                "recall"
            ]
        ),
        "macro_f1": float(
            report["macro avg"][
                "f1-score"
            ]
        ),
        "weighted_precision": float(
            report["weighted avg"][
                "precision"
            ]
        ),
        "weighted_recall": float(
            report["weighted avg"][
                "recall"
            ]
        ),
        "weighted_f1": float(
            report["weighted avg"][
                "f1-score"
            ]
        ),
        "cohen_kappa": float(
            kappa
        ),
        "class_metrics": {
            CLASS_NAMES[index]: {
                "class_accuracy": float(
                    class_accuracy[index]
                ),
                "precision": float(
                    precision[index]
                ),
                "recall": float(
                    recall[index]
                ),
                "f1": float(
                    f1[index]
                ),
                "support": int(
                    support[index]
                ),
            }
            for index in range(
                NUM_CLASSES
            )
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": (
            normalized_matrix.tolist()
        ),
    }

    # ------------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------------

    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------------
    # Save classification report
    # ------------------------------------------------------------------------

    write_classification_report_csv(
        report
    )

    # ------------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------------

    write_predictions_csv(
        test_dataset,
        all_labels,
        all_predictions,
        all_confidences,
    )

    # ------------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------------

    history = load_training_history()

    plot_training_curves(
        history
    )

    # ------------------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------------------

    plot_confusion_matrix(
        matrix,
        CONFUSION_MATRIX,
        normalized=False,
    )

    plot_confusion_matrix(
        matrix,
        CONFUSION_MATRIX_NORMALIZED,
        normalized=True,
    )

    # ------------------------------------------------------------------------
    # Console results
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Test Loss:     "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{report['macro avg']['precision']:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{report['macro avg']['recall']:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{report['macro avg']['f1-score']:.4f}"
    )

    print(
        f"Weighted F1:     "
        f"{report['weighted avg']['f1-score']:.4f}"
    )

    print(
        f"Cohen's Kappa:   "
        f"{kappa:.4f}"
    )

    # ------------------------------------------------------------------------
    # Per-class results
    # ------------------------------------------------------------------------

    print("\nPER-CLASS RESULTS")
    print("-" * 70)

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{index:2d} | "
            f"{class_name:<55} | "
            f"Precision: "
            f"{precision[index]:.4f} | "
            f"Recall: "
            f"{recall[index]:.4f} | "
            f"F1: "
            f"{f1[index]:.4f} | "
            f"Support: "
            f"{support[index]}"
        )

    # ------------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------------

    print("\nARTIFACTS")
    print("-" * 70)

    artifact_paths = (
        METRICS_JSON,
        CLASSIFICATION_REPORT_CSV,
        PREDICTIONS_CSV,
        ACCURACY_CURVE,
        LOSS_CURVE,
        CONFUSION_MATRIX,
        CONFUSION_MATRIX_NORMALIZED,
    )

    for path in artifact_paths:
        print(f"  {path}")

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()