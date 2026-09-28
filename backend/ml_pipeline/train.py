"""
PhytoSense AI
Model Training Pipeline.

EXP-001 — EfficientNetV2-B0 Baseline

Training flow:

    Manifest
        ↓
    Dataset
        ↓
    Preprocessing
        ↓
    DataLoader
        ↓
    EfficientNetV2-B0
        ↓
    Weighted Cross-Entropy Loss
        ↓
    Validation
        ↓
    Validation Macro F1
        ↓
    Checkpoint / Early Stopping

The test set is NOT used during training.
It is reserved for final evaluation.
"""

from __future__ import annotations

import json
import logging
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DATASET_VERSION,
    MANIFEST_PATH,
    COLOR_ROOT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_ROOT,
    CHECKPOINT_DIR,
    MODEL_DIR,
    HISTORY_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    LOG_DIR,
    CONFIG_JSON,
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    TRAINING_HISTORY_JSON,
    TRAINING_LOG,
    MODEL_NAME,
    MODEL_DISPLAY_NAME,
    PRETRAINED,
    NUM_CLASSES,
    CLASS_NAMES,
    RANDOM_SEED,
    IMAGE_SIZE,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_ENABLED,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    REGULARIZATION_LAMBDA,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCHEDULER_MODE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
    LOSS_NAME,
    CLASS_WEIGHT_METHOD,
    MODEL_SELECTION_METRIC,
    ensure_experiment_directories,
)

from dataset.dataset import (
    create_datasets,
    create_dataloaders,
    get_class_weights,
)

from models.efficientnetv2 import (
    create_model,
    get_num_parameters,
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif (
    getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
):
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


NUM_WORKERS = 0


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> logging.Logger:
    """
    Configure console + experiment-file logging.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("phytosense.training")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if main() is called repeatedly.
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        TRAINING_LOG,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """
    Configure random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic behaviour for the baseline experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Seed DataLoader workers deterministically.
    """

    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# METRICS
# ============================================================

def calculate_macro_f1(
    predictions,
    targets,
    num_classes: int,
) -> float:
    """
    Calculate macro F1 across all classes.

    Each class contributes equally to the final score.
    """

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    f1_scores = []

    for class_index in range(num_classes):

        true_positive = np.sum(
            (predictions == class_index)
            & (targets == class_index)
        )

        false_positive = np.sum(
            (predictions == class_index)
            & (targets != class_index)
        )

        false_negative = np.sum(
            (predictions != class_index)
            & (targets == class_index)
        )

        precision_denominator = (
            true_positive + false_positive
        )

        recall_denominator = (
            true_positive + false_negative
        )

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one complete epoch.
    """

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    all_predictions = np.asarray(
        all_predictions
    )

    all_targets = np.asarray(
        all_targets
    )

    accuracy = float(
        np.mean(
            all_predictions == all_targets
        )
    )

    macro_f1 = calculate_macro_f1(
        all_predictions,
        all_targets,
        NUM_CLASSES,
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate the model on the validation set.

    Validation loss is intentionally unweighted.
    Class weighting affects optimization, not evaluation.
    """

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    all_predictions = np.asarray(
        all_predictions
    )

    all_targets = np.asarray(
        all_targets
    )

    accuracy = float(
        np.mean(
            all_predictions == all_targets
        )
    )

    macro_f1 = calculate_macro_f1(
        all_predictions,
        all_targets,
        NUM_CLASSES,
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_macro_f1: float,
    history: dict,
    class_weights,
) -> None:
    """
    Save complete training state.

    Native PyTorch checkpoint is retained as the authoritative
    training/resume artifact for this experiment.
    """

    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,

        "epoch": epoch,

        "model_name": MODEL_NAME,
        "model_display_name": MODEL_DISPLAY_NAME,
        "pretrained": PRETRAINED,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_macro_f1":
            best_val_macro_f1,

        "class_weights":
            class_weights.detach()
            .cpu()
            .tolist(),

        "history":
            history,

        "training_config": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_enabled":
                EARLY_STOPPING_ENABLED,
            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "regularization_lambda":
                REGULARIZATION_LAMBDA,
            "optimizer":
                OPTIMIZER_NAME,
            "scheduler":
                SCHEDULER_NAME,
            "scheduler_mode":
                SCHEDULER_MODE,
            "scheduler_factor":
                SCHEDULER_FACTOR,
            "scheduler_patience":
                SCHEDULER_PATIENCE,
            "loss":
                LOSS_NAME,
            "class_weight_method":
                CLASS_WEIGHT_METHOD,
            "model_selection_metric":
                MODEL_SELECTION_METRIC,
            "random_seed":
                RANDOM_SEED,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# CONFIGURATION RECORD
# ============================================================

def save_experiment_config(
    logger: logging.Logger,
) -> None:
    """
    Save a JSON snapshot of the important experiment
    configuration and runtime environment.
    """

    config = {
        "experiment": {
            "id": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "dataset_version": DATASET_VERSION,
        },

        "dataset": {
            "manifest_path":
                str(MANIFEST_PATH),
            "image_root":
                str(COLOR_ROOT),
            "num_classes":
                NUM_CLASSES,
            "class_names":
                CLASS_NAMES,
            "split": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
        },

        "model": {
            "name":
                MODEL_NAME,
            "display_name":
                MODEL_DISPLAY_NAME,
            "pretrained":
                PRETRAINED,
            "input_size":
                IMAGE_SIZE,
        },

        "training": {
            "batch_size":
                BATCH_SIZE,
            "max_epochs":
                MAX_EPOCHS,
            "early_stopping_enabled":
                EARLY_STOPPING_ENABLED,
            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,
            "learning_rate":
                LEARNING_RATE,
            "optimizer":
                OPTIMIZER_NAME,
            "weight_decay":
                WEIGHT_DECAY,
            "regularization_lambda":
                REGULARIZATION_LAMBDA,
            "loss":
                LOSS_NAME,
            "class_weight_method":
                CLASS_WEIGHT_METHOD,
            "model_selection_metric":
                MODEL_SELECTION_METRIC,
            "random_seed":
                RANDOM_SEED,
        },

        "scheduler": {
            "name":
                SCHEDULER_NAME,
            "mode":
                SCHEDULER_MODE,
            "factor":
                SCHEDULER_FACTOR,
            "patience":
                SCHEDULER_PATIENCE,
            "min_lr":
                SCHEDULER_MIN_LR,
        },

        "runtime": {
            "device":
                str(DEVICE),
            "torch_version":
                torch.__version__,
            "numpy_version":
                np.__version__,
            "python_version":
                platform.python_version(),
            "platform":
                platform.platform(),
            "cuda_available":
                torch.cuda.is_available(),
            "cuda_version":
                torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "started_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        },
    }

    with open(
        CONFIG_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    logger.info(
        "Experiment configuration saved: %s",
        CONFIG_JSON,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_experiment_directories()

    logger = configure_logging()

    logger.info("=" * 70)
    logger.info(
        "PHYTOSENSE AI — MODEL TRAINING"
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(RANDOM_SEED)

    logger.info(
        "Random seed: %d",
        RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Configuration snapshot
    # --------------------------------------------------------

    save_experiment_config(logger)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    logger.info("DEVICE")
    logger.info("-" * 70)

    logger.info(
        "Using device: %s",
        DEVICE,
    )

    if DEVICE.type == "cuda":
        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    logger.info("DATASET")
    logger.info("-" * 70)

    logger.info(
        "Dataset version: %s",
        DATASET_VERSION,
    )

    logger.info(
        "Manifest: %s",
        MANIFEST_PATH,
    )

    logger.info(
        "Image root: %s",
        COLOR_ROOT,
    )

    logger.info(
        "Classes: %d",
        NUM_CLASSES,
    )

    logger.info(
        "Project split: 70 / 15 / 15",
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    logger.info("TRAINING CONFIGURATION")
    logger.info("-" * 70)

    logger.info(
        "Experiment: %s — %s",
        EXPERIMENT_ID,
        EXPERIMENT_NAME,
    )

    logger.info(
        "Model: %s",
        MODEL_DISPLAY_NAME,
    )

    logger.info(
        "Model identifier: %s",
        MODEL_NAME,
    )

    logger.info(
        "Pretrained ImageNet: %s",
        PRETRAINED,
    )

    logger.info(
        "Input size: %d x %d",
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    logger.info(
        "Batch size: %d",
        BATCH_SIZE,
    )

    logger.info(
        "Maximum epochs: %d",
        MAX_EPOCHS,
    )

    logger.info(
        "Early stopping: %s",
        EARLY_STOPPING_ENABLED,
    )

    logger.info(
        "Early stopping patience: %d",
        EARLY_STOPPING_PATIENCE,
    )

    logger.info(
        "Learning rate: %.6g",
        LEARNING_RATE,
    )

    logger.info(
        "Optimizer: %s",
        OPTIMIZER_NAME,
    )

    logger.info(
        "Weight decay: %.6g",
        WEIGHT_DECAY,
    )

    logger.info(
        "Regularization lambda: %.6g "
        "(AdamW weight decay)",
        REGULARIZATION_LAMBDA,
    )

    logger.info(
        "Loss: %s",
        LOSS_NAME,
    )

    logger.info(
        "Class weighting: %s",
        CLASS_WEIGHT_METHOD,
    )

    logger.info(
        "Model selection metric: %s",
        MODEL_SELECTION_METRIC,
    )

    # --------------------------------------------------------
    # Dataset creation
    # --------------------------------------------------------

    logger.info("CREATING DATASETS")
    logger.info("-" * 70)

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets(
        manifest_path=MANIFEST_PATH,
        image_root=COLOR_ROOT,
    )

    logger.info(
        "Train samples: %d",
        len(train_dataset),
    )

    logger.info(
        "Validation samples: %d",
        len(val_dataset),
    )

    logger.info(
        "Test samples: %d",
        len(test_dataset),
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    logger.info("CREATING DATALOADERS")
    logger.info("-" * 70)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    logger.info(
        "Train batches: %d",
        len(train_loader),
    )

    logger.info(
        "Validation batches: %d",
        len(val_loader),
    )

    logger.info(
        "Test batches: %d",
        len(test_loader),
    )

    # Prevent accidental test usage later in this function.
    del test_loader

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    logger.info("CLASS WEIGHTS")
    logger.info("-" * 70)

    class_weights = get_class_weights(
        train_dataset
    ).to(DEVICE)

    for index, weight in enumerate(
        class_weights
    ):
        logger.info(
            "%2d | %-55s | %.6f",
            index,
            CLASS_NAMES[index],
            weight.item(),
        )

    logger.info(
        "Mean class weight: %.6f",
        class_weights.mean().item(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    logger.info("MODEL")
    logger.info("-" * 70)

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
    )

    model = model.to(DEVICE)

    trainable_parameters = get_num_parameters(
        model
    )

    logger.info(
        "Model created successfully.",
    )

    logger.info(
        "Trainable parameters: %d",
        trainable_parameters,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    val_criterion = nn.CrossEntropyLoss()

    logger.info(
        "Training loss: weighted CrossEntropyLoss",
    )

    logger.info(
        "Validation loss: unweighted CrossEntropyLoss",
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )

    logger.info(
        "Scheduler: %s",
        SCHEDULER_NAME,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "epoch": [],

        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],

        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],

        "learning_rate": [],

        "best_val_macro_f1": [],
        "epochs_without_improvement": [],
    }

    best_val_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
            train_macro_f1,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
        ) = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=val_criterion,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_macro_f1
        )

        # ----------------------------------------------------
        # Best model determination
        # ----------------------------------------------------

        is_best = (
            val_macro_f1
            > best_val_macro_f1
        )

        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history["epoch"].append(epoch)

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["train_macro_f1"].append(
            train_macro_f1
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_macro_f1"].append(
            val_macro_f1
        )

        history["learning_rate"].append(
            current_lr
        )

        history["best_val_macro_f1"].append(
            best_val_macro_f1
        )

        history[
            "epochs_without_improvement"
        ].append(
            epochs_without_improvement
        )

        # ----------------------------------------------------
        # Epoch logging
        # ----------------------------------------------------

        logger.info(
            "Epoch %03d/%03d | "
            "Train Loss %.4f | "
            "Train Acc %.4f | "
            "Train Macro F1 %.4f | "
            "Val Loss %.4f | "
            "Val Acc %.4f | "
            "Val Macro F1 %.4f | "
            "LR %.6g",
            epoch,
            MAX_EPOCHS,
            train_loss,
            train_accuracy,
            train_macro_f1,
            val_loss,
            val_accuracy,
            val_macro_f1,
            current_lr,
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if is_best:

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                history=history,
                class_weights=class_weights,
            )

            logger.info(
                "New best model saved: %s",
                BEST_CHECKPOINT,
            )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_macro_f1=best_val_macro_f1,
            history=history,
            class_weights=class_weights,
        )

        # ----------------------------------------------------
        # Save history after every epoch
        # ----------------------------------------------------

        with open(
            TRAINING_HISTORY_JSON,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            EARLY_STOPPING_ENABLED
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            logger.info(
                "Early stopping triggered at epoch %d.",
                epoch,
            )

            logger.info(
                "No validation Macro F1 improvement "
                "for %d epochs.",
                EARLY_STOPPING_PATIENCE,
            )

            break

    # --------------------------------------------------------
    # Final history metadata
    # --------------------------------------------------------

    history["summary"] = {
        "best_epoch": best_epoch,
        "best_val_macro_f1":
            best_val_macro_f1,
        "epochs_completed":
            len(history["epoch"]),
        "early_stopping_enabled":
            EARLY_STOPPING_ENABLED,
        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,
    }

    with open(
        TRAINING_HISTORY_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    logger.info(
        "Epochs completed: %d",
        len(history["epoch"]),
    )

    logger.info(
        "Best epoch: %d",
        best_epoch,
    )

    logger.info(
        "Best validation Macro F1: %.6f",
        best_val_macro_f1,
    )

    logger.info(
        "Best checkpoint: %s",
        BEST_CHECKPOINT,
    )

    logger.info(
        "Last checkpoint: %s",
        LAST_CHECKPOINT,
    )

    logger.info(
        "Training history: %s",
        TRAINING_HISTORY_JSON,
    )

    logger.info(
        "Experiment configuration: %s",
        CONFIG_JSON,
    )

    logger.info(
        "Training log: %s",
        TRAINING_LOG,
    )

    logger.info(
        "Test set was not used during training.",
    )

    logger.info(
        "Final test evaluation must be performed "
        "separately by evaluate.py.",
    )

    logger.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()