"""
PhytoSense AI
Model Training Pipeline.

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
    Macro F1
        ↓
    Checkpoint / Early Stopping

The test set is NOT used during training.
It is reserved for final evaluation.
"""

from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    ML_PIPELINE_ROOT,
    TOMATO_MANIFEST,
    COLOR_ROOT,
    NUM_CLASSES,
    TOMATO_CLASSES,
    RANDOM_SEED,
)

from dataset.dataset import (
    create_datasets,
    create_dataloaders,
    get_class_weights,
)

from models.efficientnetv2 import (
    create_model,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

MAX_EPOCHS = 150

EARLY_STOPPING_PATIENCE = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECKPOINT / OUTPUT PATHS
# ============================================================

WEIGHTS_DIR = ML_PIPELINE_ROOT / "weights"

BEST_CHECKPOINT_PATH = (
    WEIGHTS_DIR / "efficientnetv2_best.pth"
)

LAST_CHECKPOINT_PATH = (
    WEIGHTS_DIR / "efficientnetv2_last.pth"
)

HISTORY_PATH = (
    WEIGHTS_DIR / "training_history.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Configure random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour is preferred for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MACRO F1
# ============================================================

def calculate_macro_f1(
    predictions,
    targets,
    num_classes,
):
    """
    Calculate macro F1 without requiring an additional
    sklearn dependency.

    F1 is calculated independently for every class and
    then averaged equally across all classes.
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

        if precision_denominator == 0:
            precision = 0.0
        else:
            precision = (
                true_positive
                / precision_denominator
            )

        if recall_denominator == 0:
            recall = 0.0
        else:
            recall = (
                true_positive
                / recall_denominator
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
# TRAINING EPOCH
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

        predictions = (
            outputs.argmax(dim=1)
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
            all_predictions
            == all_targets
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
# VALIDATION EPOCH
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

    Validation uses an UNWEIGHTED loss because class weights
    are intended to influence optimization, not evaluation.
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

        predictions = (
            outputs.argmax(dim=1)
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
            all_predictions
            == all_targets
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
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_macro_f1,
    history,
):
    """
    Save all important training state required for
    reproducibility and future resumption.
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_macro_f1":
            best_val_macro_f1,

        "class_names":
            TOMATO_CLASSES,

        "num_classes":
            NUM_CLASSES,

        "history":
            history,

        "training_config": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "random_seed":
                RANDOM_SEED,
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHYTOSENSE AI — MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print("\nDEVICE")
    print("-" * 70)

    print(f"Using device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Manifest: {TOMATO_MANIFEST}"
    )

    print(
        f"Image root: {COLOR_ROOT}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print(
        f"Train / Val / Test: "
        f"70 / 15 / 15"
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print("\nTRAINING CONFIGURATION")
    print("-" * 70)

    print(
        "Model: EfficientNetV2-B0"
    )

    print(
        "Pretrained: ImageNet"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Maximum epochs: {MAX_EPOCHS}"
    )

    print(
        f"Early stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\nCREATING DATASETS")
    print("-" * 70)

    train_dataset, val_dataset, test_dataset = (
        create_datasets(
            manifest_path=TOMATO_MANIFEST,
            image_root=COLOR_ROOT,
        )
    )

    print("✓ Datasets created.")

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Val:   {len(val_dataset)}"
    )

    print(
        f"Test:  {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    print("\nCREATING DATALOADERS")
    print("-" * 70)

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    print("✓ DataLoaders created.")

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Val batches:   {len(val_loader)}"
    )

    print(
        f"Test batches:  {len(test_loader)}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    print("\nCLASS WEIGHTS")
    print("-" * 70)

    class_weights = get_class_weights(
        train_dataset
    ).to(DEVICE)

    for index, weight in enumerate(
        class_weights
    ):
        print(
            f"{index:2d} | "
            f"{TOMATO_CLASSES[index]:55s} | "
            f"{weight.item():.4f}"
        )

    print(
        f"\nMean weight: "
        f"{class_weights.mean().item():.4f}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nMODEL")
    print("-" * 70)

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(DEVICE)

    print(
        "✓ EfficientNetV2-B0 created."
    )

    print(
        f"✓ Output classes: "
        f"{NUM_CLASSES}"
    )

    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    print("\nLOSS")
    print("-" * 70)

    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    val_criterion = nn.CrossEntropyLoss()

    print(
        "✓ Training: weighted "
        "CrossEntropyLoss"
    )

    print(
        "✓ Validation: unweighted "
        "CrossEntropyLoss"
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
        mode="max",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    best_val_macro_f1 = -float("inf")

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

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

        train_loss, train_accuracy, train_macro_f1 = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=train_criterion,
                optimizer=optimizer,
                device=DEVICE,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy, val_macro_f1 = (
            validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=val_criterion,
                device=DEVICE,
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_macro_f1
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Epoch output
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"{epoch:03d}/{MAX_EPOCHS}"
        )

        print(
            f"  Train | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {train_accuracy:.4f} | "
            f"Macro F1: {train_macro_f1:.4f}"
        )

        print(
            f"  Val   | "
            f"Loss: {val_loss:.4f} | "
            f"Acc: {val_accuracy:.4f} | "
            f"Macro F1: {val_macro_f1:.4f}"
        )

        print(
            f"  LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_macro_f1=best_val_macro_f1,
            history=history,
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_macro_f1 > best_val_macro_f1:

            best_val_macro_f1 = val_macro_f1

            epochs_without_improvement = 0

            save_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                history=history,
            )

            print(
                "  ✓ New best model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"No validation Macro F1 "
                f"improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
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

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Macro F1: "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Best checkpoint:\n"
        f"{BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Last checkpoint:\n"
        f"{LAST_CHECKPOINT_PATH}"
    )

    print(
        f"Training history:\n"
        f"{HISTORY_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test set was not used during training."
    )

    print(
        "Use evaluate.py for final test-set evaluation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()