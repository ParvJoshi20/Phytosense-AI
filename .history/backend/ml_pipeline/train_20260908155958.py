from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from backend.ml_pipeline.dataset import (
    create_datasets,
    create_dataloaders,
    get_class_weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "backend" / "ml_pipeline" / "data" / "combined_tomato"
WEIGHTS_DIR = ROOT_DIR / "backend" / "ml_pipeline" / "weights"

LAST_CHECKPOINT = WEIGHTS_DIR / "efficientnetv2_last.pth"
BEST_CHECKPOINT = WEIGHTS_DIR / "efficientnetv2_best.pth"

TOTAL_EPOCHS = 100

BATCH_SIZE = 16

NUM_WORKERS = 0

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_CLASSES = 8

MODEL_NAME = "tf_efficientnetv2_b0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    return model


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_accuracy,
    classes,
    history,
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "best_val_accuracy": best_val_accuracy,

        "classes": classes,

        "history": history,
    }

    torch.save(checkpoint, path)


# ============================================================
# CHECKPOINT LOAD
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_val_accuracy = checkpoint.get(
        "best_val_accuracy",
        0.0,
    )

    history = checkpoint.get(
        "history",
        {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        },
    )

    classes = checkpoint.get(
        "classes",
        None,
    )

    return (
        start_epoch,
        best_val_accuracy,
        history,
        classes,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XAI-PHYTOSENSE - EFFICIENTNETV2-B0")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)
    print("Dataset:", DATA_DIR)
    print("Total epochs:", TOTAL_EPOCHS)
    print("Batch size:", BATCH_SIZE)

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_dataset, val_dataset, test_dataset = create_datasets(
        DATA_DIR
    )

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    print()
    print("Classes:")

    for i, name in enumerate(train_dataset.classes):

        print(f"{i}: {name}")

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = get_class_weights(
        train_dataset
    ).to(DEVICE)

    print()
    print("Class weights:")

    for name, weight in zip(
        train_dataset.classes,
        class_weights,
    ):

        print(
            f"{name:30} {weight.item():.3f}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Creating EfficientNetV2-B0...")

    model = create_model()

    model = model.to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {parameters:,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # DEFAULT TRAINING STATE
    # --------------------------------------------------------

    start_epoch = 1

    best_val_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    # --------------------------------------------------------
    # RESUME CHECKPOINT
    # --------------------------------------------------------

    if LAST_CHECKPOINT.exists():

        print()
        print("=" * 70)
        print("CHECKPOINT FOUND")
        print("=" * 70)

        print(
            "Loading:",
            LAST_CHECKPOINT,
        )

        (
            start_epoch,
            best_val_accuracy,
            history,
            checkpoint_classes,
        ) = load_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            DEVICE,
        )

        print(
            "Last completed epoch:",
            start_epoch - 1,
        )

        print(
            "Next epoch:",
            start_epoch,
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.4f} "
            f"({best_val_accuracy * 100:.2f}%)"
        )

        if checkpoint_classes is not None:

            if checkpoint_classes != train_dataset.classes:

                raise RuntimeError(
                    "Checkpoint class order does not match dataset class order."
                )

        if start_epoch > TOTAL_EPOCHS:

            print()
            print(
                "Training is already complete."
            )

            return

    else:

        print()
        print("No checkpoint found.")
        print("Starting training from epoch 1.")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(
        f"Training epochs {start_epoch} "
        f"to {TOTAL_EPOCHS}"
    )

    print()

    try:

        for epoch in range(
            start_epoch,
            TOTAL_EPOCHS + 1,
        ):

            epoch_start = time.time()

            print(
                f"Epoch {epoch}/{TOTAL_EPOCHS}"
            )

            print("-" * 70)

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
            )

            val_loss, val_accuracy = validate(
                model,
                val_loader,
                criterion,
                DEVICE,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(
                val_accuracy
            )

            history["train_loss"].append(
                train_loss
            )

            history["train_accuracy"].append(
                train_accuracy
            )

            history["val_loss"].append(
                val_loss
            )

            history["val_accuracy"].append(
                val_accuracy
            )

            history["learning_rate"].append(
                current_lr
            )

            elapsed = time.time() - epoch_start

            print(
                f"Train Loss: {train_loss:.4f}"
            )

            print(
                f"Train Accuracy: "
                f"{train_accuracy * 100:.2f}%"
            )

            print(
                f"Val Loss: {val_loss:.4f}"
            )

            print(
                f"Val Accuracy: "
                f"{val_accuracy * 100:.2f}%"
            )

            print(
                f"Learning Rate: {current_lr:.7f}"
            )

            print(
                f"Time: {elapsed / 60:.2f} min"
            )

            # ------------------------------------------------
            # BEST CHECKPOINT
            # ------------------------------------------------

            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy

                save_checkpoint(
                    BEST_CHECKPOINT,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_accuracy,
                    train_dataset.classes,
                    history,
                )

                print()
                print(
                    "★ New best model saved!"
                )

                print(
                    f"Best Val Accuracy: "
                    f"{best_val_accuracy * 100:.2f}%"
                )

            # ------------------------------------------------
            # LAST CHECKPOINT
            # ------------------------------------------------

            save_checkpoint(
                LAST_CHECKPOINT,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_accuracy,
                train_dataset.classes,
                history,
            )

            print(
                "Last checkpoint saved."
            )

            print()

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("TRAINING INTERRUPTED")
        print("=" * 70)

        print(
            "The last completed epoch has already "
            "been saved to:"
        )

        print(LAST_CHECKPOINT)

        print()
        print(
            "Run the same command again to resume."
        )

        return

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        "Best model:",
        BEST_CHECKPOINT,
    )

    print(
        "Last model:",
        LAST_CHECKPOINT,
    )


if __name__ == "__main__":

    main()