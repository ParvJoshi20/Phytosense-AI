from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm

from sklearn.metrics import f1_score

from dataset import (
    create_datasets,
    create_dataloaders,
    get_class_weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "plantvillage_tomato"

WEIGHTS_DIR = BASE_DIR / "weights"

LAST_CHECKPOINT = WEIGHTS_DIR / "efficientnetv2_last.pth"
BEST_CHECKPOINT = WEIGHTS_DIR / "efficientnetv2_best.pth"

TOTAL_EPOCHS = 150

EARLY_STOPPING_PATIENCE = 10

BATCH_SIZE = 16

NUM_WORKERS = 0

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_CLASSES = 10

MODEL_NAME = "tf_efficientnetv2_b0"

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Configure reproducible random number generation.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Request deterministic behavior where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker(worker_id):
    """
    Seed individual DataLoader workers.
    """

    worker_seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(worker_seed)

    random.seed(worker_seed)


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
    best_val_macro_f1,
    epochs_without_improvement,
    classes,
    history,
):

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

        "epochs_without_improvement":
            epochs_without_improvement,

        "classes":
            classes,

        "history":
            history,

        "seed":
            SEED,

        "model_name":
            MODEL_NAME,

        "num_classes":
            NUM_CLASSES,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,
    }

    torch.save(
        checkpoint,
        path,
    )


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

    best_val_macro_f1 = checkpoint.get(
        "best_val_macro_f1",
        0.0,
    )

    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0,
    )

    history = checkpoint.get(
        "history",
        {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "learning_rate": [],
        },
    )

    classes = checkpoint.get(
        "classes",
        None,
    )

    return (
        start_epoch,
        best_val_macro_f1,
        epochs_without_improvement,
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

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    loss_criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            # IMPORTANT:
            # Validation loss is intentionally UNWEIGHTED.
            loss = loss_criterion(
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

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("PHYTO-SENSE AI - EFFICIENTNETV2-B0")
    print("=" * 70)

    print("Device:", DEVICE)

    print("Model:", MODEL_NAME)

    print("Dataset:", DATA_DIR)

    print("Maximum epochs:", TOTAL_EPOCHS)

    print(
        "Early stopping patience:",
        EARLY_STOPPING_PATIENCE,
    )

    print("Batch size:", BATCH_SIZE)

    print("Seed:", SEED)

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
    # TIMM PRETRAINED DATA CONFIG
    # --------------------------------------------------------

    data_config = timm.data.resolve_data_config(
        {},
        model=model,
    )

    print()
    print("Pretrained model data configuration:")

    print(
        "Input size:",
        data_config["input_size"],
    )

    print(
        "Interpolation:",
        data_config["interpolation"],
    )

    print(
        "Mean:",
        data_config["mean"],
    )

    print(
        "Std:",
        data_config["std"],
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_dataset, val_dataset, test_dataset = create_datasets(
        DATA_DIR,
        data_config=data_config,
    )

    print(
        "Train:",
        len(train_dataset),
    )

    print(
        "Validation:",
        len(val_dataset),
    )

    print(
        "Test:",
        len(test_dataset),
    )

    print()
    print("Classes:")

    for i, name in enumerate(
        train_dataset.classes
    ):

        print(
            f"{i}: {name}"
        )

    if len(train_dataset.classes) != NUM_CLASSES:

        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, "
            f"but dataset contains "
            f"{len(train_dataset.classes)} classes."
        )

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        SEED
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        generator=loader_generator,
        worker_init_fn=seed_worker,
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = get_class_weights(
        train_dataset
    ).to(DEVICE)

    print()
    print("Training class weights:")

    for name, weight in zip(
        train_dataset.classes,
        class_weights,
    ):

        print(
            f"{name:50} "
            f"{weight.item():.4f}"
        )

    # --------------------------------------------------------
    # LOSS FUNCTIONS
    # --------------------------------------------------------

    # Weighted loss is used ONLY for training.
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Unweighted loss is used for validation.
    val_criterion = nn.CrossEntropyLoss()

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

    # Scheduler monitors validation macro F1 because
    # macro F1 gives every class equal importance.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    start_epoch = 1

    best_val_macro_f1 = 0.0

    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
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
            best_val_macro_f1,
            epochs_without_improvement,
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
            f"Best validation macro F1: "
            f"{best_val_macro_f1:.4f} "
            f"({best_val_macro_f1 * 100:.2f}%)"
        )

        print(
            "Epochs without improvement:",
            epochs_without_improvement,
        )

        if checkpoint_classes is not None:

            if checkpoint_classes != train_dataset.classes:

                raise RuntimeError(
                    "Checkpoint class order does not "
                    "match dataset class order."
                )

        if start_epoch > TOTAL_EPOCHS:

            print()
            print(
                "Training is already complete."
            )

            return

    else:

        print()
        print(
            "No checkpoint found."
        )

        print(
            "Starting training from epoch 1."
        )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

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

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            (
                train_loss,
                train_accuracy,
            ) = train_one_epoch(
                model,
                train_loader,
                train_criterion,
                optimizer,
                DEVICE,
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            (
                val_loss,
                val_accuracy,
                val_macro_f1,
            ) = validate(
                model,
                val_loader,
                val_criterion,
                DEVICE,
            )

            # ------------------------------------------------
            # LEARNING RATE / SCHEDULER
            # ------------------------------------------------

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            scheduler.step(
                val_macro_f1
            )

            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

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

            history["val_macro_f1"].append(
                val_macro_f1
            )

            history["learning_rate"].append(
                current_lr
            )

            elapsed = (
                time.time()
                - epoch_start
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy: "
                f"{train_accuracy * 100:.2f}%"
            )

            print(
                f"Val Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Val Accuracy: "
                f"{val_accuracy * 100:.2f}%"
            )

            print(
                f"Val Macro F1: "
                f"{val_macro_f1 * 100:.2f}%"
            )

            print(
                f"Learning Rate: "
                f"{current_lr:.7f}"
            )

            print(
                f"Time: "
                f"{elapsed / 60:.2f} min"
            )

            # ------------------------------------------------
            # BEST MODEL / EARLY STOPPING
            # ------------------------------------------------

            if val_macro_f1 > best_val_macro_f1:

                best_val_macro_f1 = (
                    val_macro_f1
                )

                epochs_without_improvement = 0

                save_checkpoint(
                    BEST_CHECKPOINT,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_macro_f1,
                    epochs_without_improvement,
                    train_dataset.classes,
                    history,
                )

                print()
                print(
                    "★ New best model saved!"
                )

                print(
                    f"Best Val Macro F1: "
                    f"{best_val_macro_f1 * 100:.2f}%"
                )

            else:

                epochs_without_improvement += 1

                print()
                print(
                    "No validation macro F1 improvement."
                )

                print(
                    f"Early stopping counter: "
                    f"{epochs_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE}"
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
                best_val_macro_f1,
                epochs_without_improvement,
                train_dataset.classes,
                history,
            )

            print(
                "Last checkpoint saved."
            )

            # ------------------------------------------------
            # EARLY STOPPING
            # ------------------------------------------------

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print()
                print("=" * 70)
                print("EARLY STOPPING TRIGGERED")
                print("=" * 70)

                print(
                    "Validation macro F1 did not "
                    "improve for "
                    f"{EARLY_STOPPING_PATIENCE} "
                    "consecutive epochs."
                )

                print(
                    f"Best validation macro F1: "
                    f"{best_val_macro_f1 * 100:.2f}%"
                )

                break

            print()

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("TRAINING INTERRUPTED")
        print("=" * 70)

        print(
            "The last completed epoch was saved to:"
        )

        print(
            LAST_CHECKPOINT
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
        f"Best validation macro F1: "
        f"{best_val_macro_f1 * 100:.2f}%"
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