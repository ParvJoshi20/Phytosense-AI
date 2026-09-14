"""
PhytoSense AI

Training Pipeline Integration Test.

This test verifies that the complete training stack works together:

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
    Class Weights
        ↓
    Weighted CrossEntropyLoss
        ↓
    Forward Pass
        ↓
    Backward Pass
        ↓
    Optimizer Update

This test performs only a small number of batches.
It does NOT perform actual model training.
It does NOT modify the test set.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    TOMATO_MANIFEST,
    COLOR_ROOT,
    NUM_CLASSES,
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
# TEST CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 0

TEST_BATCHES = 2

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducible integration testing.
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHYTOSENSE AI — TRAINING PIPELINE INTEGRATION TEST")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print("\nDEVICE")
    print("-" * 70)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Manifest: {TOMATO_MANIFEST}"
    )

    print(
        f"Image root: {COLOR_ROOT}"
    )

    train_dataset, val_dataset, test_dataset = (
        create_datasets(
            manifest_path=TOMATO_MANIFEST,
            image_root=COLOR_ROOT,
        )
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Val samples:   {len(val_dataset)}"
    )

    print(
        f"Test samples:  {len(test_dataset)}"
    )

    assert len(train_dataset) == 12712
    assert len(val_dataset) == 2725
    assert len(test_dataset) == 2723

    print(
        "✓ Dataset sizes verified."
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print("\nDATALOADERS")
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
    # Training batch
    # --------------------------------------------------------

    print("\nTRAINING BATCH")
    print("-" * 70)

    batch_iterator = iter(train_loader)

    images, labels = next(batch_iterator)

    print(
        f"Images: {images.shape}"
    )

    print(
        f"Labels: {labels.shape}"
    )

    print(
        f"Image dtype: {images.dtype}"
    )

    print(
        f"Label dtype: {labels.dtype}"
    )

    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    )

    assert labels.shape == (
        BATCH_SIZE,
    )

    assert images.dtype == torch.float32

    assert labels.dtype == torch.int64

    assert torch.isfinite(images).all()

    assert labels.min().item() >= 0

    assert labels.max().item() < NUM_CLASSES

    print(
        "✓ Training batch verified."
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    print("\nCLASS WEIGHTS")
    print("-" * 70)

    class_weights = get_class_weights(
        train_dataset
    ).to(DEVICE)

    print(
        f"Shape: {class_weights.shape}"
    )

    print(
        f"Mean:  {class_weights.mean().item():.4f}"
    )

    assert class_weights.shape == (
        NUM_CLASSES,
    )

    assert torch.isfinite(
        class_weights
    ).all()

    assert torch.all(
        class_weights > 0
    )

    print(
        "✓ Class weights verified."
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

    model.train()

    print(
        "✓ EfficientNetV2-B0 initialized."
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    print("\nLOSS")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print(
        "✓ Weighted CrossEntropyLoss initialized."
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    print("\nOPTIMIZER")
    print("-" * 70)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "✓ AdamW initialized."
    )

    # --------------------------------------------------------
    # Initial parameter snapshot
    # --------------------------------------------------------

    print("\nPARAMETER UPDATE CHECK")
    print("-" * 70)

    first_parameter_before = next(
        model.parameters()
    ).detach().clone()

    # --------------------------------------------------------
    # Training integration
    # --------------------------------------------------------

    print("\nFORWARD / BACKWARD TEST")
    print("-" * 70)

    total_loss = 0.0

    for batch_index in range(
        TEST_BATCHES
    ):

        print(
            f"\nBatch "
            f"{batch_index + 1}/{TEST_BATCHES}"
        )

        images, labels = next(
            batch_iterator
        )

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        print(
            f"Output shape: {outputs.shape}"
        )

        assert outputs.shape == (
            BATCH_SIZE,
            NUM_CLASSES,
        )

        assert torch.isfinite(
            outputs
        ).all()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels,
        )

        print(
            f"Loss: {loss.item():.4f}"
        )

        assert torch.isfinite(loss)

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient validation
        # ----------------------------------------------------

        gradient_count = 0

        for parameter in model.parameters():

            if parameter.grad is not None:

                gradient_count += 1

                assert torch.isfinite(
                    parameter.grad
                ).all()

        assert gradient_count > 0

        print(
            f"Parameters with gradients: "
            f"{gradient_count}"
        )

        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        optimizer.step()

        total_loss += loss.item()

        print(
            "✓ Forward pass"
        )

        print(
            "✓ Loss calculation"
        )

        print(
            "✓ Backward pass"
        )

        print(
            "✓ Gradient validation"
        )

        print(
            "✓ Optimizer update"
        )

    # --------------------------------------------------------
    # Verify parameter update
    # --------------------------------------------------------

    first_parameter_after = next(
        model.parameters()
    ).detach().clone()

    parameters_changed = not torch.equal(
        first_parameter_before,
        first_parameter_after,
    )

    assert parameters_changed, (
        "Model parameters did not change "
        "after optimizer updates."
    )

    print(
        "\n✓ Model parameters changed "
        "after optimizer updates."
    )

    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    average_loss = (
        total_loss / TEST_BATCHES
    )

    print(
        f"\nAverage integration-test loss: "
        f"{average_loss:.4f}"
    )

    assert torch.isfinite(
        torch.tensor(average_loss)
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("✓ TRAINING PIPELINE INTEGRATION TEST PASSED")
    print("=" * 70)

    print("\nVerified:")
    print("  ✓ Dataset")
    print("  ✓ Preprocessing")
    print("  ✓ Training DataLoader")
    print("  ✓ EfficientNetV2-B0")
    print("  ✓ 10-class output")
    print("  ✓ Moderated class weights")
    print("  ✓ Weighted CrossEntropyLoss")
    print("  ✓ AdamW optimizer")
    print("  ✓ Forward propagation")
    print("  ✓ Loss calculation")
    print("  ✓ Backward propagation")
    print("  ✓ Gradient validity")
    print("  ✓ Parameter updates")

    print("\nNo full training was performed.")
    print("No test-set samples were used for optimization.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()