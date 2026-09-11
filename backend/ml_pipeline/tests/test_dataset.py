"""
PhytoSense AI
Dataset / DataLoader smoke test.

This test verifies the complete data pipeline:

    Manifest
        ↓
    Dataset
        ↓
    Preprocessing
        ↓
    DataLoader
        ↓
    Batch

The test must pass before model training begins.
"""

import torch

from config import (
    MANIFEST_ROOT,
    PLANTVILLAGE_ROOT,
)

from backend.ml_pipeline.dataset.dataset import (
    create_datasets,
    create_dataloaders,
    get_class_counts,
    get_class_weights,
)


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = (
    MANIFEST_ROOT / "plantvillage_tomato_split.csv"
)

PLANTVILLAGE_COLOR_DIR = (
    PLANTVILLAGE_ROOT / "raw" / "color"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHYTOSENSE AI — DATASET / DATALOADER TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print("\nManifest:")
    print(MANIFEST_PATH)

    print("\nPlantVillage color directory:")
    print(PLANTVILLAGE_COLOR_DIR)

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\nCreating datasets...")

    train_dataset, val_dataset, test_dataset = create_datasets(
        manifest_path=MANIFEST_PATH,
        image_root=PLANTVILLAGE_COLOR_DIR,
    )

    print("✓ Datasets created.")

    # --------------------------------------------------------
    # Dataset sizes
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("DATASET SIZES")
    print("-" * 60)

    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")

    assert len(train_dataset) == 12712, (
        f"Unexpected train size: {len(train_dataset)}"
    )

    assert len(val_dataset) == 2725, (
        f"Unexpected validation size: {len(val_dataset)}"
    )

    assert len(test_dataset) == 2723, (
        f"Unexpected test size: {len(test_dataset)}"
    )

    print("✓ Dataset sizes correct.")

    # --------------------------------------------------------
    # Verify total
    # --------------------------------------------------------

    total = (
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    )

    assert total == 18160, (
        f"Unexpected total dataset size: {total}"
    )

    print(f"✓ Total images: {total}")

    # --------------------------------------------------------
    # Class counts
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("TRAINING CLASS DISTRIBUTION")
    print("-" * 60)

    class_counts = get_class_counts(train_dataset)

    for class_index, count in class_counts.items():
        print(
            f"{class_index:2d} : {count}"
        )

    assert len(class_counts) == 10

    assert sum(class_counts.values()) == 12712, (
        "Training class counts do not sum to "
        "the training dataset size."
    )

    print("\n✓ Training class distribution valid.")

    # --------------------------------------------------------
    # Verify every class has training samples
    # --------------------------------------------------------

    for class_index, count in class_counts.items():

        assert count > 0, (
            f"Class {class_index} has zero "
            "training samples."
        )

    print("✓ All 10 classes have training samples.")

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("MODERATED CLASS WEIGHTS")
    print("-" * 60)

    class_weights = get_class_weights(train_dataset)

    for class_index, weight in enumerate(class_weights):
        print(
            f"{class_index:2d} : {weight:.4f}"
        )

    mean_weight = class_weights.mean().item()

    print(
        f"\nMean weight: {mean_weight:.4f}"
    )

    # --------------------------------------------------------
    # Validate class weights
    # --------------------------------------------------------

    assert class_weights.shape == (10,), (
        f"Unexpected class-weight shape: "
        f"{class_weights.shape}"
    )

    assert torch.isfinite(class_weights).all(), (
        "Class weights contain NaN or infinite values."
    )

    assert torch.all(class_weights > 0), (
        "Class weights must all be positive."
    )

    # Mean-normalized weights should have mean ~1.
    assert abs(mean_weight - 1.0) < 1e-5, (
        f"Class weights are not mean-normalized: "
        f"mean={mean_weight}"
    )

    print("✓ Class weights valid.")

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CREATING DATALOADERS")
    print("-" * 60)

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

    # --------------------------------------------------------
    # DataLoader lengths
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("DATALOADER INFORMATION")
    print("-" * 60)

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Val batches:   {len(val_loader)}"
    )

    print(
        f"Test batches:  {len(test_loader)}"
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    print("✓ DataLoader lengths valid.")

    # --------------------------------------------------------
    # Test training batch
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("TRAINING BATCH")
    print("-" * 60)

    images, labels = next(iter(train_loader))

    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")

    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    ), (
        f"Unexpected training image shape: "
        f"{images.shape}"
    )

    assert labels.shape == (
        BATCH_SIZE,
    ), (
        f"Unexpected training label shape: "
        f"{labels.shape}"
    )

    assert images.dtype == torch.float32, (
        f"Unexpected image dtype: {images.dtype}"
    )

    assert labels.dtype == torch.int64, (
        f"Unexpected label dtype: {labels.dtype}"
    )

    assert torch.isfinite(images).all(), (
        "Training images contain NaN or infinite values."
    )

    assert labels.min().item() >= 0
    assert labels.max().item() <= 9

    print("✓ Training batch valid.")

    # --------------------------------------------------------
    # Test validation batch
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("VALIDATION BATCH")
    print("-" * 60)

    images, labels = next(iter(val_loader))

    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")

    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    ), (
        f"Unexpected validation image shape: "
        f"{images.shape}"
    )

    assert labels.shape == (
        BATCH_SIZE,
    ), (
        f"Unexpected validation label shape: "
        f"{labels.shape}"
    )

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    assert torch.isfinite(images).all()

    assert labels.min().item() >= 0
    assert labels.max().item() <= 9

    print("✓ Validation batch valid.")

    # --------------------------------------------------------
    # Test test batch
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST BATCH")
    print("-" * 60)

    images, labels = next(iter(test_loader))

    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")

    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    ), (
        f"Unexpected test image shape: "
        f"{images.shape}"
    )

    assert labels.shape == (
        BATCH_SIZE,
    ), (
        f"Unexpected test label shape: "
        f"{labels.shape}"
    )

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    assert torch.isfinite(images).all()

    assert labels.min().item() >= 0
    assert labels.max().item() <= 9

    print("✓ Test batch valid.")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("✓ DATASET / DATALOADER TEST PASSED")
    print("=" * 60)

    print("\nPipeline verified:")
    print("  ✓ PlantVillage manifest")
    print("  ✓ 70/15/15 project split")
    print("  ✓ 10 tomato classes")
    print("  ✓ Training class distribution")
    print("  ✓ Moderated class weights")
    print("  ✓ Image loading")
    print("  ✓ Preprocessing")
    print("  ✓ Training DataLoader")
    print("  ✓ Validation DataLoader")
    print("  ✓ Test DataLoader")
    print("  ✓ Batch dimensions")
    print("  ✓ Tensor dtypes")
    print("  ✓ Label ranges")


if __name__ == "__main__":
    main()