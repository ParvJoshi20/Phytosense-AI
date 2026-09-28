"""
PhytoSense AI
Dataset / DataLoader smoke test.

This test verifies the complete data pipeline:

    Manifest
        ↓
    Manifest validation
        ↓
    Dataset
        ↓
    Preprocessing
        ↓
    DataLoader
        ↓
    Batch

The test must pass before model training begins.

Important:
    Exact and near-duplicate image analysis is intentionally
    NOT performed here. That belongs to the dataset leakage
    audit stage.
"""

import pandas as pd
import torch

from config import (
    MANIFEST_ROOT,
    TOMATO_CLASSES,
    require_plantvillage_paths,
)

from dataset.dataset import (
    create_datasets,
    create_dataloaders,
    get_class_counts,
    get_class_weights,
    validate_manifest,
)


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = (
    MANIFEST_ROOT / "plantvillage_tomato_split.csv"
)

PLANTVILLAGE_ROOT, PLANTVILLAGE_COLOR_DIR = (
    require_plantvillage_paths()
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0

EXPECTED_TRAIN_SIZE = 12712
EXPECTED_VAL_SIZE = 2725
EXPECTED_TEST_SIZE = 2723
EXPECTED_TOTAL_SIZE = 18160

NUM_CLASSES = len(TOMATO_CLASSES)


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

    print("\nPlantVillage root:")
    print(PLANTVILLAGE_ROOT)

    print("\nPlantVillage color directory:")
    print(PLANTVILLAGE_COLOR_DIR)

    assert MANIFEST_PATH.exists(), (
        f"Manifest does not exist:\n{MANIFEST_PATH}"
    )

    assert PLANTVILLAGE_COLOR_DIR.exists(), (
        "PlantVillage color directory does not exist:\n"
        f"{PLANTVILLAGE_COLOR_DIR}"
    )

    print("✓ Required paths exist.")

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("MANIFEST VALIDATION")
    print("-" * 60)

    manifest = pd.read_csv(MANIFEST_PATH)

    print(f"Manifest rows: {len(manifest)}")

    # validate_manifest() checks:
    #   - required columns
    #   - image IDs
    #   - image paths
    #   - class names
    #   - class indices
    #   - source dataset
    #   - source split
    #   - SHA-256 format
    #   - project splits

    validate_manifest(manifest)

    print("✓ Manifest structure and metadata valid.")

    # --------------------------------------------------------
    # Verify manifest total
    # --------------------------------------------------------

    assert len(manifest) == EXPECTED_TOTAL_SIZE, (
        f"Unexpected manifest size: {len(manifest)}"
    )

    print(
        f"✓ Manifest contains {len(manifest)} records."
    )

    # --------------------------------------------------------
    # Verify source dataset
    # --------------------------------------------------------

    source_datasets = set(
        manifest["source_dataset"].unique()
    )

    assert source_datasets == {"PlantVillage"}, (
        "Unexpected source dataset values: "
        f"{source_datasets}"
    )

    print("✓ Source dataset: PlantVillage.")

    # --------------------------------------------------------
    # Verify source split metadata
    # --------------------------------------------------------

    source_splits = set(
        manifest["source_split"].astype(str).unique()
    )

    assert source_splits == {"unknown"}, (
        "Unexpected source split values: "
        f"{source_splits}"
    )

    print(
        "✓ Source split metadata correctly recorded as "
        "'unknown'."
    )

    # --------------------------------------------------------
    # Verify SHA-256 metadata
    # --------------------------------------------------------

    assert manifest["sha256"].notna().all(), (
        "Manifest contains missing SHA-256 hashes."
    )

    assert (
        manifest["sha256"]
        .astype(str)
        .str.fullmatch(r"[0-9a-fA-F]{64}")
        .all()
    ), (
        "Manifest contains invalid SHA-256 hashes."
    )

    print("✓ SHA-256 metadata valid.")

    # --------------------------------------------------------
    # Verify classes
    # --------------------------------------------------------

    found_classes = set(
        manifest["class_name"].unique()
    )

    assert found_classes == set(TOMATO_CLASSES), (
        "Manifest classes do not match frozen "
        "10-class definition."
    )

    print(
        f"✓ Frozen {NUM_CLASSES}-class definition verified."
    )

    # --------------------------------------------------------
    # Verify project split distribution
    # --------------------------------------------------------

    split_counts = (
        manifest["project_split"]
        .value_counts()
        .to_dict()
    )

    print("\nProject split distribution:")

    for split in ("train", "val", "test"):
        print(
            f"{split:5s}: "
            f"{split_counts.get(split, 0)}"
        )

    assert split_counts.get("train") == EXPECTED_TRAIN_SIZE
    assert split_counts.get("val") == EXPECTED_VAL_SIZE
    assert split_counts.get("test") == EXPECTED_TEST_SIZE

    print("✓ Project split distribution valid.")

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CREATING DATASETS")
    print("-" * 60)

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

    assert len(train_dataset) == EXPECTED_TRAIN_SIZE, (
        f"Unexpected train size: {len(train_dataset)}"
    )

    assert len(val_dataset) == EXPECTED_VAL_SIZE, (
        f"Unexpected validation size: {len(val_dataset)}"
    )

    assert len(test_dataset) == EXPECTED_TEST_SIZE, (
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

    assert total == EXPECTED_TOTAL_SIZE, (
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

    assert len(class_counts) == NUM_CLASSES

    for class_index, count in class_counts.items():
        print(
            f"{class_index:2d} : {count}"
        )

    assert sum(class_counts.values()) == EXPECTED_TRAIN_SIZE, (
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

    print(
        f"✓ All {NUM_CLASSES} classes have "
        "training samples."
    )

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

    assert class_weights.shape == (
        NUM_CLASSES,
    ), (
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
    assert labels.max().item() < NUM_CLASSES

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
    assert labels.max().item() < NUM_CLASSES

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
    assert labels.max().item() < NUM_CLASSES

    print("✓ Test batch valid.")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("✓ DATASET / DATALOADER TEST PASSED")
    print("=" * 60)

    print("\nPipeline verified:")
    print("  ✓ PlantVillage manifest")
    print("  ✓ Manifest schema")
    print("  ✓ Manifest provenance")
    print("  ✓ SHA-256 metadata")
    print("  ✓ 70/15/15 project split")
    print(f"  ✓ {NUM_CLASSES} tomato classes")
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