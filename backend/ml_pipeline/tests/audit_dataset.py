"""
PhytoSense AI
Independent PlantVillage Dataset Audit

Audits the project-generated tomato dataset manifest.

This script does NOT create or modify the dataset.
It independently verifies:
    - manifest integrity
    - expected classes
    - image existence
    - duplicate paths
    - split proportions
    - class distribution
    - split leakage
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    COLOR_ROOT,
    TOMATO_CLASSES,
    CLASS_TO_INDEX,
    TOMATO_MANIFEST,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_TOTAL = 18160

TOLERANCE = 0.01


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest() -> list[dict]:
    if not TOMATO_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{TOMATO_MANIFEST}\n"
            "Run prepare_plantvillage.py first."
        )

    with TOMATO_MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)
        records = list(reader)

    if not records:
        raise ValueError("Manifest is empty.")

    return records


# ============================================================
# BASIC STRUCTURE AUDIT
# ============================================================

def audit_structure(records: list[dict]) -> None:

    required_columns = {
        "image_path",
        "class_name",
        "class_index",
        "original_split",
        "project_split",
    }

    actual_columns = set(records[0].keys())

    missing = required_columns - actual_columns

    if missing:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing)}"
        )

    print("✓ Manifest columns valid.")


# ============================================================
# CLASS AUDIT
# ============================================================

def audit_classes(records: list[dict]) -> None:

    found_classes = {
        record["class_name"]
        for record in records
    }

    expected_classes = set(TOMATO_CLASSES)

    missing = expected_classes - found_classes
    unexpected = found_classes - expected_classes

    if missing:
        raise ValueError(
            f"Missing classes: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected classes: {sorted(unexpected)}"
        )

    for record in records:
        class_name = record["class_name"]

        expected_index = CLASS_TO_INDEX[class_name]
        actual_index = int(record["class_index"])

        if expected_index != actual_index:
            raise ValueError(
                f"Incorrect class index for {class_name}: "
                f"{actual_index} != {expected_index}"
            )

    print("✓ All 10 tomato classes present.")
    print("✓ Class indices valid.")


# ============================================================
# IMAGE EXISTENCE AUDIT
# ============================================================

def audit_images(records: list[dict]) -> None:

    missing_images = []

    for record in records:

        relative_path = Path(record["image_path"])
        image_path = COLOR_ROOT / relative_path

        if not image_path.exists():
            missing_images.append(str(image_path))

    if missing_images:

        print("\nMissing images:")

        for path in missing_images[:20]:
            print(path)

        raise FileNotFoundError(
            f"{len(missing_images)} image(s) referenced "
            "by the manifest do not exist."
        )

    print(
        f"✓ All {len(records):,} manifest images exist."
    )


# ============================================================
# DUPLICATE AUDIT
# ============================================================

def audit_duplicates(records: list[dict]) -> None:

    paths = [
        record["image_path"]
        for record in records
    ]

    duplicates = [
        path
        for path, count in Counter(paths).items()
        if count > 1
    ]

    if duplicates:

        print("\nDuplicate paths:")

        for path in duplicates[:20]:
            print(path)

        raise ValueError(
            f"{len(duplicates)} duplicate image path(s) found."
        )

    print("✓ No duplicate image paths.")


# ============================================================
# SPLIT AUDIT
# ============================================================

def audit_splits(records: list[dict]) -> None:

    split_counts = Counter(
        record["project_split"]
        for record in records
    )

    expected_splits = {
        "train",
        "val",
        "test",
    }

    if set(split_counts) != expected_splits:
        raise ValueError(
            f"Unexpected splits: {dict(split_counts)}"
        )

    total = len(records)

    expected_ratios = {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    }

    print("\nSPLIT DISTRIBUTION")
    print("-" * 55)

    for split in ("train", "val", "test"):

        count = split_counts[split]
        ratio = count / total

        expected = expected_ratios[split]

        print(
            f"{split:<10}"
            f"{count:>7,}"
            f"   {ratio * 100:>6.2f}%"
            f"   expected {expected * 100:.2f}%"
        )

        if abs(ratio - expected) > TOLERANCE:
            raise ValueError(
                f"{split} ratio outside tolerance."
            )

    print("✓ Train/validation/test ratios valid.")


# ============================================================
# SPLIT LEAKAGE AUDIT
# ============================================================

def audit_split_leakage(records: list[dict]) -> None:

    split_paths = defaultdict(set)

    for record in records:

        split_paths[
            record["project_split"]
        ].add(record["image_path"])

    train_val = (
        split_paths["train"]
        & split_paths["val"]
    )

    train_test = (
        split_paths["train"]
        & split_paths["test"]
    )

    val_test = (
        split_paths["val"]
        & split_paths["test"]
    )

    if train_val or train_test or val_test:

        raise ValueError(
            "Image leakage detected between splits."
        )

    print("✓ No image-level split leakage detected.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    records: list[dict],
) -> None:

    print("\nCLASS DISTRIBUTION")
    print("-" * 90)

    print(
        f"{'CLASS':<52}"
        f"{'TRAIN':>8}"
        f"{'VAL':>8}"
        f"{'TEST':>8}"
        f"{'TOTAL':>9}"
    )

    print("-" * 90)

    for class_name in TOMATO_CLASSES:

        class_records = [
            record
            for record in records
            if record["class_name"] == class_name
        ]

        counts = Counter(
            record["project_split"]
            for record in class_records
        )

        print(
            f"{class_name:<52}"
            f"{counts['train']:>8}"
            f"{counts['val']:>8}"
            f"{counts['test']:>8}"
            f"{len(class_records):>9}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PHYTOSENSE AI — INDEPENDENT PLANTVILLAGE DATASET AUDIT")
    print("=" * 70)

    print("\nManifest:")
    print(TOMATO_MANIFEST)

    print("\nPlantVillage:")
    print(COLOR_ROOT)

    records = load_manifest()

    print(
        f"\nManifest records: {len(records):,}"
    )

    # --------------------------------------------------------
    # Total count
    # --------------------------------------------------------

    if len(records) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL:,} records, "
            f"found {len(records):,}."
        )

    print(
        f"✓ Expected total confirmed: "
        f"{EXPECTED_TOTAL:,}"
    )

    # --------------------------------------------------------
    # Audits
    # --------------------------------------------------------

    print("\nSTRUCTURE")
    print("-" * 70)

    audit_structure(records)

    print("\nCLASSES")
    print("-" * 70)

    audit_classes(records)

    print("\nIMAGE FILES")
    print("-" * 70)

    audit_images(records)

    print("\nDUPLICATES")
    print("-" * 70)

    audit_duplicates(records)

    print("\nSPLITS")
    print("-" * 70)

    audit_splits(records)

    print("\nLEAKAGE")
    print("-" * 70)

    audit_split_leakage(records)

    print_class_distribution(records)

    print("\n" + "=" * 70)
    print("✓ PLANTVILLAGE DATASET AUDIT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()