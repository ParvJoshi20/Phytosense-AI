"""
PhytoSense AI
PlantVillage Tomato Dataset Preparation

Creates a reproducible 70/15/15 train/validation/test split
from the official PlantVillage tomato color dataset.

The original PlantVillage train/test designation is preserved
as metadata but is NOT used as the project's final split.

This script does not modify the original PlantVillage dataset.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from config import (
    COLOR_ROOT,
    MANIFEST_ROOT,
    TOMATO_CLASSES,
    CLASS_TO_INDEX,
    TOMATO_MANIFEST,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

MANIFEST_COLUMNS = [
    "image_path",
    "class_name",
    "class_index",
    "original_split",
    "project_split",
]


# ============================================================
# DATA DISCOVERY
# ============================================================

def discover_images() -> list[dict]:
    """
    Discover all tomato images from PlantVillage raw/color.

    Returns:
        List of dictionaries containing image metadata.
    """

    if not COLOR_ROOT.exists():
        raise FileNotFoundError(
            f"PlantVillage color directory does not exist:\n{COLOR_ROOT}"
        )

    records = []

    print("\nDiscovering PlantVillage tomato images...")

    for class_name in TOMATO_CLASSES:
        class_dir = COLOR_ROOT / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected tomato class directory not found:\n{class_dir}"
            )

        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_name:<55} {len(images):>6}")

        for image_path in images:
            records.append(
                {
                    "image_path": image_path,
                    "class_name": class_name,
                    "class_index": CLASS_TO_INDEX[class_name],
                }
            )

    return records


# ============================================================
# ORIGINAL PLANTVILLAGE SPLIT DETECTION
# ============================================================

def detect_original_split(image_path: Path) -> str:
    """
    Determine the original PlantVillage split when possible.

    The official PlantVillage repository contains separate
    train/test metadata. Because the local image tree itself
    does not encode this information in its directory name,
    this function currently returns 'unknown'.

    The field is retained in the manifest so the provenance
    can be populated later from the official metadata.
    """

    return "unknown"


# ============================================================
# STRATIFIED SPLITTING
# ============================================================

def split_class_records(
    records: list[dict],
    rng: random.Random,
) -> None:
    """
    Assign records within each class to train/val/test.

    Splitting is performed independently for every class so
    that class proportions are approximately preserved.
    """

    by_class: dict[str, list[dict]] = {
        class_name: []
        for class_name in TOMATO_CLASSES
    }

    for record in records:
        by_class[record["class_name"]].append(record)

    for class_name in TOMATO_CLASSES:
        class_records = by_class[class_name]

        rng.shuffle(class_records)

        total = len(class_records)

        train_count = round(total * TRAIN_RATIO)
        val_count = round(total * VAL_RATIO)

        # Make sure all samples are assigned exactly once.
        test_count = total - train_count - val_count

        if train_count <= 0 or val_count <= 0 or test_count <= 0:
            raise ValueError(
                f"Class '{class_name}' is too small for the requested split."
            )

        for index, record in enumerate(class_records):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"

            record["project_split"] = split


# ============================================================
# VALIDATION
# ============================================================

def validate_records(records: list[dict]) -> None:
    """Validate the generated dataset records."""

    expected_total = len(records)

    if expected_total == 0:
        raise ValueError("No images were discovered.")

    expected_classes = set(TOMATO_CLASSES)
    found_classes = {r["class_name"] for r in records}

    missing_classes = expected_classes - found_classes

    if missing_classes:
        raise ValueError(
            f"Missing tomato classes: {sorted(missing_classes)}"
        )

    paths = [str(r["image_path"].resolve()) for r in records]

    if len(paths) != len(set(paths)):
        raise ValueError(
            "Duplicate image paths detected."
        )

    split_counts = Counter(
        record["project_split"]
        for record in records
    )

    if set(split_counts) != {"train", "val", "test"}:
        raise ValueError(
            f"Invalid project splits: {dict(split_counts)}"
        )

    if sum(split_counts.values()) != expected_total:
        raise ValueError(
            "Split counts do not add up to total image count."
        )


# ============================================================
# MANIFEST WRITING
# ============================================================

def write_manifest(records: list[dict]) -> None:
    """Write the project dataset manifest."""

    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TOMATO_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
        )

        writer.writeheader()

        for record in sorted(
            records,
            key=lambda r: (
                r["project_split"],
                r["class_index"],
                str(r["image_path"]),
            ),
        ):

            relative_path = record["image_path"].relative_to(
                COLOR_ROOT
            )

            writer.writerow(
                {
                    "image_path": str(relative_path).replace(
                        "\\",
                        "/",
                    ),
                    "class_name": record["class_name"],
                    "class_index": record["class_index"],
                    "original_split": record["original_split"],
                    "project_split": record["project_split"],
                }
            )


# ============================================================
# REPORTING
# ============================================================

def print_report(records: list[dict]) -> None:
    """Print final dataset statistics."""

    print("\n" + "=" * 70)
    print("PHY TOSENSE AI — PLANTVILLAGE DATASET PREPARATION")
    print("=" * 70)

    print("\nSOURCE")
    print("-" * 70)
    print(f"PlantVillage color directory:")
    print(COLOR_ROOT)

    print("\nDATASET")
    print("-" * 70)
    print(f"Total tomato images: {len(records):,}")
    print(f"Classes:             {len(TOMATO_CLASSES)}")
    print(f"Random seed:         {RANDOM_SEED}")

    print("\nPROJECT SPLIT")
    print("-" * 70)

    split_counts = Counter(
        record["project_split"]
        for record in records
    )

    for split in ("train", "val", "test"):
        count = split_counts[split]
        percentage = count / len(records) * 100

        print(
            f"{split:<10} {count:>6,} "
            f"({percentage:>6.2f}%)"
        )

    print("\nCLASS DISTRIBUTION")
    print("-" * 70)

    print(
        f"{'CLASS':<52}"
        f"{'TRAIN':>8}"
        f"{'VAL':>8}"
        f"{'TEST':>8}"
        f"{'TOTAL':>9}"
    )

    print("-" * 85)

    for class_name in TOMATO_CLASSES:
        class_records = [
            r
            for r in records
            if r["class_name"] == class_name
        ]

        counts = Counter(
            r["project_split"]
            for r in class_records
        )

        print(
            f"{class_name:<52}"
            f"{counts['train']:>8}"
            f"{counts['val']:>8}"
            f"{counts['test']:>8}"
            f"{len(class_records):>9}"
        )

    print("\nMANIFEST")
    print("-" * 70)
    print(TOMATO_MANIFEST)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PHYTOSENSE AI — PLANTVILLAGE DATASET PREPARATION")
    print("=" * 70)

    print("\nThis script:")
    print("  • reads the official PlantVillage color images")
    print("  • uses only the 10 tomato classes")
    print("  • creates a reproducible 70/15/15 split")
    print("  • preserves class proportions")
    print("  • writes a CSV manifest")
    print("  • does NOT modify the source dataset")

    records = discover_images()

    if len(records) != 18160:
        print(
            "\nWARNING:"
            f" Expected approximately 18,160 tomato images,"
            f" but discovered {len(records):,}."
        )

    for record in records:
        record["original_split"] = detect_original_split(
            record["image_path"]
        )

    rng = random.Random(RANDOM_SEED)

    split_class_records(
        records,
        rng,
    )

    validate_records(records)

    write_manifest(records)

    print_report(records)

    print("\n" + "=" * 70)
    print("✓ DATASET PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()