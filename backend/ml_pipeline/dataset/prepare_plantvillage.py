"""
PhytoSense AI
PlantVillage Tomato Dataset Preparation

Creates a reproducible 70/15/15 train/validation/test split
from the official PlantVillage tomato color dataset.

The generated manifest records dataset provenance and image
content hashes for downstream integrity and leakage auditing.

This script does not modify the original PlantVillage dataset.
"""

from __future__ import annotations

import csv
import hashlib
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
    DATASET_VERSION,
    require_plantvillage_paths,
)


# ============================================================
# CONSTANTS
# ============================================================

SOURCE_DATASET = "PlantVillage"
UNKNOWN_SOURCE_SPLIT = "unknown"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "class_name",
    "class_index",
    "source_dataset",
    "source_split",
    "sha256",
    "project_split",
]


# ============================================================
# IMAGE IDENTIFIERS / HASHING
# ============================================================

def calculate_sha256(image_path: Path) -> str:
    """
    Calculate the SHA-256 content hash of an image.

    The image is read in binary chunks so the complete file
    does not need to be loaded into memory at once.
    """

    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def create_image_id(relative_path: Path) -> str:
    """
    Create a deterministic image identifier from the image's
    normalized relative path.

    The identifier represents the image's dataset location.
    Content integrity is tracked separately by SHA-256.
    """

    normalized_path = relative_path.as_posix()

    return hashlib.sha256(
        normalized_path.encode("utf-8")
    ).hexdigest()


# ============================================================
# DATA DISCOVERY
# ============================================================

def discover_images() -> list[dict]:
    """
    Discover all tomato images from PlantVillage raw/color.

    Returns:
        List of dictionaries containing image metadata.
    """

    require_plantvillage_paths()

    if COLOR_ROOT is None:
        raise RuntimeError(
            "COLOR_ROOT is not configured."
        )

    if not COLOR_ROOT.exists():
        raise FileNotFoundError(
            "PlantVillage color directory does not exist:\n"
            f"{COLOR_ROOT}"
        )

    records = []

    print("\nDiscovering PlantVillage tomato images...")

    for class_name in TOMATO_CLASSES:
        class_dir = COLOR_ROOT / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                "Expected tomato class directory not found:\n"
                f"{class_dir}"
            )

        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(
            f"{class_name:<55} "
            f"{len(images):>6}"
        )

        for image_path in images:
            relative_path = image_path.relative_to(
                COLOR_ROOT
            )

            records.append(
                {
                    "image_path": image_path,
                    "relative_path": relative_path,
                    "image_id": create_image_id(
                        relative_path
                    ),
                    "class_name": class_name,
                    "class_index": CLASS_TO_INDEX[
                        class_name
                    ],
                    "source_dataset": SOURCE_DATASET,
                    "source_split": UNKNOWN_SOURCE_SPLIT,
                    "sha256": calculate_sha256(
                        image_path
                    ),
                }
            )

    return records


# ============================================================
# ORIGINAL PLANTVILLAGE SPLIT DETECTION
# ============================================================

def detect_original_split(
    image_path: Path,
) -> str:
    """
    Determine the original PlantVillage split when possible.

    The current local PlantVillage image tree does not preserve
    reliable original train/test split information in its path,
    so the source split is recorded as 'unknown'.

    The function is retained as an explicit provenance hook
    rather than fabricating source split information.
    """

    return UNKNOWN_SOURCE_SPLIT


# ============================================================
# STRATIFIED SPLITTING
# ============================================================

def split_class_records(
    records: list[dict],
    rng: random.Random,
) -> None:
    """
    Assign records within each class to train/val/test while
    keeping exact-content duplicate groups together.

    Images sharing the same SHA-256 hash are treated as one
    indivisible group for split assignment. This prevents
    byte-identical image content from appearing across
    train/validation/test.

    The records themselves are not removed or modified beyond
    receiving their project_split assignment.
    """

    by_class: dict[str, list[dict]] = {
        class_name: []
        for class_name in TOMATO_CLASSES
    }

    for record in records:
        by_class[record["class_name"]].append(record)

    for class_name in TOMATO_CLASSES:
        class_records = by_class[class_name]

        # ----------------------------------------------------
        # Group records by exact image content.
        #
        # Every record with the same SHA-256 represents
        # identical file content and must remain in the same
        # project split.
        # ----------------------------------------------------

        groups_by_sha: dict[str, list[dict]] = {}

        for record in class_records:
            groups_by_sha.setdefault(
                record["sha256"],
                [],
            ).append(record)

        groups = list(groups_by_sha.values())

        # Shuffle groups, not individual images.
        rng.shuffle(groups)

        total_records = len(class_records)

        target_train = total_records * TRAIN_RATIO
        target_val = total_records * VAL_RATIO

        train_records = 0
        val_records = 0
        test_records = 0

        for group in groups:
            group_size = len(group)

            # ------------------------------------------------
            # Greedy assignment:
            #
            # Prefer the split whose target is currently
            # furthest behind, while keeping all members of
            # the duplicate group together.
            # ------------------------------------------------

            train_deficit = target_train - train_records
            val_deficit = target_val - val_records

            target_test = total_records * TEST_RATIO
            test_deficit = target_test - test_records

            deficits = {
                "train": train_deficit,
                "val": val_deficit,
                "test": test_deficit,
            }

            split = max(
                deficits,
                key=deficits.get,
            )

            for record in group:
                record["project_split"] = split

            if split == "train":
                train_records += group_size
            elif split == "val":
                val_records += group_size
            else:
                test_records += group_size

        # ----------------------------------------------------
        # Safety checks.
        # ----------------------------------------------------

        assigned_total = (
            train_records
            + val_records
            + test_records
        )

        if assigned_total != total_records:
            raise RuntimeError(
                f"Split assignment error for class "
                f"'{class_name}': "
                f"assigned {assigned_total}, "
                f"expected {total_records}."
            )

        if not (
            train_records > 0
            and val_records > 0
            and test_records > 0
        ):
            raise ValueError(
                f"Class '{class_name}' did not receive "
                "records in all three project splits."
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_records(
    records: list[dict],
) -> None:
    """
    Validate the generated dataset records.

    This validates manifest integrity and structure.

    Duplicate content hashes are intentionally NOT rejected
    here. Exact duplicate and leakage analysis belongs to the
    dedicated dataset audit stage.
    """

    if not records:
        raise ValueError(
            "No images were discovered."
        )

    expected_classes = set(
        TOMATO_CLASSES
    )

    found_classes = {
        record["class_name"]
        for record in records
    }

    missing_classes = (
        expected_classes
        - found_classes
    )

    if missing_classes:
        raise ValueError(
            "Missing tomato classes: "
            f"{sorted(missing_classes)}"
        )

    # --------------------------------------------------------
    # Validate image paths
    # --------------------------------------------------------

    paths = [
        str(
            record["image_path"].resolve()
        )
        for record in records
    ]

    if len(paths) != len(set(paths)):
        raise ValueError(
            "Duplicate image paths detected."
        )

    # --------------------------------------------------------
    # Validate image IDs
    # --------------------------------------------------------

    image_ids = [
        record["image_id"]
        for record in records
    ]

    if len(image_ids) != len(
        set(image_ids)
    ):
        raise ValueError(
            "Duplicate image IDs detected."
        )

    # --------------------------------------------------------
    # Validate SHA-256 values
    # --------------------------------------------------------

    for record in records:
        sha256 = record["sha256"]

        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError(
                "Invalid SHA-256 value for image: "
                f"{record['relative_path']}"
            )

        try:
            int(sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "SHA-256 value contains invalid "
                f"characters for image: "
                f"{record['relative_path']}"
            ) from exc

    # --------------------------------------------------------
    # Validate source metadata
    # --------------------------------------------------------

    for record in records:
        if record["source_dataset"] != (
            SOURCE_DATASET
        ):
            raise ValueError(
                "Invalid source dataset: "
                f"{record['source_dataset']}"
            )

        if not record["source_split"]:
            raise ValueError(
                "Source split cannot be empty."
            )

    # --------------------------------------------------------
    # Validate project splits
    # --------------------------------------------------------

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
            "Invalid project splits: "
            f"{dict(split_counts)}"
        )

    if (
        sum(split_counts.values())
        != len(records)
    ):
        raise ValueError(
            "Split counts do not add up "
            "to total image count."
        )


# ============================================================
# MANIFEST WRITING
# ============================================================

def write_manifest(
    records: list[dict],
) -> None:
    """
    Write the project dataset manifest.
    """

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
                str(r["relative_path"]),
            ),
        ):
            writer.writerow(
                {
                    "image_id": record[
                        "image_id"
                    ],
                    "image_path": record[
                        "relative_path"
                    ].as_posix(),
                    "class_name": record[
                        "class_name"
                    ],
                    "class_index": record[
                        "class_index"
                    ],
                    "source_dataset": record[
                        "source_dataset"
                    ],
                    "source_split": record[
                        "source_split"
                    ],
                    "sha256": record[
                        "sha256"
                    ],
                    "project_split": record[
                        "project_split"
                    ],
                }
            )


# ============================================================
# REPORTING
# ============================================================

def print_report(
    records: list[dict],
) -> None:
    """
    Print final dataset statistics.
    """

    print("\n" + "=" * 70)
    print(
        "PHYTOSENSE AI — "
        "PLANTVILLAGE DATASET PREPARATION"
    )
    print("=" * 70)

    print("\nDATASET")
    print("-" * 70)
    print(
        f"Dataset version:     {DATASET_VERSION}"
    )
    print(
        f"Source dataset:      {SOURCE_DATASET}"
    )
    print(
        f"Total tomato images: {len(records):,}"
    )
    print(
        f"Classes:             {len(TOMATO_CLASSES)}"
    )
    print(
        f"Random seed:         {RANDOM_SEED}"
    )

    print("\nPROJECT SPLIT")
    print("-" * 70)

    split_counts = Counter(
        record["project_split"]
        for record in records
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        count = split_counts[split]

        percentage = (
            count
            / len(records)
            * 100
        )

        print(
            f"{split:<10} "
            f"{count:>6,} "
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
            record
            for record in records
            if record["class_name"]
            == class_name
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

    print("\nMANIFEST")
    print("-" * 70)
    print(TOMATO_MANIFEST)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "PHYTOSENSE AI — "
        "PLANTVILLAGE DATASET PREPARATION"
    )
    print("=" * 70)

    print("\nThis script:")
    print(
        "  • reads the official PlantVillage "
        "color images"
    )
    print(
        "  • uses only the 10 tomato classes"
    )
    print(
        "  • creates a reproducible 70/15/15 split"
    )
    print(
        "  • preserves class proportions"
    )
    print(
        "  • assigns deterministic image IDs"
    )
    print(
        "  • calculates SHA-256 content hashes"
    )
    print(
        "  • records dataset provenance"
    )
    print(
        "  • writes a CSV manifest"
    )
    print(
        "  • does NOT modify the source dataset"
    )
    print(
        "  • does NOT reject duplicate content hashes"
    )

    records = discover_images()

    for record in records:
        record["source_split"] = (
            detect_original_split(
                record["image_path"]
            )
        )

    rng = random.Random(
        RANDOM_SEED
    )

    split_class_records(
        records,
        rng,
    )

    validate_records(records)

    write_manifest(records)

    print_report(records)

    print("\n" + "=" * 70)
    print(
        "✓ DATASET PREPARATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()