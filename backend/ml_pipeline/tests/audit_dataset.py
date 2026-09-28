from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPECTED_TOTAL = 18160
TOLERANCE = 0.01

EXPECTED_COLUMNS = [
    "image_id",
    "image_path",
    "class_name",
    "class_index",
    "source_dataset",
    "source_split",
    "sha256",
    "project_split",
]

PROJECT_SPLITS = {"train", "val", "test"}

PHASH_MAX_DISTANCE = 8
PHASH_BITS = 64
PHASH_CHUNK_SIZE = 256

MAX_PRINTED_DUPLICATE_GROUPS = 20
MAX_PRINTED_CANDIDATES = 20

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# audit_dataset.py lives in:
#   backend/ml_pipeline/tests/audit_dataset.py
#
# Therefore:
#   parents[0] -> tests/
#   parents[1] -> ml_pipeline/
#   parents[2] -> backend/
#   parents[3] -> repository root/

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ML_PIPELINE_ROOT = PROJECT_ROOT / "backend" / "ml_pipeline"

AUDIT_ROOT = ML_PIPELINE_ROOT / "audit_reports"

MANIFEST_PATH = (
    ML_PIPELINE_ROOT
    / "manifests"
    / "plantvillage_tomato_split.csv"
)

COLOR_ROOT = Path(
    r"D:\Code\_Archieve\Minor Project\Datasets"
    r"\PlantVillage-Dataset\raw\color"
)

AUDIT_SUMMARY_JSON = AUDIT_ROOT / "audit_summary.json"

EXACT_DUPLICATES_CSV = (
    AUDIT_ROOT / "exact_duplicate_groups.csv"
)

CROSS_SPLIT_EXACT_DUPLICATES_CSV = (
    AUDIT_ROOT / "cross_split_exact_duplicates.csv"
)

NEAR_DUPLICATES_CSV = (
    AUDIT_ROOT / "near_duplicate_candidates.csv"
)

CROSS_SPLIT_NEAR_DUPLICATES_CSV = (
    AUDIT_ROOT / "cross_split_near_duplicates.csv"
)

REVIEW_ROOT = (
    AUDIT_ROOT / "near_duplicate_review"
)

REVIEW_MANIFEST_CSV = (
    REVIEW_ROOT / "review_manifest.csv"
)

AUDIT_SUMMARY_JSON = AUDIT_ROOT / "audit_summary.json"

EXACT_DUPLICATES_CSV = AUDIT_ROOT / "exact_duplicate_groups.csv"
CROSS_SPLIT_EXACT_DUPLICATES_CSV = (
    AUDIT_ROOT / "cross_split_exact_duplicates.csv"
)

NEAR_DUPLICATES_CSV = AUDIT_ROOT / "near_duplicate_candidates.csv"
CROSS_SPLIT_NEAR_DUPLICATES_CSV = (
    AUDIT_ROOT / "cross_split_near_duplicates.csv"
)

# Visual review artifacts.
REVIEW_ROOT = AUDIT_ROOT / "near_duplicate_review"
REVIEW_MANIFEST_CSV = REVIEW_ROOT / "review_manifest.csv"

REVIEW_PAIRS_PER_PAGE = 4
REVIEW_IMAGE_SIZE = (420, 320)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> list[dict]:
    """Load the project manifest and validate its columns."""

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                "Manifest columns do not match expected schema.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {reader.fieldnames}"
            )

        rows = list(reader)

    print(f"Loaded manifest: {path}")
    print(f"Manifest rows: {len(rows)}")

    return rows


# ---------------------------------------------------------------------------
# Basic structure audits
# ---------------------------------------------------------------------------


def audit_structure(rows: list[dict]) -> bool:
    """Verify expected total row count."""

    actual = len(rows)

    if actual != EXPECTED_TOTAL:
        print(
            f"[FAIL] Manifest row count: expected "
            f"{EXPECTED_TOTAL}, found {actual}"
        )
        return False

    print(f"[PASS] Manifest row count: {actual}")

    return True


def audit_classes(rows: list[dict]) -> bool:
    """Verify class indices and class names are internally consistent."""

    class_mapping: dict[int, str] = {}

    for row in rows:
        try:
            class_index = int(row["class_index"])
        except ValueError:
            print(
                f"[FAIL] Invalid class_index for image "
                f"{row['image_id']}: {row['class_index']}"
            )
            return False

        class_name = row["class_name"]

        if class_index in class_mapping:
            if class_mapping[class_index] != class_name:
                print(
                    "[FAIL] Class index maps to multiple class names: "
                    f"{class_index} -> "
                    f"{class_mapping[class_index]!r}, {class_name!r}"
                )
                return False
        else:
            class_mapping[class_index] = class_name

    print(
        f"[PASS] Class mapping is internally consistent: "
        f"{len(class_mapping)} classes"
    )

    return True


def audit_provenance(rows: list[dict]) -> bool:
    """Verify dataset provenance fields."""

    valid = True

    for row in rows:
        if row["source_dataset"] != "PlantVillage":
            print(
                f"[FAIL] Unexpected source_dataset for "
                f"{row['image_id']}: {row['source_dataset']}"
            )
            valid = False

    if valid:
        print("[PASS] Dataset provenance fields are valid")

    return valid


# ---------------------------------------------------------------------------
# Image/path audits
# ---------------------------------------------------------------------------


def resolve_image_path(row: dict) -> Path:
    """Resolve manifest image_path against the dataset color root."""

    return COLOR_ROOT / row["image_path"]


def audit_images(rows: list[dict]) -> bool:
    """Verify all manifest image files exist and can be opened."""

    valid = True

    for index, row in enumerate(rows, start=1):
        image_path = resolve_image_path(row)

        if not image_path.exists():
            print(
                f"[FAIL] Missing image: "
                f"{row['image_id']} -> {image_path}"
            )
            valid = False
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            print(
                f"[FAIL] Invalid image: "
                f"{row['image_id']} -> {image_path} "
                f"({exc})"
            )
            valid = False

        if index % 1000 == 0:
            print(f"Image validation progress: {index}/{len(rows)}")

    if valid:
        print(f"[PASS] All {len(rows)} manifest images exist and are readable")

    return valid


def audit_record_uniqueness(rows: list[dict]) -> bool:
    """Verify image_id and image_path are unique."""

    image_ids = [row["image_id"] for row in rows]
    image_paths = [row["image_path"] for row in rows]

    duplicate_ids = [
        image_id
        for image_id, count in Counter(image_ids).items()
        if count > 1
    ]

    duplicate_paths = [
        image_path
        for image_path, count in Counter(image_paths).items()
        if count > 1
    ]

    valid = True

    if duplicate_ids:
        print(
            f"[FAIL] Duplicate image IDs: "
            f"{len(duplicate_ids)}"
        )
        valid = False
    else:
        print("[PASS] image_id values are unique")

    if duplicate_paths:
        print(
            f"[FAIL] Duplicate image paths: "
            f"{len(duplicate_paths)}"
        )
        valid = False
    else:
        print("[PASS] image_path values are unique")

    return valid


# ---------------------------------------------------------------------------
# SHA-256 audits
# ---------------------------------------------------------------------------


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 hash for a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def audit_sha256(rows: list[dict]) -> bool:
    """Verify every manifest SHA-256 matches the actual image file."""

    valid = True

    for index, row in enumerate(rows, start=1):
        image_path = resolve_image_path(row)

        if not image_path.exists():
            continue

        actual_hash = calculate_sha256(image_path)

        if actual_hash != row["sha256"]:
            print(
                f"[FAIL] SHA-256 mismatch: "
                f"{row['image_id']}\n"
                f"       Manifest: {row['sha256']}\n"
                f"       Actual:   {actual_hash}"
            )
            valid = False

        if index % 1000 == 0:
            print(f"SHA-256 progress: {index}/{len(rows)}")

    if valid:
        print("[PASS] All SHA-256 values match image contents")

    return valid


def group_exact_duplicates(rows: list[dict]) -> dict[str, list[dict]]:
    """Group manifest records by SHA-256."""

    groups: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        groups[row["sha256"]].append(row)

    return {
        sha256: group
        for sha256, group in groups.items()
        if len(group) > 1
    }


def write_exact_duplicate_reports(
    duplicate_groups: dict[str, list[dict]],
) -> None:
    """Write exact duplicate reports."""

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    with EXACT_DUPLICATES_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "sha256",
            "group_size",
            "image_id",
            "image_path",
            "class_name",
            "project_split",
        ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sha256, group in sorted(duplicate_groups.items()):
            for row in group:
                writer.writerow(
                    {
                        "sha256": sha256,
                        "group_size": len(group),
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "class_name": row["class_name"],
                        "project_split": row["project_split"],
                    }
                )

    cross_split_groups = {
        sha256: group
        for sha256, group in duplicate_groups.items()
        if len({row["project_split"] for row in group}) > 1
    }

    with CROSS_SPLIT_EXACT_DUPLICATES_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "sha256",
            "group_size",
            "image_id",
            "image_path",
            "class_name",
            "project_split",
        ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for sha256, group in sorted(cross_split_groups.items()):
            for row in group:
                writer.writerow(
                    {
                        "sha256": sha256,
                        "group_size": len(group),
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "class_name": row["class_name"],
                        "project_split": row["project_split"],
                    }
                )


def audit_exact_duplicates(
    rows: list[dict],
) -> tuple[
    dict[str, list[dict]],
    dict[str, list[dict]],
]:
    """Audit exact duplicate content and cross-split exact duplicates."""

    duplicate_groups = group_exact_duplicates(rows)

    cross_split_groups = {
        sha256: group
        for sha256, group in duplicate_groups.items()
        if len({row["project_split"] for row in group}) > 1
    }

    unique_hashes = len({row["sha256"] for row in rows})

    print(f"[INFO] Unique SHA-256 hashes: {unique_hashes}")
    print(f"[INFO] Exact duplicate groups: {len(duplicate_groups)}")

    if duplicate_groups:
        print(
            f"[INFO] Showing up to "
            f"{MAX_PRINTED_DUPLICATE_GROUPS} duplicate groups:"
        )

        for sha256, group in list(
            sorted(duplicate_groups.items())
        )[:MAX_PRINTED_DUPLICATE_GROUPS]:
            splits = sorted(
                {row["project_split"] for row in group}
            )

            print(
                f"  {sha256[:16]}... "
                f"count={len(group)} "
                f"splits={splits}"
            )

    if cross_split_groups:
        print(
            "[FAIL] Cross-split exact duplicate groups: "
            f"{len(cross_split_groups)}"
        )
    else:
        print("[PASS] No cross-split exact duplicate groups")

    write_exact_duplicate_reports(duplicate_groups)

    return duplicate_groups, cross_split_groups


# ---------------------------------------------------------------------------
# Split audits
# ---------------------------------------------------------------------------


def audit_splits(rows: list[dict]) -> bool:
    """Verify project split values and approximate 70/15/15 distribution."""

    split_counts = Counter(row["project_split"] for row in rows)

    if set(split_counts) != PROJECT_SPLITS:
        print(
            "[FAIL] Unexpected project splits: "
            f"{sorted(split_counts)}"
        )
        return False

    total = len(rows)

    expected_ratios = {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    }

    valid = True

    for split, expected_ratio in expected_ratios.items():
        actual_ratio = split_counts[split] / total
        difference = abs(actual_ratio - expected_ratio)

        print(
            f"[INFO] {split}: "
            f"{split_counts[split]} "
            f"({actual_ratio:.4%})"
        )

        if difference > TOLERANCE:
            print(
                f"[FAIL] {split} ratio differs from expected "
                f"{expected_ratio:.2%} by more than "
                f"{TOLERANCE:.2%}"
            )
            valid = False

    if valid:
        print("[PASS] Project split ratios are within tolerance")

    return valid


def audit_path_leakage(rows: list[dict]) -> bool:
    """Verify no identical image paths occur across project splits."""

    path_splits: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        path_splits[row["image_path"]].add(row["project_split"])

    leakage = {
        path: splits
        for path, splits in path_splits.items()
        if len(splits) > 1
    }

    if leakage:
        print(
            "[FAIL] Image paths appear in multiple project splits: "
            f"{len(leakage)}"
        )

        for path, splits in list(leakage.items())[:MAX_PRINTED_CANDIDATES]:
            print(f"  {path}: {sorted(splits)}")

        return False

    print("[PASS] No image path appears across multiple project splits")

    return True


# ---------------------------------------------------------------------------
# Perceptual hash
# ---------------------------------------------------------------------------


def compute_phash(path: Path) -> int:
    """
    Compute a 64-bit perceptual hash and return it as an integer.
    """

    with Image.open(path) as image:
        hash_value = imagehash.phash(image)

    return int(str(hash_value), 16)


def build_phash_index(rows: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Compute perceptual hashes for all images.

    Returns:
        records: list of manifest records in stable input order.
        hash_values: uint64 NumPy array containing 64-bit pHash values.
    """

    records: list[dict] = []
    hash_values: list[int] = []

    total = len(rows)

    for index, row in enumerate(rows, start=1):
        image_path = resolve_image_path(row)

        try:
            phash_value = compute_phash(image_path)
        except Exception as exc:
            print(
                f"[FAIL] Could not compute pHash for "
                f"{row['image_id']}: {exc}"
            )
            raise

        records.append(row)
        hash_values.append(phash_value)

        if index % 500 == 0 or index == total:
            print(
                f"pHash progress: {index}/{total}"
            )

    return records, np.asarray(hash_values, dtype=np.uint64)


def _hamming_distance_uint64(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """
    Calculate pairwise Hamming distances between uint64 hash arrays.

    Returns a matrix with shape:
        (len(left), len(right))
    """

    xor_values = np.bitwise_xor(
        left[:, None],
        right[None, :],
    )

    xor_bytes = xor_values.view(np.uint8).reshape(
        left.shape[0],
        right.shape[0],
        8,
    )

    # Lookup table for the number of set bits in each byte.
    popcount_lookup = np.array(
        [bin(value).count("1") for value in range(256)],
        dtype=np.uint8,
    )

    return popcount_lookup[xor_bytes].sum(axis=2)


def find_near_duplicate_candidates(
    records: list[dict],
    hash_values: np.ndarray,
    max_distance: int = PHASH_MAX_DISTANCE,
    chunk_size: int = PHASH_CHUNK_SIZE,
) -> list[dict]:
    """
    Find all image pairs with pHash Hamming distance <= max_distance.

    Important:
        Every unordered image pair is compared exactly once.

    The previous chunking approach allowed overlapping left/right chunks,
    causing some cross-chunk pairs to be compared multiple times and making
    the progress counter exceed 100%.

    This implementation avoids that by:
        1. Comparing pairs inside each chunk using the upper triangle.
        2. Comparing each chunk only against chunks that come after it.

    Therefore:
        - no pair is skipped;
        - no pair is compared more than once;
        - progress reaches exactly 100%.
    """

    total_images = len(records)

    if hash_values.shape[0] != total_images:
        raise ValueError(
            "records and hash_values must have the same length"
        )

    if total_images < 2:
        return []

    # Number of unordered image pairs.
    total_pairs = total_images * (total_images - 1) // 2
    processed_pairs = 0

    candidates: list[dict] = []

    for left_start in range(
        0,
        total_images - 1,
        chunk_size,
    ):
        left_end = min(
            left_start + chunk_size,
            total_images,
        )

        left_hashes = hash_values[left_start:left_end]
        left_count = left_end - left_start

        # ---------------------------------------------------------------
        # 1. Compare images within the same chunk.
        # ---------------------------------------------------------------

        if left_count > 1:
            distances = _hamming_distance_uint64(
                left_hashes,
                left_hashes,
            )

            upper_indices = np.triu_indices(
                left_count,
                k=1,
            )

            within_distances = distances[
                upper_indices
            ]

            matching_positions = np.where(
                within_distances <= max_distance
            )[0]

            for position in matching_positions:
                local_left = int(upper_indices[0][position])
                local_right = int(upper_indices[1][position])

                left_index = left_start + local_left
                right_index = left_start + local_right

                distance = int(within_distances[position])

                left_record = records[left_index]
                right_record = records[right_index]

                candidates.append(
                    {
                        "left_image_id": left_record["image_id"],
                        "right_image_id": right_record["image_id"],
                        "left_image_path": left_record["image_path"],
                        "right_image_path": right_record["image_path"],
                        "left_class_name": left_record["class_name"],
                        "right_class_name": right_record["class_name"],
                        "left_split": left_record["project_split"],
                        "right_split": right_record["project_split"],
                        "hamming_distance": distance,
                        "cross_split": (
                            left_record["project_split"]
                            != right_record["project_split"]
                        ),
                    }
                )

            processed_pairs += left_count * (left_count - 1) // 2

        # ---------------------------------------------------------------
        # 2. Compare this chunk only against later chunks.
        # ---------------------------------------------------------------

        for right_start in range(
            left_end,
            total_images,
            chunk_size,
        ):
            right_end = min(
                right_start + chunk_size,
                total_images,
            )

            right_hashes = hash_values[
                right_start:right_end
            ]

            distances = _hamming_distance_uint64(
                left_hashes,
                right_hashes,
            )

            matching_positions = np.where(
                distances <= max_distance
            )

            for local_left, local_right in zip(
                matching_positions[0],
                matching_positions[1],
            ):
                left_index = left_start + int(local_left)
                right_index = right_start + int(local_right)

                distance = int(
                    distances[local_left, local_right]
                )

                left_record = records[left_index]
                right_record = records[right_index]

                candidates.append(
                    {
                        "left_image_id": left_record["image_id"],
                        "right_image_id": right_record["image_id"],
                        "left_image_path": left_record["image_path"],
                        "right_image_path": right_record["image_path"],
                        "left_class_name": left_record["class_name"],
                        "right_class_name": right_record["class_name"],
                        "left_split": left_record["project_split"],
                        "right_split": right_record["project_split"],
                        "hamming_distance": distance,
                        "cross_split": (
                            left_record["project_split"]
                            != right_record["project_split"]
                        ),
                    }
                )

            processed_pairs += (
                left_count * (right_end - right_start)
            )

        progress = (
            processed_pairs / total_pairs * 100
            if total_pairs
            else 100.0
        )

        print(
            "Near-duplicate comparison progress: "
            f"{progress:.1f}%"
        )

    # Defensive validation.
    if processed_pairs != total_pairs:
        raise RuntimeError(
            "Near-duplicate comparison did not process exactly "
            f"all unordered pairs: "
            f"{processed_pairs} / {total_pairs}"
        )

    # Sort deterministically.
    candidates.sort(
        key=lambda item: (
            item["hamming_distance"],
            item["left_image_id"],
            item["right_image_id"],
        )
    )

    return candidates


# ---------------------------------------------------------------------------
# Near-duplicate reports
# ---------------------------------------------------------------------------


def write_near_duplicate_reports(
    candidates: list[dict],
) -> tuple[Path, Path]:
    """Write all near-duplicate and cross-split candidate reports."""

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "left_image_id",
        "right_image_id",
        "left_image_path",
        "right_image_path",
        "left_class_name",
        "right_class_name",
        "left_split",
        "right_split",
        "hamming_distance",
        "cross_split",
    ]

    with NEAR_DUPLICATES_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(candidates)

    cross_split_candidates = [
        candidate
        for candidate in candidates
        if candidate["cross_split"]
    ]

    with CROSS_SPLIT_NEAR_DUPLICATES_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(cross_split_candidates)

    return (
        NEAR_DUPLICATES_CSV,
        CROSS_SPLIT_NEAR_DUPLICATES_CSV,
    )


# ---------------------------------------------------------------------------
# Visual near-duplicate review
# ---------------------------------------------------------------------------


def _load_review_image(
    image_path: Path,
) -> Image.Image:
    """
    Load and prepare an image for the review contact sheet.

    The returned image is RGB and resized to fit inside REVIEW_IMAGE_SIZE
    while preserving aspect ratio.
    """

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail(REVIEW_IMAGE_SIZE)

        canvas = Image.new(
            "RGB",
            REVIEW_IMAGE_SIZE,
            "white",
        )

        x = (
            REVIEW_IMAGE_SIZE[0] - image.width
        ) // 2

        y = (
            REVIEW_IMAGE_SIZE[1] - image.height
        ) // 2

        canvas.paste(image, (x, y))

    return canvas


def _draw_review_text(
    draw,
    x: int,
    y: int,
    lines: list[str],
    max_width: int,
    line_height: int = 18,
) -> int:
    """
    Draw metadata lines and return the new y position.

    Uses PIL's default font to avoid external font dependencies.
    """

    from PIL import ImageFont

    font = ImageFont.load_default()

    current_y = y

    for line in lines:
        # Basic wrapping based on approximate character width.
        max_chars = max(20, max_width // 7)

        if len(line) <= max_chars:
            wrapped_lines = [line]
        else:
            wrapped_lines = [
                line[index:index + max_chars]
                for index in range(
                    0,
                    len(line),
                    max_chars,
                )
            ]

        for wrapped in wrapped_lines:
            draw.text(
                (x, current_y),
                wrapped,
                fill="black",
                font=font,
            )
            current_y += line_height

    return current_y


def create_near_duplicate_review(
    candidates: list[dict],
) -> tuple[Path, Path]:
    """
    Create visual contact sheets for cross-split pHash candidates.

    The generated artifacts are intentionally for human review only.
    No candidate is automatically classified as leakage.
    """

    cross_split_candidates = [
        candidate
        for candidate in candidates
        if candidate["cross_split"]
    ]

    REVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale generated image files only.
    # The source dataset and manifest are never touched.
    for path in REVIEW_ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }
        ):
            path.unlink()

    # Stable deterministic ordering.
    cross_split_candidates.sort(
        key=lambda item: (
            item["hamming_distance"],
            item["left_image_id"],
            item["right_image_id"],
        )
    )

    review_rows: list[dict] = []

    pair_id_counter = 1

    grouped_by_distance: dict[int, list[dict]] = defaultdict(list)

    for candidate in cross_split_candidates:
        grouped_by_distance[
            candidate["hamming_distance"]
        ].append(candidate)

    for distance in sorted(grouped_by_distance):
        distance_candidates = grouped_by_distance[distance]

        distance_root = (
            REVIEW_ROOT
            / f"distance_{distance:02d}"
        )

        distance_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for page_offset in range(
            0,
            len(distance_candidates),
            REVIEW_PAIRS_PER_PAGE,
        ):
            page_candidates = distance_candidates[
                page_offset:
                page_offset + REVIEW_PAIRS_PER_PAGE
            ]

            page_number = (
                page_offset // REVIEW_PAIRS_PER_PAGE
            ) + 1

            page_name = (
                f"page_{page_number:03d}.jpg"
            )

            page_path = distance_root / page_name

            # Each pair gets:
            #   left image
            #   right image
            #   metadata
            #
            # Pair blocks are stacked vertically.
            pair_block_height = 420
            page_height = (
                pair_block_height
                * len(page_candidates)
            )

            page_width = (
                REVIEW_IMAGE_SIZE[0] * 2
                + 40
            )

            sheet = Image.new(
                "RGB",
                (page_width, page_height),
                "white",
            )

            draw = ImageDraw.Draw(sheet)

            for pair_index, candidate in enumerate(
                page_candidates
            ):
                pair_id = (
                    f"PAIR-{pair_id_counter:04d}"
                )
                pair_id_counter += 1

                top = (
                    pair_index
                    * pair_block_height
                )

                left_path = resolve_image_path(
                    {
                        "image_path": candidate[
                            "left_image_path"
                        ]
                    }
                )

                right_path = resolve_image_path(
                    {
                        "image_path": candidate[
                            "right_image_path"
                        ]
                    }
                )

                left_image = _load_review_image(
                    left_path
                )

                right_image = _load_review_image(
                    right_path
                )

                image_y = top + 10

                sheet.paste(
                    left_image,
                    (10, image_y),
                )

                sheet.paste(
                    right_image,
                    (
                        REVIEW_IMAGE_SIZE[0] + 20,
                        image_y,
                    ),
                )

                metadata_y = (
                    image_y
                    + REVIEW_IMAGE_SIZE[1]
                    + 10
                )

                metadata_lines = [
                    (
                        f"{pair_id} | "
                        f"pHash distance="
                        f"{candidate['hamming_distance']}"
                    ),
                    (
                        f"LEFT: "
                        f"{candidate['left_image_id']} | "
                        f"{candidate['left_split']} | "
                        f"{candidate['left_class_name']}"
                    ),
                    (
                        f"RIGHT: "
                        f"{candidate['right_image_id']} | "
                        f"{candidate['right_split']} | "
                        f"{candidate['right_class_name']}"
                    ),
                    (
                        "LEFT PATH: "
                        f"{candidate['left_image_path']}"
                    ),
                    (
                        "RIGHT PATH: "
                        f"{candidate['right_image_path']}"
                    ),
                    (
                        "REVIEW: visually inspect whether "
                        "these images represent the same "
                        "or effectively duplicated sample."
                    ),
                ]

                _draw_review_text(
                    draw=draw,
                    x=10,
                    y=metadata_y,
                    lines=metadata_lines,
                    max_width=page_width - 20,
                )

                review_rows.append(
                    {
                        "pair_id": pair_id,
                        "left_image_id": candidate[
                            "left_image_id"
                        ],
                        "right_image_id": candidate[
                            "right_image_id"
                        ],
                        "left_image_path": candidate[
                            "left_image_path"
                        ],
                        "right_image_path": candidate[
                            "right_image_path"
                        ],
                        "left_class_name": candidate[
                            "left_class_name"
                        ],
                        "right_class_name": candidate[
                            "right_class_name"
                        ],
                        "left_split": candidate[
                            "left_split"
                        ],
                        "right_split": candidate[
                            "right_split"
                        ],
                        "hamming_distance": candidate[
                            "hamming_distance"
                        ],
                        "review_status": "",
                        "review_notes": "",
                        "review_page": str(
                            page_path.relative_to(
                                REVIEW_ROOT
                            )
                        ),
                    }
                )

            sheet.save(
                page_path,
                quality=90,
            )

    review_fields = [
        "pair_id",
        "left_image_id",
        "right_image_id",
        "left_image_path",
        "right_image_path",
        "left_class_name",
        "right_class_name",
        "left_split",
        "right_split",
        "hamming_distance",
        "review_status",
        "review_notes",
        "review_page",
    ]

    with REVIEW_MANIFEST_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=review_fields,
        )

        writer.writeheader()
        writer.writerows(review_rows)

    print(
        "[INFO] Created visual review artifacts: "
        f"{len(review_rows)} cross-split candidate pairs"
    )

    print(
        f"[INFO] Review root: {REVIEW_ROOT}"
    )

    print(
        f"[INFO] Review manifest: "
        f"{REVIEW_MANIFEST_CSV}"
    )

    return REVIEW_MANIFEST_CSV, REVIEW_ROOT


def audit_near_duplicates(
    rows: list[dict],
) -> dict:
    """
    Run the complete perceptual near-duplicate audit.
    """

    print()
    print("=" * 72)
    print("PERCEPTUAL NEAR-DUPLICATE AUDIT")
    print("=" * 72)

    records, hash_values = build_phash_index(rows)

    candidates = find_near_duplicate_candidates(
        records=records,
        hash_values=hash_values,
        max_distance=PHASH_MAX_DISTANCE,
        chunk_size=PHASH_CHUNK_SIZE,
    )

    (
        near_duplicate_report,
        cross_split_report,
    ) = write_near_duplicate_reports(
        candidates
    )

    cross_split_candidates = [
        candidate
        for candidate in candidates
        if candidate["cross_split"]
    ]

    print(
        f"[INFO] Total pHash candidates "
        f"(distance <= {PHASH_MAX_DISTANCE}): "
        f"{len(candidates)}"
    )

    print(
        f"[INFO] Cross-split pHash candidates: "
        f"{len(cross_split_candidates)}"
    )

    if cross_split_candidates:
        print(
            f"[WARN] Cross-split near-duplicate candidates "
            f"require human review."
        )

        print(
            f"[INFO] Showing up to "
            f"{MAX_PRINTED_CANDIDATES} candidates:"
        )

        for candidate in cross_split_candidates[
            :MAX_PRINTED_CANDIDATES
        ]:
            print(
                f"  distance={candidate['hamming_distance']} | "
                f"{candidate['left_image_id']} "
                f"({candidate['left_split']}) <-> "
                f"{candidate['right_image_id']} "
                f"({candidate['right_split']}) | "
                f"{candidate['left_class_name']} <-> "
                f"{candidate['right_class_name']}"
            )
    else:
        print(
            "[PASS] No cross-split pHash candidates "
            f"at distance <= {PHASH_MAX_DISTANCE}"
        )

    review_manifest, review_root = (
        create_near_duplicate_review(candidates)
    )

    return {
        "candidates": candidates,
        "cross_split_candidates": cross_split_candidates,
        "near_duplicate_report": str(
            near_duplicate_report
        ),
        "cross_split_report": str(
            cross_split_report
        ),
        "review_manifest": str(
            review_manifest
        ),
        "review_root": str(
            review_root
        ),
    }


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------


def calculate_class_distribution(
    rows: list[dict],
) -> dict:
    """Calculate class counts by project split."""

    distribution: dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    for row in rows:
        split = row["project_split"]
        class_name = row["class_name"]

        if split in distribution:
            distribution[split][class_name] += 1

    return distribution


def print_class_distribution(
    distribution: dict[str, Counter],
) -> None:
    """Print class distribution by split."""

    print()
    print("=" * 72)
    print("CLASS DISTRIBUTION")
    print("=" * 72)

    classes = sorted(
        {
            class_name
            for split_counts in distribution.values()
            for class_name in split_counts
        }
    )

    for class_name in classes:
        train = distribution["train"][class_name]
        val = distribution["val"][class_name]
        test = distribution["test"][class_name]

        print(
            f"{class_name}: "
            f"train={train}, "
            f"val={val}, "
            f"test={test}, "
            f"total={train + val + test}"
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_audit_summary(
    rows: list[dict],
    audit_results: dict,
    near_duplicate_results: dict,
) -> None:
    """Write a machine-readable audit summary."""

    distribution = calculate_class_distribution(rows)

    summary = {
        "manifest_path": str(MANIFEST_PATH),
        "dataset_root": str(COLOR_ROOT),
        "expected_total": EXPECTED_TOTAL,
        "actual_total": len(rows),
        "unique_sha256": len(
            {row["sha256"] for row in rows}
        ),
        "exact_duplicate_groups": len(
            audit_results["exact_duplicate_groups"]
        ),
        "cross_split_exact_duplicate_groups": len(
            audit_results[
                "cross_split_exact_duplicate_groups"
            ]
        ),
        "near_duplicate_candidates": len(
            near_duplicate_results["candidates"]
        ),
        "cross_split_near_duplicate_candidates": len(
            near_duplicate_results[
                "cross_split_candidates"
            ]
        ),
        "phash_max_distance": PHASH_MAX_DISTANCE,
        "phash_chunk_size": PHASH_CHUNK_SIZE,
        "project_split_counts": dict(
            Counter(
                row["project_split"]
                for row in rows
            )
        ),
        "class_distribution": {
            split: dict(counts)
            for split, counts in distribution.items()
        },
        "near_duplicate_review_manifest": (
            near_duplicate_results[
                "review_manifest"
            ]
        ),
        "near_duplicate_review_root": (
            near_duplicate_results[
                "review_root"
            ]
        ),
        "audit_status": {
            "structure": audit_results["structure"],
            "classes": audit_results["classes"],
            "provenance": audit_results["provenance"],
            "images": audit_results["images"],
            "record_uniqueness": audit_results[
                "record_uniqueness"
            ],
            "sha256": audit_results["sha256"],
            "splits": audit_results["splits"],
            "path_leakage": audit_results[
                "path_leakage"
            ],
            "cross_split_exact_duplicates": (
                len(
                    audit_results[
                        "cross_split_exact_duplicate_groups"
                    ]
                )
                == 0
            ),
            # Perceptual duplicates are deliberately not treated
            # as an automatic pass/fail condition because pHash
            # candidates require human inspection.
            "cross_split_near_duplicate_review_required": (
                len(
                    near_duplicate_results[
                        "cross_split_candidates"
                    ]
                )
                > 0
            ),
        },
    }

    AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Main audit pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("PHYTOSENSE AI - PLANTVILLAGE TOMATO DATASET AUDIT")
    print("=" * 72)

    rows = load_manifest(MANIFEST_PATH)

    print()
    print("=" * 72)
    print("STRUCTURE / METADATA AUDITS")
    print("=" * 72)

    structure_ok = audit_structure(rows)
    classes_ok = audit_classes(rows)
    provenance_ok = audit_provenance(rows)
    images_ok = audit_images(rows)
    uniqueness_ok = audit_record_uniqueness(rows)

    print()
    print("=" * 72)
    print("CONTENT / LEAKAGE AUDITS")
    print("=" * 72)

    sha256_ok = audit_sha256(rows)

    (
        exact_duplicate_groups,
        cross_split_exact_duplicate_groups,
    ) = audit_exact_duplicates(rows)

    splits_ok = audit_splits(rows)
    path_leakage_ok = audit_path_leakage(rows)

    distribution = calculate_class_distribution(rows)
    print_class_distribution(distribution)

    near_duplicate_results = audit_near_duplicates(rows)

    audit_results = {
        "structure": structure_ok,
        "classes": classes_ok,
        "provenance": provenance_ok,
        "images": images_ok,
        "record_uniqueness": uniqueness_ok,
        "sha256": sha256_ok,
        "exact_duplicate_groups": exact_duplicate_groups,
        "cross_split_exact_duplicate_groups": (
            cross_split_exact_duplicate_groups
        ),
        "splits": splits_ok,
        "path_leakage": path_leakage_ok,
    }

    write_audit_summary(
        rows=rows,
        audit_results=audit_results,
        near_duplicate_results=near_duplicate_results,
    )

    structural_checks = [
        structure_ok,
        classes_ok,
        provenance_ok,
        images_ok,
        uniqueness_ok,
        sha256_ok,
        splits_ok,
        path_leakage_ok,
        len(cross_split_exact_duplicate_groups) == 0,
    ]

    print()
    print("=" * 72)
    print("FINAL AUDIT STATUS")
    print("=" * 72)

    if all(structural_checks):
        print("[PASS] Structural and exact-leakage audits passed")
    else:
        print(
            "[FAIL] One or more structural/exact-leakage "
            "audits failed"
        )

    if near_duplicate_results["cross_split_candidates"]:
        print(
            "[REVIEW REQUIRED] Cross-split perceptual "
            "near-duplicate candidates exist."
        )
        print(
            "Review the generated contact sheets before "
            "freezing the dataset."
        )
    else:
        print(
            "[PASS] No cross-split perceptual candidates "
            f"at pHash distance <= {PHASH_MAX_DISTANCE}"
        )

    print()
    print(
        f"Audit summary: {AUDIT_SUMMARY_JSON}"
    )

    print(
        f"Near-duplicate report: "
        f"{near_duplicate_results['near_duplicate_report']}"
    )

    print(
        f"Cross-split near-duplicate report: "
        f"{near_duplicate_results['cross_split_report']}"
    )

    print(
        f"Visual review manifest: "
        f"{near_duplicate_results['review_manifest']}"
    )

    print(
        f"Visual review directory: "
        f"{near_duplicate_results['review_root']}"
    )


if __name__ == "__main__":
    main()