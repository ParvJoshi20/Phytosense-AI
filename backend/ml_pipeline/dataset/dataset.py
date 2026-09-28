"""
PhytoSense AI
Manifest-based PlantVillage Dataset Pipeline.

The CSV manifest is the single source of truth for the
train / validation / test split.

The original PlantVillage dataset is never modified.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
)

from dataset.transforms import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
)


# ============================================================
# MANIFEST SCHEMA
# ============================================================

REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "image_path",
    "class_name",
    "class_index",
    "source_dataset",
    "source_split",
    "sha256",
    "project_split",
}

EXPECTED_PROJECT_SPLITS = {
    "train",
    "val",
    "test",
}


# ============================================================
# DATASET
# ============================================================

class PlantVillageTomatoDataset(Dataset):
    """
    PyTorch Dataset backed by the PlantVillage project manifest.

    The manifest determines:
        - which image is used
        - its class
        - its project split

    This class does not perform dataset splitting.
    """

    def __init__(
        self,
        dataframe,
        image_root,
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"PlantVillage image directory does not exist:\n"
                f"{self.image_root}"
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = self.image_root / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image referenced by manifest does not exist:\n"
                f"{image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        label = int(row["class_index"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# MANIFEST VALIDATION
# ============================================================

def validate_manifest(manifest):
    """
    Validate the structure and class mapping of the manifest.

    The manifest is treated as the source of truth for the
    project split.

    Duplicate content hashes are intentionally NOT rejected here.
    Duplicate-content auditing belongs to the dataset leakage
    audit stage.
    """

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = (
        REQUIRED_MANIFEST_COLUMNS
        - set(manifest.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Manifest is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    if manifest.empty:
        raise RuntimeError(
            "Manifest contains no records."
        )

    # --------------------------------------------------------
    # Validate image IDs
    # --------------------------------------------------------

    if manifest["image_id"].isna().any():
        raise RuntimeError(
            "Manifest contains missing image IDs."
        )

    if manifest["image_id"].duplicated().any():
        raise RuntimeError(
            "Manifest contains duplicate image IDs."
        )

    # --------------------------------------------------------
    # Validate image paths
    # --------------------------------------------------------

    if manifest["image_path"].isna().any():
        raise RuntimeError(
            "Manifest contains missing image paths."
        )

    if manifest["image_path"].duplicated().any():
        raise RuntimeError(
            "Manifest contains duplicate image paths."
        )

    # --------------------------------------------------------
    # Validate classes
    # --------------------------------------------------------

    found_classes = set(
        manifest["class_name"].unique()
    )

    expected_classes = set(CLASS_NAMES)

    if found_classes != expected_classes:
        raise RuntimeError(
            "Manifest classes do not match the expected "
            "PlantVillage tomato classes.\n\n"
            f"Expected:\n{CLASS_NAMES}\n\n"
            f"Found:\n{sorted(found_classes)}"
        )

    # --------------------------------------------------------
    # Validate class indices
    # --------------------------------------------------------

    for _, row in manifest.iterrows():

        class_name = row["class_name"]
        class_index = int(row["class_index"])

        if class_name not in CLASS_TO_INDEX:
            raise RuntimeError(
                f"Unknown class in manifest: {class_name}"
            )

        expected_index = CLASS_TO_INDEX[class_name]

        if class_index != expected_index:
            raise RuntimeError(
                "Invalid class mapping:\n"
                f"{class_name} → {class_index}\n"
                f"Expected → {expected_index}"
            )

    # --------------------------------------------------------
    # Validate source dataset
    # --------------------------------------------------------

    source_datasets = set(
        manifest["source_dataset"].unique()
    )

    if source_datasets != {"PlantVillage"}:
        raise RuntimeError(
            "Manifest contains unexpected source dataset values.\n"
            f"Expected: {{'PlantVillage'}}\n"
            f"Found: {source_datasets}"
        )

    # --------------------------------------------------------
    # Validate source split
    # --------------------------------------------------------

    if manifest["source_split"].isna().any():
        raise RuntimeError(
            "Manifest contains missing source split values."
        )

    # We deliberately allow "unknown".
    #
    # The preparation stage must not fabricate an original
    # PlantVillage train/test assignment when it cannot be
    # established from the authoritative source.
    if (manifest["source_split"].astype(str).str.strip() == "").any():
        raise RuntimeError(
            "Manifest contains empty source split values."
        )

    # --------------------------------------------------------
    # Validate SHA-256 hashes
    # --------------------------------------------------------

    if manifest["sha256"].isna().any():
        raise RuntimeError(
            "Manifest contains missing SHA-256 hashes."
        )

    invalid_hashes = manifest[
        ~manifest["sha256"].astype(str).str.fullmatch(
            r"[0-9a-fA-F]{64}"
        )
    ]

    if not invalid_hashes.empty:
        raise RuntimeError(
            "Manifest contains invalid SHA-256 hashes:\n"
            f"{len(invalid_hashes)} invalid records."
        )

    # --------------------------------------------------------
    # Validate project splits
    # --------------------------------------------------------

    found_splits = set(
        manifest["project_split"].unique()
    )

    if found_splits != EXPECTED_PROJECT_SPLITS:
        raise RuntimeError(
            "Manifest project splits are invalid.\n"
            f"Expected: {EXPECTED_PROJECT_SPLITS}\n"
            f"Found: {found_splits}"
        )


# ============================================================
# DATASET CREATION
# ============================================================

def create_datasets(
    manifest_path,
    image_root,
):
    """
    Create train, validation and test datasets from the
    project manifest.
    """

    manifest_path = Path(manifest_path)
    image_root = Path(image_root)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist:\n{manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    validate_manifest(manifest)

    # --------------------------------------------------------
    # Split manifest
    # --------------------------------------------------------

    train_df = manifest[
        manifest["project_split"] == "train"
    ]

    val_df = manifest[
        manifest["project_split"] == "val"
    ]

    test_df = manifest[
        manifest["project_split"] == "test"
    ]

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform = get_train_transform()

    val_transform = get_val_transform()

    test_transform = get_test_transform()

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = PlantVillageTomatoDataset(
        dataframe=train_df,
        image_root=image_root,
        transform=train_transform,
    )

    val_dataset = PlantVillageTomatoDataset(
        dataframe=val_df,
        image_root=image_root,
        transform=val_transform,
    )

    test_dataset = PlantVillageTomatoDataset(
        dataframe=test_df,
        image_root=image_root,
        transform=test_transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=0,
    generator=None,
    worker_init_fn=None,
):
    """
    Create PyTorch DataLoaders.

    Train:
        shuffled

    Validation/Test:
        deterministic ordering
    """

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def get_class_counts(dataset):
    """
    Return class counts for a dataset.

    This function should be used with the TRAINING dataset
    when calculating training imbalance statistics.
    """

    counts = Counter(
        int(label)
        for label in dataset.dataframe["class_index"]
    )

    return {
        class_index: counts.get(class_index, 0)
        for class_index in range(len(CLASS_NAMES))
    }


# ============================================================
# MODERATED CLASS WEIGHTS
# ============================================================

def get_class_weights(dataset):
    """
    Calculate moderated inverse-frequency class weights.

    Square-root inverse frequency reduces the influence of
    the majority/minority ratio without allowing rare classes
    to dominate the loss.

    IMPORTANT:
        Weights are calculated ONLY from the training dataset.
    """

    counts = get_class_counts(dataset)

    weights = []

    for class_index in range(len(CLASS_NAMES)):

        count = counts[class_index]

        if count == 0:
            raise RuntimeError(
                f"Class index {class_index} has "
                f"zero training samples."
            )

        weight = 1.0 / (count ** 0.5)

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    # Normalize so mean class weight = 1.
    weights = weights / weights.mean()

    return weights