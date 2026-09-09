"""
PhytoSense AI
Manifest-based PlantVillage Dataset Pipeline.

The CSV manifest is the single source of truth for the
train / validation / test split.

The original PlantVillage dataset is never modified.
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from preprocessing.transforms import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
)


# ============================================================
# EXPECTED CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(EXPECTED_CLASSES)
}


# ============================================================
# DATASET
# ============================================================

class PlantVillageTomatoDataset(Dataset):
    """
    PyTorch Dataset backed by the PlantVillage project manifest.
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
    """

    required_columns = {
        "image_path",
        "class_name",
        "class_index",
        "original_split",
        "project_split",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise RuntimeError(
            f"Manifest is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    if manifest.empty:
        raise RuntimeError("Manifest contains no records.")

    # --------------------------------------------------------
    # Validate classes
    # --------------------------------------------------------

    found_classes = set(manifest["class_name"].unique())
    expected_classes = set(EXPECTED_CLASSES)

    if found_classes != expected_classes:
        raise RuntimeError(
            "Manifest classes do not match the expected "
            "PlantVillage tomato classes.\n\n"
            f"Expected:\n{EXPECTED_CLASSES}\n\n"
            f"Found:\n{sorted(found_classes)}"
        )

    # --------------------------------------------------------
    # Validate class indices
    # --------------------------------------------------------

    for _, row in manifest.iterrows():

        class_name = row["class_name"]
        class_index = int(row["class_index"])

        expected_index = CLASS_TO_INDEX[class_name]

        if class_index != expected_index:
            raise RuntimeError(
                f"Invalid class mapping:\n"
                f"{class_name} → {class_index}\n"
                f"Expected → {expected_index}"
            )

    # --------------------------------------------------------
    # Validate project splits
    # --------------------------------------------------------

    expected_splits = {
        "train",
        "val",
        "test",
    }

    found_splits = set(
        manifest["project_split"].unique()
    )

    if found_splits != expected_splits:
        raise RuntimeError(
            "Manifest project splits are invalid.\n"
            f"Expected: {expected_splits}\n"
            f"Found: {found_splits}"
        )

    # --------------------------------------------------------
    # Validate duplicate image paths
    # --------------------------------------------------------

    duplicate_paths = manifest[
        manifest["image_path"].duplicated()
    ]

    if not duplicate_paths.empty:
        raise RuntimeError(
            f"Manifest contains "
            f"{len(duplicate_paths)} duplicate image paths."
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

    train_transform = get_train_transform(
        
    )

    val_transform = get_val_transform(
        
    )

    test_transform = get_test_transform(
        
    )

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
    Return training class counts.

    This function should be used with the TRAINING dataset
    when calculating imbalance statistics.
    """

    counts = Counter(
        int(label)
        for label in dataset.dataframe["class_index"]
    )

    return {
        class_index: counts.get(class_index, 0)
        for class_index in range(len(EXPECTED_CLASSES))
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

    for class_index in range(
        len(EXPECTED_CLASSES)
    ):

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