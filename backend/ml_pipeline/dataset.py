from pathlib import Path
from collections import Counter

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

CLASSES = [
    "bacterial_spot",
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "mosaic_virus",
    "septoria_leaf_spot",
    "yellow_leaf_curl_virus",
]


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# DATASETS
# ============================================================

def create_datasets(
    data_dir,
    train_transform=None,
    val_transform=None,
):

    data_dir = Path(data_dir)

    if train_transform is None:
        train_transform = get_train_transform()

    if val_transform is None:
        val_transform = get_val_transform()

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=val_transform,
    )

    test_dataset = datasets.ImageFolder(
        data_dir / "test",
        transform=val_transform,
    )

    return train_dataset, val_dataset, test_dataset


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=16,
    num_workers=0,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_class_weights(dataset):

    counts = Counter(dataset.targets)

    total = len(dataset)
    num_classes = len(dataset.classes)

    weights = []

    for class_index in range(num_classes):

        count = counts[class_index]

        weight = total / (num_classes * count)

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )