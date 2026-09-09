from pathlib import Path
from collections import Counter

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from timm.data import create_transform


# ============================================================
# CONFIGURATION
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


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transform(data_config):
    """
    Build training preprocessing from the pretrained timm
    model's data configuration.

    The timm configuration supplies:
    - input size
    - interpolation
    - mean
    - standard deviation

    Training augmentation remains stochastic.
    """

    return create_transform(
        input_size=data_config["input_size"],
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        is_training=True,
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.2,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.0,
    )


def get_val_transform(data_config):
    """
    Deterministic preprocessing for validation and test.
    """

    return create_transform(
        input_size=data_config["input_size"],
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        is_training=False,
    )


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset_classes(
    train_dataset,
    val_dataset,
    test_dataset,
):
    """
    Validate that train, validation, and test contain exactly
    the expected PlantVillage tomato classes and use the same
    class-to-index mapping.
    """

    expected_class_set = set(EXPECTED_CLASSES)

    datasets_to_check = [
        ("training", train_dataset),
        ("validation", val_dataset),
        ("test", test_dataset),
    ]

    for split_name, dataset in datasets_to_check:

        if set(dataset.classes) != expected_class_set:

            raise RuntimeError(
                f"{split_name.capitalize()} dataset classes do not "
                f"match the expected PlantVillage tomato classes.\n\n"
                f"Expected:\n{EXPECTED_CLASSES}\n\n"
                f"Found:\n{dataset.classes}"
            )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:

        raise RuntimeError(
            "Train and validation class mappings do not match.\n\n"
            f"Train mapping:\n{train_dataset.class_to_idx}\n\n"
            f"Validation mapping:\n{val_dataset.class_to_idx}"
        )

    if train_dataset.class_to_idx != test_dataset.class_to_idx:

        raise RuntimeError(
            "Train and test class mappings do not match.\n\n"
            f"Train mapping:\n{train_dataset.class_to_idx}\n\n"
            f"Test mapping:\n{test_dataset.class_to_idx}"
        )

    if len(train_dataset.classes) != len(EXPECTED_CLASSES):

        raise RuntimeError(
            f"Expected {len(EXPECTED_CLASSES)} classes, "
            f"but found {len(train_dataset.classes)}."
        )


# ============================================================
# DATASETS
# ============================================================

def create_datasets(
    data_dir,
    data_config,
    train_transform=None,
    val_transform=None,
):

    data_dir = Path(data_dir)

    if not data_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{data_dir}"
        )

    if train_transform is None:
        train_transform = get_train_transform(data_config)

    if val_transform is None:
        val_transform = get_val_transform(data_config)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for split_name, split_dir in [
        ("train", train_dir),
        ("validation", val_dir),
        ("test", test_dir),
    ]:

        if not split_dir.exists():

            raise FileNotFoundError(
                f"{split_name.capitalize()} dataset directory "
                f"does not exist:\n{split_dir}"
            )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform,
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=val_transform,
    )

    validate_dataset_classes(
        train_dataset,
        val_dataset,
        test_dataset,
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
    batch_size=16,
    num_workers=0,
    generator=None,
    worker_init_fn=None,
):

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
# CLASS WEIGHTS
# ============================================================

def get_class_weights(dataset):
    """
    Calculate moderated inverse-frequency class weights.

    Square-root inverse frequency reduces the influence of the
    majority/minority ratio without allowing rare classes to
    dominate the loss.

    Weights are calculated ONLY from the training dataset.
    """

    counts = Counter(dataset.targets)

    total = len(dataset)
    num_classes = len(dataset.classes)

    weights = []

    for class_index in range(num_classes):

        count = counts[class_index]

        if count == 0:

            raise RuntimeError(
                f"Class index {class_index} contains "
                f"zero training samples."
            )

        weight = 1.0 / (count ** 0.5)

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    # Normalize so the average class weight is 1.
    weights = weights / weights.mean()

    return weights