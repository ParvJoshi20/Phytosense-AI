"""
PhytoSense AI
Image preprocessing and augmentation pipeline.

Training:
    Random augmentation is applied to improve generalization.

Validation / Test:
    Deterministic preprocessing is used to ensure reproducible evaluation.

Normalization:
    ImageNet statistics are used because the planned model uses
    pretrained ImageNet weights.
"""

from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# TRAINING TRANSFORM
# ============================================================

def get_train_transform():
    """
    Preprocessing and augmentation used for training images.

    Random transformations are intentionally applied only
    to the training split.
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0),
            ratio=(0.90, 1.10),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5,
        ),

        transforms.RandomRotation(
            degrees=15,
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

def get_val_transform():
    """
    Deterministic preprocessing for validation images.
    """

    return transforms.Compose([
        transforms.Resize(
            256,
        ),

        transforms.CenterCrop(
            IMAGE_SIZE,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


# ============================================================
# TEST TRANSFORM
# ============================================================

def get_test_transform():
    """
    Deterministic preprocessing for test images.

    This must not contain random augmentation.
    """

    return transforms.Compose([
        transforms.Resize(
            256,
        ),

        transforms.CenterCrop(
            IMAGE_SIZE,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])