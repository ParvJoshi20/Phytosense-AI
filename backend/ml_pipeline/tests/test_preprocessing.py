"""
PhytoSense AI
Preprocessing smoke test.

This test checks that the preprocessing transforms for training, validation, and testing are working correctly. 

Real PlantVillage image
        ↓
       PIL
        ↓
   RGB conversion
        ↓
 ┌──────┼──────┐
 ↓      ↓      ↓
Train   Val   Test
 ↓      ↓      ↓
Tensor Tensor Tensor
 ↓      ↓      ↓
3 × 224 × 224

The test must pass before model training begins.
"""

from PIL import Image

from backend.ml_pipeline.dataset.transforms import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
)


IMAGE_PATH = r"D:\Code\_Archieve\Minor Project\Datasets\PlantVillage-Dataset\raw\color\Tomato___healthy"


def find_first_image():
    import os

    for filename in os.listdir(IMAGE_PATH):
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            return os.path.join(
                IMAGE_PATH,
                filename,
            )

    raise FileNotFoundError(
        f"No image found in:\n{IMAGE_PATH}"
    )


def main():

    image_path = find_first_image()

    print("=" * 60)
    print("PHYTOSENSE AI — PREPROCESSING TEST")
    print("=" * 60)

    print("\nImage:")
    print(image_path)

    image = Image.open(image_path).convert("RGB")

    print("\nOriginal image:")
    print(f"Size: {image.size}")
    print(f"Mode: {image.mode}")

    train_transform = get_train_transform()
    val_transform = get_val_transform()
    test_transform = get_test_transform()

    train_image = train_transform(image)
    val_image = val_transform(image)
    test_image = test_transform(image)

    print("\nTransformed tensors:")
    print(
        f"Train: shape={train_image.shape}, "
        f"dtype={train_image.dtype}"
    )

    print(
        f"Val:   shape={val_image.shape}, "
        f"dtype={val_image.dtype}"
    )

    print(
        f"Test:  shape={test_image.shape}, "
        f"dtype={test_image.dtype}"
    )

    assert train_image.shape == (
        3,
        224,
        224,
    )

    assert val_image.shape == (
        3,
        224,
        224,
    )

    assert test_image.shape == (
        3,
        224,
        224,
    )

    print("\n✓ All preprocessing checks passed.")


if __name__ == "__main__":
    main()