
"""
PhytoSense AI
DenseNet121 model smoke test.

Checks that:
- DenseNet121 can be constructed from the model registry.
- Its output has the expected 10-class shape.
- Outputs contain no NaN or Inf values.

This is not a training or evaluation run.
Pretrained weights are disabled to avoid a download.
"""

import sys
from pathlib import Path

# Allow imports from backend/ml_pipeline when this file is run directly.
ML_PIPELINE_DIR = Path(__file__).resolve().parents[1]

if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))

import torch

from config import NUM_CLASSES
from models.registry import (
    create_model_by_name,
    get_num_parameters,
)


MODEL_NAME = "densenet121"
INPUT_SIZE = 224
BATCH_SIZE = 2


def main():
    print("=" * 65)
    print("PHYTOSENSE AI — DENSENET121 SMOKE TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU")

    print("\nCreating DenseNet121 without pretrained weights...")

    model = create_model_by_name(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    model = model.to(device)
    model.eval()

    images = torch.randn(
        BATCH_SIZE,
        3,
        INPUT_SIZE,
        INPUT_SIZE,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(images)

    expected_shape = (BATCH_SIZE, NUM_CLASSES)

    print("Model:", MODEL_NAME)
    print("Output shape:", tuple(outputs.shape))
    print(f"Trainable parameters: {get_num_parameters(model):,}")

    assert tuple(outputs.shape) == expected_shape, (
        f"Expected output shape {expected_shape}, "
        f"got {tuple(outputs.shape)}"
    )

    assert torch.isfinite(outputs).all().item(), (
        "Model output contains NaN or Inf values."
    )

    print("\nDENSENET121 SMOKE TEST PASSED")
    print("No dataset was loaded.")
    print("No training or test-set evaluation was performed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
