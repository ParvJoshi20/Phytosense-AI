"""
PhytoSense AI
Model registry and ResNet50 smoke tests.
"""

import sys
from pathlib import Path

ML_PIPELINE_DIR = Path(__file__).resolve().parents[1]

if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))

import torch

from config import NUM_CLASSES
from models.registry import (
    list_models,
    get_model_display_name,
    create_model_by_name,
)


def main():
    print("=" * 65)
    print("PHYTOSENSE AI — MODEL REGISTRY TEST")
    print("=" * 65)

    models = list_models()
    print("Registered models:", models)

    assert "tf_efficientnetv2_b0" in models
    assert "resnet50" in models
    assert get_model_display_name("resnet50") == "ResNet50"

    print("\nCreating ResNet50 without pretrained weights...")

    model = create_model_by_name(
        model_name="resnet50",
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    images = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        outputs = model(images)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Device:", device)
    print("Output shape:", tuple(outputs.shape))
    print(f"Trainable parameters: {parameter_count:,}")

    assert tuple(outputs.shape) == (2, NUM_CLASSES)
    assert torch.isfinite(outputs).all()

    try:
        create_model_by_name("not_a_real_model")
    except ValueError:
        print("Unsupported-model validation passed.")
    else:
        raise AssertionError(
            "Registry should reject unsupported model identifiers."
        )

    print("\nMODEL REGISTRY TEST PASSED")
    print("No training was performed.")
    print("No dataset or test-set samples were loaded.")


if __name__ == "__main__":
    main()