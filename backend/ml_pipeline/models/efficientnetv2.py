"""
PhytoSense AI
EfficientNetV2-B0 model definition.

This module is responsible only for constructing the
classification model used by the ML training pipeline.
"""

import torch
import timm

from config import (
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED,
)


def create_model(
    num_classes=NUM_CLASSES,
    pretrained=PRETRAINED,
):
    """
    Create the configured EfficientNetV2-B0 classifier.

    The architecture and class-count defaults come from the
    centralized experiment configuration so the model definition
    cannot silently drift from the training/evaluation pipeline.
    """
    return timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_model_name():
    """Return the configured model architecture name."""
    return MODEL_NAME


def get_num_parameters(model):
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_model(
    num_classes=NUM_CLASSES,
    input_size=224,
):
    """
    Perform a basic forward-pass smoke test.
    """
    model = create_model(
        num_classes=num_classes,
        pretrained=False,
    )
    model.eval()

    dummy_input = torch.randn(
        2,
        3,
        input_size,
        input_size,
    )

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (2, num_classes)

    assert output.shape == expected_shape, (
        f"Unexpected output shape: {output.shape}; "
        f"expected {expected_shape}"
    )

    assert torch.isfinite(output).all(), (
        "Model output contains NaN or Inf values."
    )

    return model, output


if __name__ == "__main__":
    print("=" * 60)
    print("PHYTOSENSE AI — MODEL SMOKE TEST")
    print("=" * 60)

    model, output = test_model()

    print(f"\nModel: {get_model_name()}")
    print(f"Output shape: {output.shape}")
    print(
        f"Trainable parameters: "
        f"{get_num_parameters(model):,}"
    )
    print("\n✓ Model forward-pass test passed.")
    print("=" * 60)