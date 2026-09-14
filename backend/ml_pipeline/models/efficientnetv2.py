"""
PhytoSense AI
EfficientNetV2-B0 Model Definition.

This module is responsible only for constructing the
classification model used by the ML training pipeline.

Architecture:

    Image
      ↓
    EfficientNetV2-B0
      ↓
    Classification Head
      ↓
    10 Tomato Classes
"""

import torch
import torch.nn as nn
import timm


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "tf_efficientnetv2_b0"

DEFAULT_NUM_CLASSES = 10

DEFAULT_PRETRAINED = True


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(
    num_classes=DEFAULT_NUM_CLASSES,
    pretrained=DEFAULT_PRETRAINED,
):
    """
    Create the EfficientNetV2-B0 classification model.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether to initialize the backbone with ImageNet
        pretrained weights.

    Returns
    -------
    torch.nn.Module
        Configured EfficientNetV2-B0 model.
    """

    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_name():
    """
    Return the model architecture name.
    """

    return MODEL_NAME


def get_num_parameters(model):
    """
    Return the total number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# MODEL SMOKE TEST
# ============================================================

def test_model(
    num_classes=DEFAULT_NUM_CLASSES,
    input_size=224,
):
    """
    Perform a basic forward-pass test.

    This verifies that:

        Input
          ↓
        Model
          ↓
        Expected output

    works correctly before training begins.
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

    expected_shape = (
        2,
        num_classes,
    )

    assert output.shape == expected_shape, (
        f"Unexpected output shape: "
        f"{output.shape}; "
        f"expected {expected_shape}"
    )

    assert torch.isfinite(output).all(), (
        "Model output contains NaN or Inf values."
    )

    return model, output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PHYTOSENSE AI — MODEL SMOKE TEST")
    print("=" * 60)

    model, output = test_model()

    print(
        f"\nModel: {get_model_name()}"
    )

    print(
        f"Output shape: {output.shape}"
    )

    print(
        f"Trainable parameters: "
        f"{get_num_parameters(model):,}"
    )

    print(
        "\n✓ Model forward-pass test passed."
    )

    print("=" * 60)