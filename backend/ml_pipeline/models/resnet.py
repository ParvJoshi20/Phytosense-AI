
"""
PhytoSense AI
ResNet50 model definition.

Responsible only for constructing and smoke-testing ResNet50.
"""

import torch
import timm

from config import NUM_CLASSES


MODEL_NAME = "resnet50"
MODEL_DISPLAY_NAME = "ResNet50"


def create_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """Create a ResNet50 classifier for the configured classes."""
    return timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_model_name():
    """Return the model architecture identifier."""
    return MODEL_NAME


def get_num_parameters(model):
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_model(
    num_classes=NUM_CLASSES,
    input_size=224,
    device=None,
):
    """Run a small forward-pass smoke test without training."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = create_model(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model.eval()

    dummy_input = torch.randn(
        2, 3, input_size, input_size,
        device=device,
    )

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (2, num_classes)

    assert tuple(output.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(output.shape)}"
    )
    assert torch.isfinite(output).all(), (
        "Model output contains NaN or Inf."
    )

    return model, output


if __name__ == "__main__":
    model, output = test_model()

    print(f"Model: {get_model_name()}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Trainable parameters: {get_num_parameters(model):,}")
    print("ResNet50 smoke test passed.")

"""
PhytoSense AI
ResNet50 model definition.

Responsible only for constructing and smoke-testing ResNet50.
"""

import torch
import timm

from config import NUM_CLASSES


MODEL_NAME = "resnet50"
MODEL_DISPLAY_NAME = "ResNet50"


def create_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """Create a ResNet50 classifier for the configured classes."""
    return timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_model_name():
    """Return the model architecture identifier."""
    return MODEL_NAME


def get_num_parameters(model):
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_model(
    num_classes=NUM_CLASSES,
    input_size=224,
    device=None,
):
    """Run a small forward-pass smoke test without training."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = create_model(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model.eval()

    dummy_input = torch.randn(
        2, 3, input_size, input_size,
        device=device,
    )

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (2, num_classes)

    assert tuple(output.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(output.shape)}"
    )
    assert torch.isfinite(output).all(), (
        "Model output contains NaN or Inf."
    )

    return model, output


if __name__ == "__main__":
    model, output = test_model()

    print(f"Model: {get_model_name()}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Trainable parameters: {get_num_parameters(model):,}")
    print("ResNet50 smoke test passed.")
