
"""
PhytoSense AI
Model registry for supported classification architectures.
"""

from config import NUM_CLASSES

from models.efficientnetv2 import (
    create_model as create_efficientnetv2,
)
from models.resnet import (
    create_model as create_resnet50,
)


MODEL_REGISTRY = {
    "tf_efficientnetv2_b0": {
        "display_name": "EfficientNetV2-B0",
        "factory": create_efficientnetv2,
    },
    "resnet50": {
        "display_name": "ResNet50",
        "factory": create_resnet50,
    },
}


def list_models():
    """Return the identifiers of all registered models."""
    return list(MODEL_REGISTRY.keys())


def get_model_display_name(model_name):
    """Return a human-readable model name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {list_models()}"
        )

    return MODEL_REGISTRY[model_name]["display_name"]


def create_model_by_name(
    model_name,
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """Construct a registered model by architecture identifier."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {list_models()}"
        )

    factory = MODEL_REGISTRY[model_name]["factory"]

    return factory(
        num_classes=num_classes,
        pretrained=pretrained,
    )
