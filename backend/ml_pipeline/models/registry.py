"""Model registry for the PhytoSense AI image-classification experiments."""

import timm

from config import NUM_CLASSES

# Keep the registry independent of the selected experiment so all architectures
# can be instantiated for smoke tests and model-comparison runs.
MODEL_REGISTRY = {
    "tf_efficientnetv2_b0": {
        "display_name": "EfficientNetV2-B0",
        "timm_name": "tf_efficientnetv2_b0",
    },
    "resnet50": {
        "display_name": "ResNet50",
        "timm_name": "resnet50",
    },
    "mobilenetv3_large_100": {
        "display_name": "MobileNetV3-Large",
        "timm_name": "mobilenetv3_large_100",
    },
    "densenet121": {
        "display_name": "DenseNet121",
        "timm_name": "densenet121",
    },
    "convnext_tiny": {
        "display_name": "ConvNeXt-Tiny",
        "timm_name": "convnext_tiny",
    },
}


def list_models():
    """Return supported architecture identifiers in comparison order."""
    return list(MODEL_REGISTRY)


def get_model_display_name(model_name):
    """Return the readable name for a registered architecture."""
    try:
        return MODEL_REGISTRY[model_name]["display_name"]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {list_models()}"
        ) from exc


def create_model_by_name(model_name, num_classes=NUM_CLASSES, pretrained=True):
    """Create a classifier using the registered timm architecture."""
    try:
        model_spec = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {list_models()}"
        ) from exc

    return timm.create_model(
        model_spec["timm_name"],
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_num_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
