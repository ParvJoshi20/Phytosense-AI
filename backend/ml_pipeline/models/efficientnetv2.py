import timm
import torch.nn as nn


def create_model(num_classes=8, pretrained=True):
    model = timm.create_model(
        "tf_efficientnetv2_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True