from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from backend.ml_pipeline.models.efficientnetv2 import create_model
from backend.ml_pipeline.preprocessing.transforms import get_val_transform


TARGET_LAYER = "blocks.5.7.conv_pwl"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_module = dict(model.named_modules())[target_layer]

        target_module.register_forward_hook(self._save_activation)
        target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)

        output = self.model(image_tensor)

        if class_index is None:
            class_index = output.argmax(dim=1).item()

        score = output[:, class_index]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_index, output.detach()


def load_model(checkpoint_path, device):
    model = create_model(
        num_classes=8,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


def create_overlay(original_image, cam):
    original = np.array(original_image.convert("RGB"))

    height, width = original.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height),
    )

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0,
    )

    return overlay


def generate_gradcam(
    image_path,
    checkpoint_path,
    output_path,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(
        checkpoint_path,
        device,
    )

    transform = get_val_transform()

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(
        model,
        TARGET_LAYER,
    )

    cam, class_index, output = gradcam.generate(
        image_tensor
    )

    overlay = create_overlay(
        image,
        cam,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(overlay).save(
        output_path
    )

    probabilities = torch.softmax(
        output,
        dim=1,
    )[0]

    return {
        "predicted_class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "output_path": str(output_path),
        "target_layer": TARGET_LAYER,
    }
