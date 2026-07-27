from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import EfficientNetB0CBAM, build_model
from utils import ensure_dir, get_device, load_config, resolve_path


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            _ = grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, EfficientNetB0CBAM):
        return model.features[-1]
    if hasattr(model, "layer4"):  # resnet
        return model.layer4[-1]
    if hasattr(model, "features"):  # efficientnet
        return model.features[-1]
    raise ValueError("Could not infer target layer for Grad-CAM.")


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * image_rgb + 0.4 * heatmap).astype(np.uint8)
    return overlay


def main(config_path: str, image_path: str):
    config = load_config(config_path)
    device = get_device()
    out_dir = resolve_path(config["paths"]["gradcam_dir"])
    ensure_dir(out_dir)

    model = build_model(
        model_name=config["training"]["model_name"],
        num_classes=int(config["project"]["num_classes"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(resolve_path(config["paths"]["best_model_path"]), map_location=device))
    model.eval()

    image_size = int(config["data"]["image_size"])
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    pil_img = Image.open(image_path).convert("RGB")
    x = tfm(pil_img).unsqueeze(0).to(device)

    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(x)

    img_np = np.array(pil_img.resize((image_size, image_size)))
    overlay = overlay_heatmap(img_np, cam)

    stem = Path(image_path).stem
    out_path = Path(out_dir) / f"{stem}_gradcam.png"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.axis("off")
    plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Grad-CAM to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    main(args.config, args.image)

