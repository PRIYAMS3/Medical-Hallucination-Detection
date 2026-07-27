from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CariesManifestDataset, load_manifest
from models import build_model
from train import compute_metrics
from utils import get_device, load_config, resolve_path, save_json


def main(config_path: str):
    config = load_config(config_path)
    device = get_device()

    manifest_path = resolve_path(config["paths"]["manifest_csv"])
    best_model_path = resolve_path(config["paths"]["best_model_path"])
    metrics_json = resolve_path(config["paths"]["metrics_json"])

    df = load_manifest(manifest_path)
    test_df = df[df["split"] == "test"].copy()

    image_size = int(config["data"]["image_size"])
    tfm = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    ds = CariesManifestDataset(
        test_df,
        transform=tfm,
        grayscale_to_rgb=bool(config["data"].get("grayscale_to_rgb", True)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(config["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
    )

    model = build_model(
        model_name=config["training"]["model_name"],
        num_classes=int(config["project"]["num_classes"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    threshold = float(config["evaluation"]["threshold"])
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    if y_prob.shape[1] == 2:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
        auc_for_report = roc_auc_score(y_true, y_prob[:, 1])
    else:
        y_pred = np.argmax(y_prob, axis=1)
        auc_for_report = roc_auc_score(y_true, y_prob, multi_class="ovr")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    payload = {
        "auc": float(auc_for_report),
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    save_json(metrics_json, payload)
    print(f"Saved evaluation to: {metrics_json}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)

