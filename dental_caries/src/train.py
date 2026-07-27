from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CariesManifestDataset, load_manifest
from models import build_model
from utils import ensure_dir, get_device, load_config, resolve_path, save_json, set_seed


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


def build_transforms(config: dict):
    image_size = config["data"]["image_size"]
    aug = config["augmentation"]

    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
            transforms.RandomAffine(
                degrees=aug["rotation_degrees"],
                scale=(aug["zoom_scale_min"], aug["zoom_scale_max"]),
            ),
            transforms.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
            transforms.ToTensor(),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return train_tfms, eval_tfms


def make_loaders(config: dict):
    manifest_path = resolve_path(config["paths"]["manifest_csv"])
    df = load_manifest(manifest_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_tfms, eval_tfms = build_transforms(config)
    grayscale_to_rgb = bool(config["data"].get("grayscale_to_rgb", True))

    train_ds = CariesManifestDataset(train_df, transform=train_tfms, grayscale_to_rgb=grayscale_to_rgb)
    val_ds = CariesManifestDataset(val_df, transform=eval_tfms, grayscale_to_rgb=grayscale_to_rgb)
    test_ds = CariesManifestDataset(test_df, transform=eval_tfms, grayscale_to_rgb=grayscale_to_rgb)

    num_workers = int(config["data"]["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["data"]["train_batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_counts = train_df["label"].value_counts().sort_index()
    weights = 1.0 / class_counts.values.astype(np.float32)
    weights = weights / weights.sum() * len(weights)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    return train_loader, val_loader, test_loader, class_weights


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        pos_prob = y_prob if y_prob.ndim == 1 else y_prob[:, 0]
        y_pred = (pos_prob >= threshold).astype(int)
        auc = roc_auc_score(y_true, pos_prob)
        pr_auc = average_precision_score(y_true, pos_prob)
    else:
        if y_prob.shape[1] == 2:
            pos_prob = y_prob[:, 1]
            y_pred = (pos_prob >= threshold).astype(int)
            auc = roc_auc_score(y_true, pos_prob)
            pr_auc = average_precision_score(y_true, pos_prob)
        else:
            y_pred = np.argmax(y_prob, axis=1)
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            pr_auc = float("nan")

    f1 = f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro")
    precision = precision_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro")
    recall = recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro")
    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
    }


def run_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().numpy())
        losses.append(loss.item())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main(config_path: str):
    config = load_config(config_path)
    set_seed(int(config["project"]["random_seed"]))
    device = get_device()

    output_dir = resolve_path(config["paths"]["output_dir"])
    logs_dir = resolve_path(config["paths"]["logs_dir"])
    best_model_path = resolve_path(config["paths"]["best_model_path"])
    ensure_dir(output_dir)
    ensure_dir(logs_dir)

    train_loader, val_loader, test_loader, class_weights = make_loaders(config)
    class_weights = class_weights.to(device)

    model = build_model(
        model_name=config["training"]["model_name"],
        num_classes=int(config["project"]["num_classes"]),
        pretrained=bool(config["training"]["pretrained"]),
    ).to(device)

    if bool(config["training"]["use_focal_loss"]):
        criterion = FocalLoss(
            gamma=float(config["training"]["focal_gamma"]),
            alpha=class_weights,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
    )

    best_auc = -1.0
    best_state = None
    patience = int(config["training"]["early_stopping_patience"])
    wait = 0
    history = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step()

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, best_model_path)
    print(f"Saved best model to: {best_model_path}")

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, device, criterion, optimizer=None)

    summary = {
        "project": config["project"]["name"],
        "device": str(device),
        "best_val_auc": best_auc,
        "test_metrics": test_metrics,
        "history": history,
        "model_name": config["training"]["model_name"],
    }
    save_json(str(Path(logs_dir) / "training_summary.json"), summary)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)

