from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from phishing_hybrid.models import build_model


class PhishingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = PhishingDataset(x_train, y_train)
    test_ds = PhishingDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: np.ndarray | None = None,
) -> tuple[nn.Module, list[float]]:
    model = model.to(device)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        losses.append(epoch_loss)

    return model, losses


def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_true.extend(y_batch.numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_prob.extend(probs[:, 1].cpu().numpy().tolist())

    return np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    y_true, y_pred, y_prob = predict_logits(model, loader, device)
    return evaluate_predictions(y_true, y_pred, y_prob)


def save_checkpoint(
    output_path: Path,
    model_name: str,
    model: nn.Module,
    input_dim: int,
    feature_names: list[str],
    metrics: dict[str, Any],
) -> None:
    payload = {
        "model_name": model_name,
        "input_dim": input_dim,
        "feature_names": feature_names,
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt["model_name"]
    input_dim = int(ckpt["input_dim"])
    model = build_model(model_name, input_dim)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    metadata = {
        "model_name": model_name,
        "input_dim": input_dim,
        "feature_names": ckpt.get("feature_names", []),
        "metrics": ckpt.get("metrics", {}),
    }
    return model, metadata


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
