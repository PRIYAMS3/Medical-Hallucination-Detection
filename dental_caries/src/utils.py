import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_path(path_value: str, workspace_root: str | None = None) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    root = Path(workspace_root) if workspace_root else Path.cwd()
    return str((root / p).resolve())


def save_json(path: str, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

