from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def setup_logging(log_path: str | Path) -> logging.Logger:
    ensure_parent(log_path)
    logger = logging.getLogger("inventory_optimization")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


def movement_class_from_demand(avg_daily_demand: float) -> str:
    if avg_daily_demand >= 3.0:
        return "fast_moving"
    if avg_daily_demand >= 0.5:
        return "slow_moving"
    return "dead_stock"
