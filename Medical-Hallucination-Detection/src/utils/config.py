"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]


def load_config(config_path: str | Path) -> Config:
    """Load a YAML configuration file using ``yaml.safe_load``.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty or not a mapping.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError("Top-level YAML configuration must be a mapping.")

    return config
