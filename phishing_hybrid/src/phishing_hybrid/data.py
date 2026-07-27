from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def _parse_arff(path: Path) -> pd.DataFrame:
    attribute_pattern = re.compile(r"^@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))", re.IGNORECASE)
    attributes: list[str] = []
    rows: list[list[str]] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            low = line.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    match = attribute_pattern.match(line)
                    if not match:
                        continue
                    name = next(group for group in match.groups() if group is not None)
                    attributes.append(name)
                elif low.startswith("@data"):
                    in_data = True
                continue

            if line.startswith("{"):
                raise ValueError("Sparse ARFF format is not supported by this parser.")
            rows.append([token.strip() for token in line.split(",")])

    if not attributes:
        raise ValueError(f"No attributes found in ARFF file: {path}")
    if not rows:
        raise ValueError(f"No data rows found in ARFF file: {path}")

    df = pd.DataFrame(rows, columns=attributes)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_dataset(path: str | Path, target_column: str, fillna_value: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".arff":
        df = _parse_arff(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from dataset columns.")

    df = df.replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(fillna_value)

    y_raw = df[target_column].astype(int)
    classes = set(np.unique(y_raw))
    if classes.issubset({-1, 1}):
        y = y_raw.map({-1: 0, 1: 1}).astype(int)
    elif classes.issubset({0, 1}):
        y = y_raw.astype(int)
    else:
        raise ValueError(
            f"Unsupported label values in '{target_column}': {sorted(classes)}. Expected [-1,1] or [0,1]."
        )

    x = df.drop(columns=[target_column]).astype(np.float32)
    return x, y


def split_dataset(
    x: pd.DataFrame, y: pd.Series, test_size: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=test_size,
        stratify=y.values,
        random_state=random_seed,
    )
    return x_train, x_test, y_train, y_test


def get_class_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    return compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
