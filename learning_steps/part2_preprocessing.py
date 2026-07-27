from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


DATASET_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
TARGET_COL = "Result"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    report: dict[str, object] = {
        "dataset_path": str(DATASET_PATH),
        "shape_before": list(df.shape),
        "target_unique_before": sorted([int(v) for v in df[TARGET_COL].unique().tolist()]),
    }

    # Correct single-pass label mapping (-1 => 0, 1 => 1)
    y = df[TARGET_COL].map({-1: 0, 1: 1})
    if y.isnull().any():
        raise ValueError("Label mapping produced NaN values. Check target values.")

    x = df.drop(columns=[TARGET_COL]).astype(np.float32)
    y = y.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    report.update(
        {
            "shape_features": list(x.shape),
            "shape_target": int(y.shape[0]),
            "target_unique_after": sorted([int(v) for v in pd.Series(y).unique().tolist()]),
            "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
            "train_distribution": {
                "0": int((y_train == 0).sum()),
                "1": int((y_train == 1).sum()),
            },
            "test_distribution": {
                "0": int((y_test == 0).sum()),
                "1": int((y_test == 1).sum()),
            },
            "class_weights": {
                "0": float(class_weights[0]),
                "1": float(class_weights[1]),
            },
        }
    )

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part2_preprocessing_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
