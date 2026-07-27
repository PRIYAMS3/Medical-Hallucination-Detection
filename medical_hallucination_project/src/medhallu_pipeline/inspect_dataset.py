"""Inspect MedHallu splits and fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medhallu_pipeline.data import load_medhallu


def make_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="pqa_labeled")
    parser.add_argument("--out", default="outputs/dataset_inspection.json")
    args = parser.parse_args()

    df = load_medhallu(split=args.split)
    report = {
        "split": args.split,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sample_row": make_json_safe(df.head(1).to_dict("records")),
        "value_counts": {},
    }

    for column in df.columns:
        normalized = column.lower()
        if any(key in normalized for key in ["label", "category", "difficulty"]):
            report["value_counts"][column] = (
                df[column].astype(str).value_counts(dropna=False).head(30).to_dict()
            )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
