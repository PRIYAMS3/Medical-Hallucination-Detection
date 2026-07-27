from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def gather_class_images(class_dir: Path, label: int):
    rows = []
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            rows.append({"image_path": str(path.resolve()), "label": int(label)})
    return rows


def main(data_root: str, out_csv: str, val_size: float, test_size: float, seed: int):
    root = Path(data_root)
    non_caries_dir = root / "non_caries"
    caries_dir = root / "caries_or_deep_caries"
    if not non_caries_dir.exists() or not caries_dir.exists():
        raise FileNotFoundError(
            "Expected class folders:\n"
            f"- {non_caries_dir}\n"
            f"- {caries_dir}"
        )

    rows = gather_class_images(non_caries_dir, label=0) + gather_class_images(caries_dir, label=1)
    if len(rows) == 0:
        raise ValueError("No images found in class folders.")

    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df["label"],
    )
    val_ratio_in_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_ratio_in_temp,
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    out_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved manifest: {out_path.resolve()}")
    print(out_df["split"].value_counts().to_dict())
    print(out_df["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Folder with non_caries/ and caries_or_deep_caries/")
    parser.add_argument("--out_csv", required=True, help="Output manifest CSV path")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.data_root, args.out_csv, args.val_size, args.test_size, args.seed)

