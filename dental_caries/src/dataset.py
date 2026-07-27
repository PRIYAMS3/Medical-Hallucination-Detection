from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ManifestColumns:
    image_path: str = "image_path"
    label: str = "label"
    split: str = "split"


class CariesManifestDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        transform: Callable | None = None,
        image_col: str = "image_path",
        label_col: str = "label",
        grayscale_to_rgb: bool = True,
    ) -> None:
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.grayscale_to_rgb = grayscale_to_rgb

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = Path(row[self.image_col])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path)
        if self.grayscale_to_rgb:
            image = image.convert("RGB")
        label = int(row[self.label_col])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_manifest(manifest_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv_path)
    required = {"image_path", "label", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}. "
            "Required columns: image_path,label,split"
        )
    df["split"] = df["split"].astype(str).str.lower()
    for split_name in ("train", "val", "test"):
        if (df["split"] == split_name).sum() == 0:
            raise ValueError(f"Manifest has no rows for split='{split_name}'")
    return df

