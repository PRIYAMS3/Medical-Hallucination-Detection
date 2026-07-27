"""Download MedHallu parquet files without using the Hugging Face cache manager."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from medhallu_pipeline.data import LOCAL_MEDHALLU_DIR


URLS = {
    "pqa_labeled": "https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu/resolve/main/pqa_labeled/train-00000-of-00001.parquet",
    "pqa_artificial": "https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu/resolve/main/pqa_artificial/train-00000-of-00001.parquet",
}


def download(split: str) -> Path:
    if split not in URLS:
        raise ValueError(f"Unknown split/config {split!r}. Choose one of: {sorted(URLS)}")

    LOCAL_MEDHALLU_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOCAL_MEDHALLU_DIR / f"{split}.parquet"
    temp_path = output_path.with_suffix(".parquet.part")

    response = requests.get(URLS[split], stream=True, timeout=60, proxies={})
    response.raise_for_status()

    with temp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    temp_path.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(URLS), default="pqa_labeled")
    args = parser.parse_args()
    path = download(args.split)
    print(path)


if __name__ == "__main__":
    main()

