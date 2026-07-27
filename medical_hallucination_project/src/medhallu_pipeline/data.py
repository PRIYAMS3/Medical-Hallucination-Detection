"""Dataset loading and normalization for MedHallu-style records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MEDHALLU_DIR = PROJECT_ROOT / "data" / "medhallu"


@dataclass(frozen=True)
class MedHalluRecord:
    """Normalized row used by the prototype pipeline."""

    sample_id: str
    question: str
    answer: str
    context: str
    label: str | None = None
    category: str | None = None
    difficulty: str | None = None


def load_medhallu(split: str = "pqa_labeled") -> pd.DataFrame:
    """Load MedHallu from Hugging Face and return a pandas DataFrame.

    MedHallu exposes `pqa_labeled` and `pqa_artificial` as dataset configs.
    Each config currently uses a standard `train` split.
    """

    local_path = LOCAL_MEDHALLU_DIR / f"{split}.parquet"
    partial_path = LOCAL_MEDHALLU_DIR / f"{split}.parquet.part"
    if local_path.exists():
        return pd.read_parquet(local_path)
    if partial_path.exists():
        return pd.read_parquet(partial_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install requirements.txt first."
        ) from exc

    dataset = load_dataset("UTAustin-AIHealth/MedHallu", split, split="train")
    return dataset.to_pandas()


def _first_existing(row: dict[str, Any], candidates: list[str], default: str = "") -> str:
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def normalize_record(row: dict[str, Any], index: int) -> MedHalluRecord:
    """Normalize likely MedHallu fields without assuming exact column names."""

    sample_id = _first_existing(row, ["id", "sample_id", "qid"], str(index))
    question = _first_existing(row, ["question", "query", "Question"])
    answer = _first_existing(
        row,
        [
            "answer",
            "hallucinated_answer",
            "generated_answer",
            "model_answer",
            "Answer",
        ],
    )
    context = _first_existing(
        row,
        ["context", "evidence", "passage", "reference", "Context", "pubmed_context"],
    )
    label = _first_existing(
        row,
        ["label", "hallucination_label", "is_hallucinated", "Label"],
        default="",
    )
    category = _first_existing(
        row,
        ["category", "Category of Hallucination", "hallucination_category"],
        default="",
    )
    difficulty = _first_existing(row, ["difficulty", "Difficulty"], default="")

    return MedHalluRecord(
        sample_id=sample_id,
        question=question,
        answer=answer,
        context=context,
        label=label or None,
        category=category or None,
        difficulty=difficulty or None,
    )


def normalize_dataframe(df: pd.DataFrame) -> list[MedHalluRecord]:
    """Convert a raw MedHallu dataframe into normalized records."""

    return [normalize_record(row, i) for i, row in enumerate(df.to_dict("records"))]
