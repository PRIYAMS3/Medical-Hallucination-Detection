"""Dataset engineering utilities for the MedHallu benchmark.

This module downloads the MedHallu dataset from Hugging Face, validates its
schema, performs basic data quality checks, reports dataset statistics, and
saves an un-tokenized processed CSV file for downstream analysis.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

from src.utils.config import Config


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetValidationResult:
    """Summary of dataset validation and data quality checks."""

    missing_columns: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    duplicate_questions: int


class MedHalluDataset:
    """Download, validate, inspect, and persist the MedHallu dataset.

    Args:
        dataset_name: Hugging Face dataset repository name.
        subset: Hugging Face dataset configuration, such as ``pqa_labeled``.
        split: Dataset split to load, such as ``train``.
        processed_data_dir: Directory where processed CSV files are saved.
        processed_file_name: Name of the processed CSV output file.
        required_columns: Columns expected in the loaded dataset.
        question_column: Column containing medical questions.
        logger: Optional logger instance.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str,
        split: str,
        processed_data_dir: str | Path,
        processed_file_name: str,
        required_columns: list[str],
        question_column: str = "Question",
        logger: logging.Logger | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_file_name = processed_file_name
        self.required_columns = required_columns
        self.question_column = question_column
        self.logger = logger or LOGGER
        self.dataset: Dataset | None = None
        self.dataframe: pd.DataFrame | None = None

    @classmethod
    def from_config(cls, config: Config) -> "MedHalluDataset":
        """Create a dataset wrapper from project configuration.

        Args:
            config: Loaded project configuration.

        Returns:
            Configured ``MedHalluDataset`` instance.
        """
        dataset_config = config["dataset"]
        return cls(
            dataset_name=dataset_config["name"],
            subset=dataset_config.get("subset", "pqa_labeled"),
            split=dataset_config.get("split", "train"),
            processed_data_dir=dataset_config.get(
                "processed_data_dir", "data/processed"
            ),
            processed_file_name=dataset_config.get(
                "processed_file_name", "medhallu_processed.csv"
            ),
            required_columns=list(dataset_config.get("required_columns", [])),
            question_column=dataset_config.get("question_column", "Question"),
        )

    @property
    def processed_path(self) -> Path:
        """Return the path where the processed CSV will be stored."""
        return self.processed_data_dir / self.processed_file_name

    def download(self) -> pd.DataFrame:
        """Download the configured MedHallu split from Hugging Face.

        Returns:
            Loaded dataset as a pandas DataFrame.
        """
        self.logger.info(
            "Downloading dataset '%s' with subset '%s' and split '%s'.",
            self.dataset_name,
            self.subset,
            self.split,
        )
        loaded_dataset = load_dataset(self.dataset_name, self.subset, split=self.split)

        if not isinstance(loaded_dataset, Dataset):
            raise TypeError("Expected a Hugging Face Dataset for a single split.")

        self.dataset = loaded_dataset
        self.dataframe = loaded_dataset.to_pandas()
        self.logger.info("Downloaded %s samples.", len(self.dataframe))
        return self.dataframe

    def validate_schema(self, dataframe: pd.DataFrame | None = None) -> list[str]:
        """Validate that all required columns are present.

        Args:
            dataframe: Optional DataFrame to validate. Uses the downloaded
                DataFrame when omitted.

        Returns:
            List of missing required columns.

        Raises:
            ValueError: If required columns are missing.
        """
        df = self._resolve_dataframe(dataframe)
        missing_columns = [
            column for column in self.required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info("Schema validation passed with %s columns.", len(df.columns))
        return missing_columns

    def check_missing_values(
        self, dataframe: pd.DataFrame | None = None
    ) -> dict[str, int]:
        """Count missing values for each column.

        Args:
            dataframe: Optional DataFrame to inspect. Uses the downloaded
                DataFrame when omitted.

        Returns:
            Mapping of column names to missing-value counts.
        """
        df = self._resolve_dataframe(dataframe)
        missing_values = df.isna().sum().astype(int).to_dict()
        total_missing = sum(missing_values.values())
        self.logger.info("Detected %s total missing values.", total_missing)
        return missing_values

    def check_duplicate_samples(
        self, dataframe: pd.DataFrame | None = None
    ) -> tuple[int, int]:
        """Count duplicate rows and duplicate questions.

        Args:
            dataframe: Optional DataFrame to inspect. Uses the downloaded
                DataFrame when omitted.

        Returns:
            Tuple containing duplicate full-row count and duplicate-question
            count.
        """
        df = self._resolve_dataframe(dataframe)
        comparable_df = df.map(self._to_comparable_value)
        duplicate_rows = int(comparable_df.duplicated().sum())
        duplicate_questions = (
            int(df[self.question_column].duplicated().sum())
            if self.question_column in df.columns
            else 0
        )

        self.logger.info("Detected %s duplicate full rows.", duplicate_rows)
        self.logger.info("Detected %s duplicate questions.", duplicate_questions)
        return duplicate_rows, duplicate_questions

    def validate(self, dataframe: pd.DataFrame | None = None) -> DatasetValidationResult:
        """Run schema, missing-value, and duplicate checks.

        Args:
            dataframe: Optional DataFrame to validate. Uses the downloaded
                DataFrame when omitted.

        Returns:
            Structured validation summary.
        """
        df = self._resolve_dataframe(dataframe)
        missing_columns = self.validate_schema(df)
        missing_values = self.check_missing_values(df)
        duplicate_rows, duplicate_questions = self.check_duplicate_samples(df)

        return DatasetValidationResult(
            missing_columns=missing_columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            duplicate_questions=duplicate_questions,
        )

    def print_statistics(self, dataframe: pd.DataFrame | None = None) -> None:
        """Log dataset shape, columns, and simple column distributions.

        Args:
            dataframe: Optional DataFrame to summarize. Uses the downloaded
                DataFrame when omitted.
        """
        df = self._resolve_dataframe(dataframe)
        self.logger.info("Dataset size: %s rows x %s columns", *df.shape)
        self.logger.info("Dataset columns: %s", list(df.columns))

        for column in ("Difficulty Level", "Category of Hallucination"):
            if column in df.columns:
                distribution = df[column].value_counts(dropna=False).to_dict()
                self.logger.info("%s distribution: %s", column, distribution)

    def save_processed_csv(self, dataframe: pd.DataFrame | None = None) -> Path:
        """Save the un-tokenized dataset as a processed CSV file.

        Args:
            dataframe: Optional DataFrame to save. Uses the downloaded DataFrame
                when omitted.

        Returns:
            Path to the saved CSV file.
        """
        df = self._resolve_dataframe(dataframe)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        self.logger.info("Saved processed dataset to %s", self.processed_path)
        return self.processed_path

    def prepare(self) -> Path:
        """Download, validate, summarize, and save the configured dataset.

        Returns:
            Path to the saved processed CSV file.
        """
        dataframe = self.download()
        self.validate(dataframe)
        self.print_statistics(dataframe)
        return self.save_processed_csv(dataframe)

    def _resolve_dataframe(self, dataframe: pd.DataFrame | None) -> pd.DataFrame:
        """Return the provided DataFrame or the internally stored DataFrame."""
        resolved = dataframe if dataframe is not None else self.dataframe
        if resolved is None:
            raise RuntimeError("Dataset has not been downloaded yet.")
        return resolved

    @staticmethod
    def _to_comparable_value(value: Any) -> Any:
        """Convert array-like cell values into duplicate-checkable values."""
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, sort_keys=True)

        if hasattr(value, "tolist"):
            return json.dumps(value.tolist(), sort_keys=True)

        return value


def dataframe_from_csv(path: str | Path) -> pd.DataFrame:
    """Load a processed dataset CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    return pd.read_csv(path)


def config_value(config: Config, section: str, key: str, default: Any) -> Any:
    """Read a nested configuration value with a default fallback."""
    return config.get(section, {}).get(key, default)
