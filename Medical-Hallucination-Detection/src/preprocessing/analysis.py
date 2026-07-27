"""Dataset analysis report generation for MedHallu.

The report intentionally avoids tokenization, feature engineering, and model
training. It provides only descriptive dataset engineering diagnostics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.preprocessing.dataset import MedHalluDataset, dataframe_from_csv
from src.utils.config import Config, load_config
from src.utils.logging import setup_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset analysis."""
    parser = argparse.ArgumentParser(description="Generate a MedHallu dataset report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload and overwrite the processed dataset CSV before analysis.",
    )
    return parser.parse_args()


def initialize_logging(config: Config) -> logging.Logger:
    """Initialize logging from configuration values."""
    logging_config = config.get("logging", {})
    paths_config = config.get("paths", {})
    return setup_logging(
        log_dir=paths_config.get("logs_dir", "outputs/logs"),
        file_name="dataset_analysis.log",
        level=logging_config.get("level", "INFO"),
        log_format=logging_config.get(
            "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
        date_format=logging_config.get("date_format", "%Y-%m-%d %H:%M:%S"),
    )


def length_statistics(series: pd.Series) -> dict[str, float]:
    """Compute simple whitespace-delimited text length statistics.

    Args:
        series: Text column to analyze.

    Returns:
        Dictionary of descriptive length statistics.
    """
    lengths = series.fillna("").astype(str).str.split().str.len()
    return {
        "min": float(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "max": float(lengths.max()),
        "std": float(lengths.std(ddof=0)),
    }


def format_distribution(series: pd.Series) -> str:
    """Format a categorical distribution as a markdown table."""
    counts = series.value_counts(dropna=False)
    lines = ["| Value | Count | Percentage |", "|---|---:|---:|"]
    total = int(counts.sum())

    for value, count in counts.items():
        percentage = (int(count) / total * 100) if total else 0.0
        lines.append(f"| {value} | {int(count)} | {percentage:.2f}% |")

    return "\n".join(lines)


def format_length_statistics(title: str, statistics: dict[str, float]) -> str:
    """Format length statistics as a markdown section."""
    lines = [
        f"### {title}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {metric} | {value:.2f} |" for metric, value in statistics.items())
    return "\n".join(lines)


def load_or_prepare_dataset(config: Config, refresh: bool) -> pd.DataFrame:
    """Load the processed dataset CSV, downloading it first when needed."""
    medhallu_dataset = MedHalluDataset.from_config(config)
    processed_path = medhallu_dataset.processed_path

    if refresh or not processed_path.exists():
        LOGGER.info("Preparing processed dataset CSV.")
        medhallu_dataset.prepare()
    else:
        LOGGER.info("Using existing processed dataset CSV at %s", processed_path)

    dataframe = dataframe_from_csv(processed_path)
    medhallu_dataset.validate(dataframe)
    medhallu_dataset.print_statistics(dataframe)
    return dataframe


def generate_markdown_report(dataframe: pd.DataFrame, config: Config) -> str:
    """Generate a markdown dataset analysis report.

    Args:
        dataframe: Processed MedHallu DataFrame.
        config: Loaded project configuration.

    Returns:
        Markdown report text.
    """
    dataset_config = config["dataset"]
    question_column = dataset_config.get("question_column", "Question")
    answer_columns = dataset_config.get(
        "answer_columns", ["Ground Truth", "Hallucinated Answer"]
    )
    label_column = dataset_config.get("label_column", "Difficulty Level")
    hallucination_type_column = dataset_config.get(
        "hallucination_type_column", "Category of Hallucination"
    )

    duplicate_questions = (
        int(dataframe[question_column].duplicated().sum())
        if question_column in dataframe.columns
        else 0
    )

    sections = [
        "# MedHallu Dataset Report",
        "",
        "## Dataset Overview",
        "",
        f"- Dataset: `{dataset_config.get('name')}`",
        f"- Subset: `{dataset_config.get('subset')}`",
        f"- Split: `{dataset_config.get('split')}`",
        f"- Rows: `{len(dataframe)}`",
        f"- Columns: `{len(dataframe.columns)}`",
        f"- Duplicate questions: `{duplicate_questions}`",
        "",
        "## Label Distribution",
        "",
    ]

    if label_column in dataframe.columns:
        sections.append(format_distribution(dataframe[label_column]))
    else:
        sections.append(f"`{label_column}` column not available.")

    sections.extend(["", "## Hallucination-Type Distribution", ""])
    if hallucination_type_column in dataframe.columns:
        sections.append(format_distribution(dataframe[hallucination_type_column]))
    else:
        sections.append(f"`{hallucination_type_column}` column not available.")

    sections.extend(["", "## Question Length Statistics", ""])
    if question_column in dataframe.columns:
        sections.append(
            format_length_statistics(
                f"`{question_column}` word lengths",
                length_statistics(dataframe[question_column]),
            )
        )
    else:
        sections.append(f"`{question_column}` column not available.")

    sections.extend(["", "## Answer Length Statistics", ""])
    for answer_column in answer_columns:
        if answer_column in dataframe.columns:
            sections.append(
                format_length_statistics(
                    f"`{answer_column}` word lengths",
                    length_statistics(dataframe[answer_column]),
                )
            )
            sections.append("")
        else:
            sections.append(f"`{answer_column}` column not available.")
            sections.append("")

    sections.extend(
        [
            "## Duplicate Question Detection",
            "",
            f"- Duplicate question count: `{duplicate_questions}`",
            "",
        ]
    )

    if question_column in dataframe.columns and duplicate_questions:
        duplicates = dataframe.loc[
            dataframe[question_column].duplicated(keep=False), question_column
        ].value_counts()
        sections.extend(["| Question | Count |", "|---|---:|"])
        for question, count in duplicates.head(25).items():
            escaped_question = str(question).replace("|", "\\|")
            sections.append(f"| {escaped_question} | {int(count)} |")

    return "\n".join(sections).rstrip() + "\n"


def save_report(report: str, report_path: str | Path) -> Path:
    """Save a markdown report to disk."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    LOGGER.info("Saved dataset report to %s", path)
    return path


def main() -> None:
    """Run dataset preparation and descriptive analysis."""
    args = parse_args()
    config = load_config(args.config)
    initialize_logging(config)

    dataframe = load_or_prepare_dataset(config, refresh=args.refresh)
    report = generate_markdown_report(dataframe, config)
    report_path = config.get("paths", {}).get(
        "dataset_report", "outputs/dataset_report.md"
    )
    save_report(report, report_path)


if __name__ == "__main__":
    main()

