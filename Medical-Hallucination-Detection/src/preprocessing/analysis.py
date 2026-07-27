"""Publication-quality exploratory data analysis for MedHallu.

This module extends dataset engineering with descriptive statistics, figures,
semantic duplicate detection, quality reporting, and an IEEE-ready LaTeX table.
It intentionally avoids tokenization, feature engineering, and model training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.preprocessing.dataset import MedHalluDataset, dataframe_from_csv
from src.utils.config import Config, load_config
from src.utils.logging import setup_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
WORD_PATTERN = re.compile(r"\b\w+\b")

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset analysis."""
    parser = argparse.ArgumentParser(description="Generate MedHallu EDA artifacts.")
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


def text_lengths(series: pd.Series) -> pd.Series:
    """Compute whitespace-delimited word counts for descriptive EDA."""
    return series.fillna("").astype(str).str.split().str.len()


def length_statistics(series: pd.Series) -> dict[str, float]:
    """Compute descriptive text length statistics."""
    lengths = text_lengths(series)
    return {
        "min": float(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "max": float(lengths.max()),
        "std": float(lengths.std(ddof=0)),
    }


def series_to_text(series: pd.Series) -> pd.Series:
    """Convert a text-like column to strings for reporting."""
    return series.fillna("").astype(str)


def distribution_dict(series: pd.Series) -> dict[str, int]:
    """Return a JSON-serializable value-count distribution."""
    counts = series.value_counts(dropna=False)
    return {str(value): int(count) for value, count in counts.items()}


def compute_split_counts(dataframe: pd.DataFrame, config: Config) -> dict[str, int]:
    """Compute train/validation/test counts from config or a split column."""
    dataset_config = config["dataset"]
    split_names = dataset_config.get("split_names", ["train", "validation", "test"])
    split_column = dataset_config.get("split_column")
    configured_split = dataset_config.get("split", "train")

    if split_column and split_column in dataframe.columns:
        counts = dataframe[split_column].value_counts(dropna=False).to_dict()
        return {split: int(counts.get(split, 0)) for split in split_names}

    return {
        split: int(len(dataframe)) if split == configured_split else 0
        for split in split_names
    }


def estimate_vocabulary_size(dataframe: pd.DataFrame, config: Config) -> int:
    """Estimate lexical vocabulary size from configured text columns.

    This is a simple whitespace/regex-based EDA estimate and is not model
    tokenization.
    """
    eda_config = config.get("eda", {})
    columns = eda_config.get("vocabulary_columns", [])
    lowercase = bool(eda_config.get("vocabulary_lowercase", True))
    vocabulary: set[str] = set()

    for column in columns:
        if column not in dataframe.columns:
            LOGGER.warning("Skipping vocabulary column not present: %s", column)
            continue

        for value in series_to_text(dataframe[column]):
            text = value.lower() if lowercase else value
            vocabulary.update(WORD_PATTERN.findall(text))

    return len(vocabulary)


def answer_length_series(dataframe: pd.DataFrame, config: Config) -> pd.Series:
    """Return row-level answer lengths across configured answer columns."""
    answer_columns = config["dataset"].get(
        "answer_columns", ["Ground Truth", "Hallucinated Answer"]
    )
    available_columns = [column for column in answer_columns if column in dataframe]

    if not available_columns:
        return pd.Series([0] * len(dataframe), index=dataframe.index)

    return pd.concat(
        [text_lengths(dataframe[column]) for column in available_columns], axis=1
    ).mean(axis=1)


def compute_dataset_statistics(dataframe: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Compute publication-oriented dataset statistics."""
    dataset_config = config["dataset"]
    question_column = dataset_config.get("question_column", "Question")
    evidence_column = dataset_config.get("evidence_column", "Knowledge")
    answer_columns = dataset_config.get(
        "answer_columns", ["Ground Truth", "Hallucinated Answer"]
    )
    label_column = dataset_config.get("label_column", "Difficulty Level")
    hallucination_column = dataset_config.get(
        "hallucination_type_column", "Category of Hallucination"
    )

    LOGGER.info("Computing dataset statistics.")
    answer_lengths = answer_length_series(dataframe, config)
    statistics: dict[str, Any] = {
        "total_samples": int(len(dataframe)),
        "split_counts": compute_split_counts(dataframe, config),
        "label_distribution": distribution_dict(dataframe[label_column])
        if label_column in dataframe
        else {},
        "hallucination_type_distribution": distribution_dict(
            dataframe[hallucination_column]
        )
        if hallucination_column in dataframe
        else {},
        "vocabulary_size": estimate_vocabulary_size(dataframe, config),
    }

    if question_column in dataframe:
        question_stats = length_statistics(dataframe[question_column])
        statistics["question_length"] = question_stats
        statistics["average_question_length"] = question_stats["mean"]
        statistics["maximum_question_length"] = question_stats["max"]
        statistics["minimum_question_length"] = question_stats["min"]

    statistics["answer_length"] = {
        "min": float(answer_lengths.min()),
        "mean": float(answer_lengths.mean()),
        "median": float(answer_lengths.median()),
        "max": float(answer_lengths.max()),
        "std": float(answer_lengths.std(ddof=0)),
    }
    statistics["average_answer_length"] = statistics["answer_length"]["mean"]
    statistics["maximum_answer_length"] = statistics["answer_length"]["max"]
    statistics["minimum_answer_length"] = statistics["answer_length"]["min"]

    for answer_column in answer_columns:
        if answer_column in dataframe:
            statistics[f"{answer_column}_length"] = length_statistics(
                dataframe[answer_column]
            )

    if evidence_column in dataframe:
        evidence_stats = length_statistics(dataframe[evidence_column])
        statistics["evidence_length"] = evidence_stats
        statistics["average_evidence_length"] = evidence_stats["mean"]
        statistics["maximum_evidence_length"] = evidence_stats["max"]
        statistics["minimum_evidence_length"] = evidence_stats["min"]

    return statistics


def flatten_statistics(statistics: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten nested statistics into metric/value rows for CSV export."""
    rows: list[dict[str, str]] = []

    def add_rows(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                add_rows(f"{prefix}.{nested_key}" if prefix else str(nested_key), nested_value)
        else:
            rows.append({"metric": prefix, "value": str(value)})

    add_rows("", statistics)
    return rows


def save_dataset_statistics(statistics: dict[str, Any], config: Config) -> None:
    """Save dataset statistics as JSON and CSV."""
    paths_config = config.get("paths", {})
    json_path = Path(paths_config.get("dataset_statistics_json", "outputs/dataset_statistics.json"))
    csv_path = Path(paths_config.get("dataset_statistics_csv", "outputs/dataset_statistics.csv"))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    pd.DataFrame(flatten_statistics(statistics)).to_csv(csv_path, index=False)
    LOGGER.info("Saved dataset statistics JSON to %s", json_path)
    LOGGER.info("Saved dataset statistics CSV to %s", csv_path)


def save_bar_figure(
    distribution: dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Save a publication-quality bar chart with matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(distribution.keys())
    counts = list(distribution.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color="#2F5597", edgecolor="black", linewidth=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved figure to %s", output_path)


def save_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
    bins: int,
) -> None:
    """Save a publication-quality histogram with matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="#4F81BD", edgecolor="black", linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved figure to %s", output_path)


def generate_figures(dataframe: pd.DataFrame, config: Config) -> None:
    """Generate all configured EDA figures at publication DPI."""
    LOGGER.info("Generating publication-quality dataset figures.")
    dataset_config = config["dataset"]
    eda_config = config.get("eda", {})
    figures_dir = Path(config.get("paths", {}).get("figures_dir", "outputs/figures"))
    dpi = int(eda_config.get("figure_dpi", 300))
    bins = int(eda_config.get("histogram_bins", 30))

    label_column = dataset_config.get("label_column", "Difficulty Level")
    hallucination_column = dataset_config.get(
        "hallucination_type_column", "Category of Hallucination"
    )
    question_column = dataset_config.get("question_column", "Question")
    evidence_column = dataset_config.get("evidence_column", "Knowledge")

    if label_column in dataframe:
        save_bar_figure(
            distribution_dict(dataframe[label_column]),
            "Label Distribution",
            "Label",
            "Samples",
            figures_dir / "label_distribution.png",
            dpi,
        )

    if hallucination_column in dataframe:
        save_bar_figure(
            distribution_dict(dataframe[hallucination_column]),
            "Hallucination-Type Distribution",
            "Hallucination Type",
            "Samples",
            figures_dir / "hallucination_distribution.png",
            dpi,
        )

    if question_column in dataframe:
        save_histogram(
            text_lengths(dataframe[question_column]),
            "Question Length Distribution",
            "Question length (words)",
            "Samples",
            figures_dir / "question_length_histogram.png",
            dpi,
            bins,
        )

    save_histogram(
        answer_length_series(dataframe, config),
        "Answer Length Distribution",
        "Average answer length per sample (words)",
        "Samples",
        figures_dir / "answer_length_histogram.png",
        dpi,
        bins,
    )

    if evidence_column in dataframe:
        save_histogram(
            text_lengths(dataframe[evidence_column]),
            "Evidence Length Distribution",
            "Evidence length (words)",
            "Samples",
            figures_dir / "evidence_length_histogram.png",
            dpi,
            bins,
        )


def detect_semantic_duplicates(
    dataframe: pd.DataFrame, config: Config
) -> dict[str, Any]:
    """Detect semantic duplicate text pairs using sentence-transformers."""
    semantic_config = config.get("eda", {}).get("semantic_duplicates", {})
    enabled = bool(semantic_config.get("enabled", True))
    text_column = semantic_config.get(
        "text_column", config["dataset"].get("question_column", "Question")
    )
    threshold = float(semantic_config.get("cosine_threshold", 0.9))
    max_examples = int(semantic_config.get("max_examples", 10))
    batch_size = int(semantic_config.get("batch_size", 32))
    model_name = semantic_config.get(
        "model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )

    result: dict[str, Any] = {
        "enabled": enabled,
        "model_name": model_name,
        "text_column": text_column,
        "cosine_similarity_threshold": threshold,
        "number_of_semantic_duplicates": 0,
        "examples": [],
    }

    if not enabled:
        LOGGER.info("Semantic duplicate detection is disabled.")
        return result

    if text_column not in dataframe:
        LOGGER.warning("Semantic duplicate text column is unavailable: %s", text_column)
        result["error"] = f"Column not available: {text_column}"
        return result

    LOGGER.info(
        "Detecting semantic duplicates in '%s' using %s at threshold %.3f.",
        text_column,
        model_name,
        threshold,
    )
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    LOGGER.info("Importing sentence-transformers for semantic duplicate detection.")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        LOGGER.exception("sentence-transformers is required for semantic duplicates.")
        result["error"] = str(error)
        return result

    texts = series_to_text(dataframe[text_column]).tolist()
    LOGGER.info("Loading sentence-transformers model: %s", model_name)
    model = SentenceTransformer(model_name)
    LOGGER.info("Encoding %s texts for semantic duplicate detection.", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=bool(semantic_config.get("show_progress_bar", False)),
    )
    similarity_matrix = np.asarray(embeddings) @ np.asarray(embeddings).T
    upper_rows, upper_cols = np.triu_indices(len(texts), k=1)
    similarities = similarity_matrix[upper_rows, upper_cols]
    duplicate_positions = np.where(similarities >= threshold)[0]

    examples = []
    for position in duplicate_positions[:max_examples]:
        row_index = int(upper_rows[position])
        col_index = int(upper_cols[position])
        examples.append(
            {
                "row_index_1": row_index,
                "row_index_2": col_index,
                "cosine_similarity": float(similarities[position]),
                "text_1": texts[row_index],
                "text_2": texts[col_index],
            }
        )

    result["number_of_semantic_duplicates"] = int(len(duplicate_positions))
    result["examples"] = examples
    LOGGER.info(
        "Detected %s semantic duplicate pairs at threshold %.3f.",
        result["number_of_semantic_duplicates"],
        threshold,
    )
    return result


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
    lines = [f"### {title}", "", "| Metric | Value |", "|---|---:|"]
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
    """Generate the original descriptive dataset analysis report."""
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


def imbalance_summary(dataframe: pd.DataFrame, label_column: str) -> dict[str, Any]:
    """Compute simple label imbalance diagnostics."""
    if label_column not in dataframe:
        return {"available": False}

    counts = dataframe[label_column].value_counts()
    majority = int(counts.max()) if not counts.empty else 0
    minority = int(counts.min()) if not counts.empty else 0
    ratio = float(majority / minority) if minority else 0.0
    return {
        "available": True,
        "majority_class": str(counts.idxmax()) if not counts.empty else "",
        "minority_class": str(counts.idxmin()) if not counts.empty else "",
        "majority_to_minority_ratio": ratio,
    }


def generate_quality_report(
    dataframe: pd.DataFrame,
    config: Config,
    validation: Any,
    semantic_duplicates: dict[str, Any],
    statistics: dict[str, Any],
) -> str:
    """Generate a dataset quality report for research documentation."""
    dataset_config = config["dataset"]
    label_column = dataset_config.get("label_column", "Difficulty Level")
    imbalance = imbalance_summary(dataframe, label_column)
    missing_total = sum(validation.missing_values.values())
    potential_issues: list[str] = []
    recommendations: list[str] = []

    if missing_total:
        potential_issues.append("Missing values are present and require review.")
        recommendations.append("Inspect missing-value rows before model development.")
    if validation.duplicate_rows or validation.duplicate_questions:
        potential_issues.append("Exact duplicates are present.")
        recommendations.append("Decide whether exact duplicates should be removed.")
    if semantic_duplicates["number_of_semantic_duplicates"]:
        potential_issues.append("Semantic duplicate question pairs were detected.")
        recommendations.append(
            "Review semantic duplicate examples before defining train/test splits."
        )
    if imbalance.get("available") and imbalance["majority_to_minority_ratio"] >= 2.0:
        potential_issues.append("The label distribution may be imbalanced.")
        recommendations.append(
            "Use stratified splitting and imbalance-aware metrics in future modeling."
        )
    if not potential_issues:
        potential_issues.append("No major dataset quality issues were detected.")
    if not recommendations:
        recommendations.append("Proceed with documented split design and baseline setup.")

    sections = [
        "# Dataset Quality Report",
        "",
        "## Schema",
        "",
        f"- Columns: `{list(dataframe.columns)}`",
        f"- Missing required columns: `{validation.missing_columns}`",
        "",
        "## Missing Values",
        "",
        "| Column | Missing Values |",
        "|---|---:|",
    ]
    sections.extend(
        f"| {column} | {count} |"
        for column, count in validation.missing_values.items()
    )
    sections.extend(
        [
            "",
            "## Exact Duplicates",
            "",
            f"- Duplicate full rows: `{validation.duplicate_rows}`",
            f"- Duplicate questions: `{validation.duplicate_questions}`",
            "",
            "## Semantic Duplicates",
            "",
            f"- Enabled: `{semantic_duplicates['enabled']}`",
            f"- Model: `{semantic_duplicates['model_name']}`",
            f"- Text column: `{semantic_duplicates['text_column']}`",
            "- Cosine similarity threshold: "
            f"`{semantic_duplicates['cosine_similarity_threshold']}`",
            "- Number of semantic duplicate pairs: "
            f"`{semantic_duplicates['number_of_semantic_duplicates']}`",
            "",
        ]
    )

    examples = semantic_duplicates.get("examples", [])
    if examples:
        sections.extend(["| Row 1 | Row 2 | Similarity | Text 1 | Text 2 |", "|---:|---:|---:|---|---|"])
        for example in examples:
            text_1 = str(example["text_1"]).replace("|", "\\|")
            text_2 = str(example["text_2"]).replace("|", "\\|")
            sections.append(
                "| {row1} | {row2} | {score:.4f} | {text1} | {text2} |".format(
                    row1=example["row_index_1"],
                    row2=example["row_index_2"],
                    score=example["cosine_similarity"],
                    text1=text_1,
                    text2=text_2,
                )
            )
    else:
        sections.append("No semantic duplicate examples above the configured threshold.")

    sections.extend(
        [
            "",
            "## Dataset Imbalance",
            "",
            f"- Label distribution: `{statistics.get('label_distribution', {})}`",
            "- Majority-to-minority ratio: "
            f"`{imbalance.get('majority_to_minority_ratio', 'N/A')}`",
            "",
            "## Potential Issues",
            "",
        ]
    )
    sections.extend(f"- {issue}" for issue in potential_issues)
    sections.extend(["", "## Recommendations", ""])
    sections.extend(f"- {recommendation}" for recommendation in recommendations)

    return "\n".join(sections).rstrip() + "\n"


def latex_escape(value: Any) -> str:
    """Escape text for LaTeX table cells."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def generate_latex_table(statistics: dict[str, Any]) -> str:
    """Generate an IEEE-ready LaTeX dataset summary table."""
    split_counts = statistics.get("split_counts", {})
    rows = [
        ("Total samples", statistics.get("total_samples", 0)),
        ("Train samples", split_counts.get("train", 0)),
        ("Validation samples", split_counts.get("validation", 0)),
        ("Test samples", split_counts.get("test", 0)),
        ("Vocabulary size", statistics.get("vocabulary_size", 0)),
        ("Avg. question length", f"{statistics.get('average_question_length', 0):.2f}"),
        ("Avg. answer length", f"{statistics.get('average_answer_length', 0):.2f}"),
        ("Avg. evidence length", f"{statistics.get('average_evidence_length', 0):.2f}"),
        ("Max question length", f"{statistics.get('maximum_question_length', 0):.0f}"),
        ("Max answer length", f"{statistics.get('maximum_answer_length', 0):.0f}"),
        ("Max evidence length", f"{statistics.get('maximum_evidence_length', 0):.0f}"),
    ]
    body = "\n".join(
        f"{latex_escape(metric)} & {latex_escape(value)} \\\\" for metric, value in rows
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Summary statistics for the MedHallu dataset.}\n"
        "\\label{tab:dataset-summary}\n"
        "\\begin{tabular}{lr}\n"
        "\\hline\n"
        "Statistic & Value \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def save_latex_table(table: str, config: Config) -> Path:
    """Save the LaTeX dataset summary table."""
    path = Path(
        config.get("paths", {}).get(
            "dataset_summary_tex", "outputs/tables/dataset_summary.tex"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table, encoding="utf-8")
    LOGGER.info("Saved LaTeX dataset summary table to %s", path)
    return path


def main() -> None:
    """Run dataset preparation and publication-quality EDA."""
    args = parse_args()
    config = load_config(args.config)
    initialize_logging(config)

    dataframe = load_or_prepare_dataset(config, refresh=args.refresh)
    medhallu_dataset = MedHalluDataset.from_config(config)
    validation = medhallu_dataset.validate(dataframe)

    statistics = compute_dataset_statistics(dataframe, config)
    semantic_duplicates = detect_semantic_duplicates(dataframe, config)
    statistics["semantic_duplicates"] = semantic_duplicates

    save_dataset_statistics(statistics, config)
    generate_figures(dataframe, config)

    report = generate_markdown_report(dataframe, config)
    save_report(report, config.get("paths", {}).get("dataset_report", "outputs/dataset_report.md"))

    quality_report = generate_quality_report(
        dataframe, config, validation, semantic_duplicates, statistics
    )
    save_report(
        quality_report,
        config.get("paths", {}).get(
            "dataset_quality_report", "outputs/dataset_quality_report.md"
        ),
    )

    save_latex_table(generate_latex_table(statistics), config)
    LOGGER.info("Dataset EDA pipeline completed successfully.")


if __name__ == "__main__":
    main()
