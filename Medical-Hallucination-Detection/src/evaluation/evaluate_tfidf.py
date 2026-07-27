"""Evaluation utilities for the TF-IDF Logistic Regression baseline."""

from __future__ import annotations
from sklearn.preprocessing import label_binarize
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.models.tfidf_logistic import build_input_text, get_target, load_pipeline
from src.utils.config import Config, load_config
from src.utils.logging import setup_logging


matplotlib.use("Agg")

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TF-IDF baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the TF-IDF Logistic Regression baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def initialize_logging(config: Config) -> logging.Logger:
    """Initialize evaluation logging from project configuration."""
    logging_config = config.get("logging", {})
    paths_config = config.get("paths", {})
    return setup_logging(
        log_dir=paths_config.get("logs_dir", "outputs/logs"),
        file_name="evaluate_tfidf.log",
        level=logging_config.get("level", "INFO"),
        log_format=logging_config.get(
            "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
        date_format=logging_config.get("date_format", "%Y-%m-%d %H:%M:%S"),
    )


def split_path(config: Config, split_name: str) -> Path:
    """Return the configured path for a dataset split CSV."""
    split_config = config.get("dataset_split", {})
    splits_dir = Path(split_config.get("splits_dir", "data/splits"))
    file_names = {
        "train": split_config.get("train_file_name", "train.csv"),
        "validation": split_config.get("validation_file_name", "validation.csv"),
        "test": split_config.get("test_file_name", "test.csv"),
    }
    return splits_dir / str(file_names[split_name])


def load_split(config: Config, split_name: str) -> pd.DataFrame:
    """Load a reproducible dataset split."""
    path = split_path(config, split_name)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    LOGGER.info("Loading %s split from %s", split_name, path)
    return pd.read_csv(path)


def configured_labels(config: Config, y_true: pd.Series) -> list[str]:
    """Return configured class labels, including any unexpected observed labels."""
    configured = [
        str(label) for label in config.get("tfidf_logistic", {}).get("class_order", [])
    ]
    observed = [str(label) for label in sorted(y_true.unique())]
    return configured + [label for label in observed if label not in configured]


def evaluate_split(
    pipeline: Any,
    dataframe: pd.DataFrame,
    config: Config,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate the trained pipeline on one split."""
    LOGGER.info("Evaluating TF-IDF Logistic Regression on %s split.", split_name)
    x_values = build_input_text(dataframe, config)
    y_true = get_target(dataframe, config)
    y_pred = pipeline.predict(x_values)
    labels = configured_labels(config, y_true)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics: dict[str, Any] = {
        "split": split_name,
        "number_of_samples": int(len(dataframe)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1_score),
        "roc_auc_ovr": compute_roc_auc(pipeline, x_values, y_true, labels),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    LOGGER.info(
        "%s metrics: accuracy=%.4f, precision_macro=%.4f, recall_macro=%.4f, "
        "f1_macro=%.4f, roc_auc_ovr=%s",
        split_name,
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["roc_auc_ovr"],
    )
    return metrics


from sklearn.preprocessing import label_binarize

def compute_roc_auc(
    pipeline: Any,
    x_values: pd.Series,
    y_true: pd.Series,
    labels: list[str],
) -> float | None:
    """Compute multiclass ROC-AUC using One-vs-Rest."""

    if not hasattr(pipeline, "predict_proba"):
        LOGGER.warning("Pipeline does not expose predict_proba.")
        return None

    try:
        probabilities = pipeline.predict_proba(x_values)

        # Use the classifier's own class ordering
        class_order = list(pipeline.classes_)

        y_true_bin = label_binarize(y_true, classes=class_order)

        return float(
            roc_auc_score(
                y_true_bin,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )

    except Exception as e:
        LOGGER.warning("ROC-AUC could not be computed: %s", e)
        return None

def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Save evaluation metrics as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s metrics to %s", metrics["split"], output_path)
    return output_path


def save_confusion_matrix(
    metrics: dict[str, Any],
    output_path: str | Path,
    dpi: int,
) -> Path:
    """Save a confusion matrix figure for one split."""
    labels = metrics["labels"]
    matrix = np.asarray(metrics["confusion_matrix"])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"TF-IDF Logistic Regression - {metrics['split'].title()}")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved %s confusion matrix to %s", metrics["split"], path)
    return path


def evaluate_and_save(
    pipeline: Any,
    dataframe: pd.DataFrame,
    config: Config,
    split_name: str,
    metrics_path: str | Path,
    confusion_matrix_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one split and save JSON metrics plus confusion matrix figure."""
    metrics = evaluate_split(pipeline, dataframe, config, split_name)
    figure_dpi = int(config.get("tfidf_logistic", {}).get("figure_dpi", 300))
    save_metrics(metrics, metrics_path)
    save_confusion_matrix(metrics, confusion_matrix_path, figure_dpi)
    return metrics


def main() -> None:
    """Evaluate a saved TF-IDF Logistic Regression model."""
    args = parse_args()
    config = load_config(args.config)
    initialize_logging(config)
    baseline_config = config.get("tfidf_logistic", {})

    pipeline = load_pipeline(
        baseline_config.get("model_path", "outputs/models/tfidf_logistic.pkl")
    )
    validation_df = load_split(config, "validation")
    test_df = load_split(config, "test")

    evaluate_and_save(
        pipeline,
        validation_df,
        config,
        "validation",
        baseline_config.get(
            "validation_metrics_path",
            "outputs/results/tfidf_validation_metrics.json",
        ),
        baseline_config.get(
            "validation_confusion_matrix_path",
            "outputs/figures/confusion_matrix_validation.png",
        ),
    )
    evaluate_and_save(
        pipeline,
        test_df,
        config,
        "test",
        baseline_config.get(
            "test_metrics_path",
            "outputs/results/tfidf_test_metrics.json",
        ),
        baseline_config.get(
            "test_confusion_matrix_path",
            "outputs/figures/confusion_matrix_test.png",
        ),
    )
    LOGGER.info("TF-IDF Logistic Regression evaluation completed successfully.")


if __name__ == "__main__":
    main()

