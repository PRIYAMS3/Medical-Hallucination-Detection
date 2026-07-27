"""Train the TF-IDF Logistic Regression baseline on fixed MedHallu splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.evaluation.evaluate_tfidf import evaluate_and_save, load_split
from src.models.tfidf_logistic import (
    build_input_text,
    build_tfidf_logistic_pipeline,
    get_target,
    save_pipeline,
)
from src.utils.config import Config, load_config
from src.utils.logging import setup_logging


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TF-IDF baseline training."""
    parser = argparse.ArgumentParser(
        description="Train the TF-IDF Logistic Regression baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def initialize_logging(config: Config) -> logging.Logger:
    """Initialize training logging from project configuration."""
    logging_config = config.get("logging", {})
    paths_config = config.get("paths", {})
    return setup_logging(
        log_dir=paths_config.get("logs_dir", "outputs/logs"),
        file_name="train_tfidf.log",
        level=logging_config.get("level", "INFO"),
        log_format=logging_config.get(
            "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
        date_format=logging_config.get("date_format", "%Y-%m-%d %H:%M:%S"),
    )


def train(config: Config) -> Path:
    """Train and save the TF-IDF Logistic Regression pipeline."""
    baseline_config = config.get("tfidf_logistic", {})
    train_df = load_split(config, "train")
    x_train = build_input_text(train_df, config)
    y_train = get_target(train_df, config)

    LOGGER.info("Building TF-IDF Logistic Regression pipeline.")
    pipeline = build_tfidf_logistic_pipeline(config)
    LOGGER.info("Training baseline on %s samples.", len(train_df))
    pipeline.fit(x_train, y_train)

    model_path = save_pipeline(
        pipeline, baseline_config.get("model_path", "outputs/models/tfidf_logistic.pkl")
    )
    LOGGER.info("Saved trained TF-IDF Logistic Regression model to %s", model_path)

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
    return model_path


def main() -> None:
    """Run TF-IDF Logistic Regression baseline training and evaluation."""
    args = parse_args()
    config = load_config(args.config)
    initialize_logging(config)
    train(config)
    LOGGER.info("TF-IDF Logistic Regression training completed successfully.")


if __name__ == "__main__":
    main()
