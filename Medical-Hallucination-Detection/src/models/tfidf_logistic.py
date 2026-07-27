"""TF-IDF plus Logistic Regression baseline for MedHallu difficulty prediction."""

from __future__ import annotations
from sklearn.preprocessing import label_binarize

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.config import Config


TfidfLogisticPipeline = Pipeline


def get_baseline_config(config: Config) -> dict[str, Any]:
    """Return TF-IDF Logistic Regression configuration values."""
    return config.get("tfidf_logistic", {})


def build_input_text(dataframe: pd.DataFrame, config: Config) -> pd.Series:
    """Construct model input text from configured MedHallu columns.

    The text format is intentionally transparent and reproducible:

    ``Question: ... Knowledge: ... Hallucinated Answer: ...``

    Args:
        dataframe: Input split DataFrame.
        config: Loaded project configuration.

    Returns:
        Series of concatenated input strings.

    Raises:
        ValueError: If any required input column is missing.
    """
    baseline_config = get_baseline_config(config)
    input_columns = baseline_config.get("input_columns", {})
    question_column = input_columns.get("question", "Question")
    knowledge_column = input_columns.get("knowledge", "Knowledge")
    answer_column = input_columns.get("hallucinated_answer", "Hallucinated Answer")
    required_columns = [question_column, knowledge_column, answer_column]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    template = baseline_config.get(
        "text_template",
        "Question:\n{question}\n\nKnowledge:\n{knowledge}\n\n"
        "Hallucinated Answer:\n{hallucinated_answer}",
    )

    return dataframe.apply(
        lambda row: template.format(
            question=_clean_text(row[question_column]),
            knowledge=_clean_text(row[knowledge_column]),
            hallucinated_answer=_clean_text(row[answer_column]),
        ),
        axis=1,
    )


def get_target(dataframe: pd.DataFrame, config: Config) -> pd.Series:
    """Return the configured target labels.

    Args:
        dataframe: Input split DataFrame.
        config: Loaded project configuration.

    Returns:
        Series containing target labels.

    Raises:
        ValueError: If the target column is missing.
    """
    target_column = get_baseline_config(config).get(
        "target_column", "Difficulty Level"
    )
    if target_column not in dataframe.columns:
        raise ValueError(f"Missing target column: {target_column}")

    return dataframe[target_column].astype(str)


def build_tfidf_logistic_pipeline(config: Config) -> TfidfLogisticPipeline:
    """Build a configurable scikit-learn TF-IDF Logistic Regression pipeline."""
    baseline_config = get_baseline_config(config)
    tfidf_config = baseline_config.get("tfidf", {})
    logistic_config = baseline_config.get("logistic_regression", {})

    vectorizer = TfidfVectorizer(
    lowercase=bool(tfidf_config.get("lowercase", True)),
    strip_accents=tfidf_config.get("strip_accents", "unicode"),
    analyzer=tfidf_config.get("analyzer", "word"),
    stop_words=tfidf_config.get("stop_words", "english"),
    ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
    max_features=tfidf_config.get("max_features", 75000),
    min_df=tfidf_config.get("min_df", 2),
    max_df=tfidf_config.get("max_df", 0.95),
    sublinear_tf=bool(tfidf_config.get("sublinear_tf", True)),
)

    classifier = LogisticRegression(
    C=float(logistic_config.get("C", 1.0)),
    penalty=logistic_config.get("penalty", "l2"),
    solver=logistic_config.get("solver", "lbfgs"),
    max_iter=int(logistic_config.get("max_iter", 2000)),
    class_weight=logistic_config.get("class_weight", "balanced"),
    multi_class="multinomial",
    n_jobs=logistic_config.get("n_jobs", -1),
    random_state=logistic_config.get("random_state", 42),
)

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("logistic_regression", classifier),
        ]
    )


def save_pipeline(pipeline: TfidfLogisticPipeline, path: str | Path) -> Path:
    """Persist a trained scikit-learn pipeline with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def load_pipeline(path: str | Path) -> TfidfLogisticPipeline:
    """Load a trained scikit-learn pipeline from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def _clean_text(value: Any) -> str:
    """Convert a dataset cell to a safe string for concatenation."""
    if pd.isna(value):
        return ""
    return str(value)

