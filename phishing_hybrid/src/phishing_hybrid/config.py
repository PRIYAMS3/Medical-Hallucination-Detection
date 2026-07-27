from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    dataset_path: str
    output_root: str


@dataclass
class DatasetConfig:
    target_column: str
    test_size: float
    random_seed: int
    fillna_value: float


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    use_class_weights: bool
    device: str


@dataclass
class RuleConfig:
    feature: str
    op: str
    value: float
    reason: str


@dataclass
class HybridConfig:
    threshold: float
    rules: list[RuleConfig]


@dataclass
class ExplainabilityConfig:
    background_size: int
    explain_size: int


@dataclass
class ExperimentSection:
    models: list[str]
    ensemble_models: list[str]


@dataclass
class AppConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    training: TrainingConfig
    experiment: ExperimentSection
    hybrid: HybridConfig
    explainability: ExplainabilityConfig


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required config key '{key}'")
    return section[key]


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths = raw.get("paths", {})
    dataset = raw.get("dataset", {})
    training = raw.get("training", {})
    experiment = raw.get("experiment", {})
    hybrid = raw.get("hybrid", {})
    explainability = raw.get("explainability", {})

    rule_items = []
    for item in hybrid.get("rules", []):
        rule_items.append(
            RuleConfig(
                feature=_require(item, "feature"),
                op=_require(item, "op"),
                value=float(_require(item, "value")),
                reason=_require(item, "reason"),
            )
        )

    return AppConfig(
        paths=PathsConfig(
            dataset_path=str(_require(paths, "dataset_path")),
            output_root=str(paths.get("output_root", "outputs")),
        ),
        dataset=DatasetConfig(
            target_column=str(dataset.get("target_column", "Result")),
            test_size=float(dataset.get("test_size", 0.2)),
            random_seed=int(dataset.get("random_seed", 42)),
            fillna_value=float(dataset.get("fillna_value", 0.0)),
        ),
        training=TrainingConfig(
            batch_size=int(training.get("batch_size", 64)),
            epochs=int(training.get("epochs", 20)),
            learning_rate=float(training.get("learning_rate", 0.001)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            use_class_weights=bool(training.get("use_class_weights", True)),
            device=str(training.get("device", "auto")),
        ),
        experiment=ExperimentSection(
            models=list(experiment.get("models", [])),
            ensemble_models=list(experiment.get("ensemble_models", [])),
        ),
        hybrid=HybridConfig(
            threshold=float(hybrid.get("threshold", 0.8)),
            rules=rule_items,
        ),
        explainability=ExplainabilityConfig(
            background_size=int(explainability.get("background_size", 100)),
            explain_size=int(explainability.get("explain_size", 50)),
        ),
    )
