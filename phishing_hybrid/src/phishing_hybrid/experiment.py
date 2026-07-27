from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from phishing_hybrid.config import AppConfig
from phishing_hybrid.data import get_class_weights, load_dataset, split_dataset
from phishing_hybrid.models import build_model
from phishing_hybrid.training import (
    create_data_loaders,
    dump_json,
    evaluate_predictions,
    predict_logits,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_model,
)


def _ensemble_metrics(
    trained_models: dict[str, torch.nn.Module],
    model_names: list[str],
    test_loader,
    device: torch.device,
) -> dict[str, Any]:
    probs_by_model = []
    y_true_reference = None
    for name in model_names:
        model = trained_models[name]
        y_true, _, y_prob = predict_logits(model, test_loader, device)
        if y_true_reference is None:
            y_true_reference = y_true
        probs_by_model.append(y_prob)

    stacked = torch.tensor(probs_by_model, dtype=torch.float32)
    mean_prob = torch.mean(stacked, dim=0).numpy()
    preds = (mean_prob >= 0.5).astype(int)
    return evaluate_predictions(y_true_reference, preds, mean_prob)


def run_baseline_experiment(config: AppConfig) -> dict[str, Any]:
    set_seed(config.dataset.random_seed)
    device = resolve_device(config.training.device)

    x, y = load_dataset(
        path=config.paths.dataset_path,
        target_column=config.dataset.target_column,
        fillna_value=config.dataset.fillna_value,
    )
    feature_names = list(x.columns)

    x_train, x_test, y_train, y_test = split_dataset(
        x=x,
        y=y,
        test_size=config.dataset.test_size,
        random_seed=config.dataset.random_seed,
    )

    class_weights = None
    if config.training.use_class_weights:
        class_weights = get_class_weights(y_train)

    train_loader, test_loader = create_data_loaders(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=config.training.batch_size,
    )

    output_root = Path(config.paths.output_root)
    models_dir = output_root / "models"
    results_dir = output_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    input_dim = x_train.shape[1]
    summary_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {
        "dataset_path": config.paths.dataset_path,
        "device": str(device),
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "class_distribution_train": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum()),
        },
        "class_distribution_test": {
            "0": int((y_test == 0).sum()),
            "1": int((y_test == 1).sum()),
        },
        "models": {},
    }

    trained_models: dict[str, torch.nn.Module] = {}

    for model_name in config.experiment.models:
        model = build_model(model_name, input_dim)
        model, losses = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            class_weights=class_weights,
        )

        metrics = evaluate_predictions(*predict_logits(model, test_loader, device))
        trained_models[model_name] = model

        save_checkpoint(
            output_path=models_dir / f"{model_name}.pt",
            model_name=model_name,
            model=model,
            input_dim=input_dim,
            feature_names=feature_names,
            metrics=metrics,
        )

        detailed["models"][model_name] = {
            "metrics": metrics,
            "train_loss_by_epoch": losses,
        }

        summary_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1-score": round(metrics["f1_score"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
            }
        )

    valid_ensemble = [name for name in config.experiment.ensemble_models if name in trained_models]
    if valid_ensemble:
        ensemble_metrics = _ensemble_metrics(
            trained_models=trained_models,
            model_names=valid_ensemble,
            test_loader=test_loader,
            device=device,
        )
        detailed["models"]["ensemble"] = {"members": valid_ensemble, "metrics": ensemble_metrics}
        summary_rows.append(
            {
                "Model": "ensemble",
                "Accuracy": round(ensemble_metrics["accuracy"], 4),
                "Precision": round(ensemble_metrics["precision"], 4),
                "Recall": round(ensemble_metrics["recall"], 4),
                "F1-score": round(ensemble_metrics["f1_score"], 4),
                "ROC-AUC": round(ensemble_metrics["roc_auc"], 4),
            }
        )

    df_results = pd.DataFrame(summary_rows).sort_values(by="F1-score", ascending=False)
    results_csv = results_dir / "results.csv"
    df_results.to_csv(results_csv, index=False)

    detailed_path = results_dir / "metrics_detailed.json"
    dump_json(detailed_path, detailed)

    return {
        "results_csv": str(results_csv),
        "detailed_json": str(detailed_path),
        "best_model": str(df_results.iloc[0]["Model"]) if not df_results.empty else None,
        "num_models": len(config.experiment.models),
    }
