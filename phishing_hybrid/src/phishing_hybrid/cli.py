from __future__ import annotations

import argparse
import json
from pathlib import Path

from phishing_hybrid.config import load_config
from phishing_hybrid.data import load_dataset
from phishing_hybrid.experiment import run_baseline_experiment
from phishing_hybrid.explainability import run_shap_summary
from phishing_hybrid.hybrid import HybridPhishingSystem
from phishing_hybrid.training import load_checkpoint, resolve_device


def _cmd_run_baseline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summary = run_baseline_experiment(config)
    print(json.dumps(summary, indent=2))


def _cmd_explain(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    plot_path = run_shap_summary(config=config, checkpoint_path=args.checkpoint)
    print(f"SHAP summary plot saved to: {plot_path}")


def _cmd_predict_one(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset_path = args.dataset_path or config.paths.dataset_path

    x, _ = load_dataset(
        path=dataset_path,
        target_column=config.dataset.target_column,
        fillna_value=config.dataset.fillna_value,
    )

    idx = args.sample_index
    if idx < 0 or idx >= len(x):
        raise IndexError(f"sample-index {idx} is out of range [0, {len(x)-1}]")

    device = resolve_device(config.training.device)
    model, metadata = load_checkpoint(args.checkpoint, device)

    system = HybridPhishingSystem(
        model=model,
        feature_names=metadata["feature_names"] or list(x.columns),
        rules=config.hybrid.rules,
        threshold=config.hybrid.threshold,
        device=device,
    )

    sample = x.iloc[idx].to_numpy(dtype="float32")
    decision = system.predict_as_dict(sample)
    print(json.dumps(decision, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intelligent Hybrid Phishing Detection CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_baseline = sub.add_parser("run-baseline", help="Train/evaluate all configured models and ensemble.")
    run_baseline.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    run_baseline.set_defaults(func=_cmd_run_baseline)

    explain = sub.add_parser("explain", help="Run SHAP summary for a saved checkpoint.")
    explain.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    explain.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    explain.set_defaults(func=_cmd_explain)

    predict_one = sub.add_parser("predict-one", help="Run hybrid prediction for one dataset row.")
    predict_one.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    predict_one.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    predict_one.add_argument("--sample-index", type=int, default=0, help="Row index from dataset.")
    predict_one.add_argument("--dataset-path", default=None, help="Optional override for dataset path.")
    predict_one.set_defaults(func=_cmd_predict_one)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
