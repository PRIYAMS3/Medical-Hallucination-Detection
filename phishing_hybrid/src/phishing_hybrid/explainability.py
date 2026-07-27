from __future__ import annotations

from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from phishing_hybrid.config import AppConfig
from phishing_hybrid.data import load_dataset
from phishing_hybrid.training import load_checkpoint


def _to_positive_class_shap(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def run_shap_summary(config: AppConfig, checkpoint_path: str | Path) -> str:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install requirements first.") from exc

    x, _ = load_dataset(
        path=config.paths.dataset_path,
        target_column=config.dataset.target_column,
        fillna_value=config.dataset.fillna_value,
    )

    device = torch.device("cpu")
    model, metadata = load_checkpoint(checkpoint_path, device)
    feature_names = metadata["feature_names"] or list(x.columns)

    bg_size = min(config.explainability.background_size, len(x))
    explain_size = min(config.explainability.explain_size, len(x))

    background = x.iloc[:bg_size].to_numpy(dtype=np.float32)
    explain_x = x.iloc[:explain_size].to_numpy(dtype=np.float32)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(batch, dtype=torch.float32)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    explainer = shap.KernelExplainer(cast(Callable[[np.ndarray], np.ndarray], predict_fn), background)
    shap_values = explainer.shap_values(explain_x)
    shap_pos = _to_positive_class_shap(shap_values)

    out_dir = Path(config.paths.output_root) / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "shap_summary.png"

    shap.summary_plot(shap_pos, explain_x, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

    return str(plot_path)
