"""Classification metrics and threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

Metrics = dict[str, Any]


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)  # type: ignore[arg-type]
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def calculate_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> Metrics:
    predictions = (probabilities >= threshold).astype(int)
    metrics: Metrics = {
        "threshold": threshold,
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0  # type: ignore[arg-type]
        ),
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(y_true, probabilities)
        )
    return metrics


def flatten_metrics(metrics: Metrics) -> dict[str, float]:
    """Flatten nested metric dictionaries for MLflow or tabular reporting."""
    flat: dict[str, float] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for name, nested_value in value.items():
                visit(nested_value, f"{prefix}_{name}" if prefix else str(name))
        elif isinstance(value, (int, float)):
            flat[prefix] = float(value)

    visit(metrics, "")
    return flat
