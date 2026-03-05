from __future__ import annotations

import math
import warnings

import numpy as np

from emotions.binary.metrics_binary import compute_binary_metrics
from emotions.multiclass.metrics_multiclass import compute_multiclass_metrics


def test_binary_single_class_auc_is_nan_without_runtime_warning() -> None:
    y_true = np.array([1, 1, 1, 1], dtype=int)
    y_pred = np.array([0.7, 0.8, 0.9, 0.6], dtype=float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = compute_binary_metrics(y_pred=y_pred, y_true=y_true, threshold=0.5)

    assert math.isnan(metrics["auc"])
    assert not any("AUC undefined" in str(item.message) for item in caught)


def test_multiclass_undefined_auc_is_nan_without_warning_spam() -> None:
    y_true = np.array([1, 1, 1], dtype=int)
    y_pred_proba = np.array(
        [
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.8, 0.1],
        ],
        dtype=float,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = compute_multiclass_metrics(
            y_pred_proba=y_pred_proba,
            y_true=y_true,
            class_labels=[0, 1, 2],
        )

    assert math.isnan(metrics["macro_auc_ovr"])
    assert math.isnan(metrics["weighted_auc_ovr"])
    assert not any("AUC" in str(item.message) for item in caught)
    assert not any("y_pred contains classes not in y_true" in str(item.message) for item in caught)

