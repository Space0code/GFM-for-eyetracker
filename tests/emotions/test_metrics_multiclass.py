import math

import numpy as np

from emotions.multiclass.metrics_multiclass import compute_multiclass_metrics


def test_multiclass_metrics_include_macro_weighted_and_auc() -> None:
    y_true = np.array([0, 1, 2, 1, 0, 2], dtype=int)
    y_pred_proba = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.10, 0.80, 0.10],
            [0.80, 0.10, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=float,
    )

    metrics = compute_multiclass_metrics(y_pred_proba=y_pred_proba, y_true=y_true, class_labels=[0, 1, 2])

    expected_keys = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_auc_ovr",
        "weighted_auc_ovr",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["accuracy"] > 0.9
    assert metrics["balanced_accuracy"] > 0.9
    assert not math.isnan(metrics["macro_auc_ovr"])
    assert not math.isnan(metrics["weighted_auc_ovr"])


def test_multiclass_auc_is_nan_for_single_class_truth() -> None:
    y_true = np.array([1, 1, 1], dtype=int)
    y_pred_proba = np.array(
        [
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.8, 0.1],
        ],
        dtype=float,
    )

    metrics = compute_multiclass_metrics(y_pred_proba=y_pred_proba, y_true=y_true, class_labels=[0, 1, 2])

    assert math.isnan(metrics["macro_auc_ovr"])
    assert math.isnan(metrics["weighted_auc_ovr"])
