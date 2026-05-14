"""Metrics for multiclass classification experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "cpu"):
        return data.cpu().numpy()
    if hasattr(data, "values"):
        return data.values
    return np.asarray(data)


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy without warning-producing edge cases."""
    labels = np.unique(y_true)
    if len(labels) < 2:
        return float("nan")

    recalls: List[float] = []
    for label in labels:
        mask = y_true == label
        denom = int(np.sum(mask))
        if denom == 0:
            continue
        recalls.append(float(np.sum(y_pred[mask] == label) / denom))

    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def _safe_ovr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray, classes: np.ndarray, average: str) -> float:
    """Compute OVR AUC robustly, returning NaN when undefined.

    This avoids terminal spam from per-fold UndefinedMetricWarning.
    """
    supports = np.array([np.sum(y_true == label) for label in classes], dtype=float)
    per_class_auc: List[float] = []
    per_class_weight: List[float] = []

    for class_idx, class_label in enumerate(classes):
        y_true_binary = (y_true == class_label).astype(int)
        positives = int(np.sum(y_true_binary))
        negatives = int(len(y_true_binary) - positives)
        if positives == 0 or negatives == 0:
            continue

        try:
            auc_value = float(roc_auc_score(y_true_binary, y_pred_proba[:, class_idx]))
        except ValueError:
            continue

        per_class_auc.append(auc_value)
        per_class_weight.append(float(supports[class_idx]))

    if not per_class_auc:
        return float("nan")

    auc_arr = np.asarray(per_class_auc, dtype=float)
    if average == "macro":
        return float(np.mean(auc_arr))
    if average == "weighted":
        weight_arr = np.asarray(per_class_weight, dtype=float)
        total_weight = float(np.sum(weight_arr))
        if total_weight <= 0:
            return float("nan")
        return float(np.sum(auc_arr * weight_arr) / total_weight)

    raise ValueError(f"Unsupported average='{average}' for OVR AUC")


def compute_multiclass_metrics(
    y_pred_proba: Any,
    y_true: Any,
    class_labels: List[int],
) -> Dict[str, float]:
    """Compute requested multiclass metrics for one prediction set."""
    y_true_np = _to_numpy(y_true).reshape(-1).astype(int)
    y_pred_proba_np = _to_numpy(y_pred_proba)

    if y_pred_proba_np.ndim != 2:
        raise ValueError("y_pred_proba must be 2D: [n_samples, n_classes].")

    classes = np.asarray(class_labels, dtype=int)
    if y_pred_proba_np.shape[1] != len(classes):
        raise ValueError(
            "Probability columns do not match class label count: "
            f"{y_pred_proba_np.shape[1]} vs {len(classes)}"
        )

    pred_idx = np.argmax(y_pred_proba_np, axis=1)
    y_pred = classes[pred_idx]

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "balanced_accuracy": _safe_balanced_accuracy(y_true_np, y_pred),
    }

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred,
        labels=classes,
        average="weighted",
        zero_division=0,
    )

    metrics.update(
        {
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "macro_auc_ovr": _safe_ovr_auc(y_true_np, y_pred_proba_np, classes=classes, average="macro"),
            "weighted_auc_ovr": _safe_ovr_auc(
                y_true_np,
                y_pred_proba_np,
                classes=classes,
                average="weighted",
            ),
        }
    )

    return metrics


def _aggregate_pairs(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    metadata: Dict[str, List[Any]],
) -> tuple[np.ndarray, np.ndarray]:
    subjects = metadata.get("subjects", [])
    recordings = metadata.get("recordings", [])
    if len(subjects) != len(y_true) or len(recordings) != len(y_true):
        raise ValueError("Metadata length mismatch for pair aggregation.")

    pair_data: Dict[tuple[str, str], Dict[str, Any]] = {}
    for idx, (subject, recording) in enumerate(zip(subjects, recordings)):
        key = (str(subject), str(recording))
        if key not in pair_data:
            pair_data[key] = {"pred": [], "true": []}
        pair_data[key]["pred"].append(y_pred_proba[idx])
        pair_data[key]["true"].append(int(y_true[idx]))

    pair_pred: List[np.ndarray] = []
    pair_true: List[int] = []
    for item in pair_data.values():
        pred_mean = np.mean(np.vstack(item["pred"]), axis=0)
        labels = np.asarray(item["true"], dtype=int)
        values, counts = np.unique(labels, return_counts=True)
        majority = int(values[np.argmax(counts)])
        pair_pred.append(pred_mean)
        pair_true.append(majority)

    return np.vstack(pair_pred), np.asarray(pair_true, dtype=int)


def evaluate_multiclass_classification(
    y_pred_proba: Any,
    y_true: Any,
    class_labels: List[int],
    metadata: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """Comprehensive multiclass evaluation with optional pair aggregation."""
    y_pred_proba_np = _to_numpy(y_pred_proba)
    y_true_np = _to_numpy(y_true).reshape(-1).astype(int)

    standard_metrics = compute_multiclass_metrics(
        y_pred_proba=y_pred_proba_np,
        y_true=y_true_np,
        class_labels=class_labels,
    )

    result: Dict[str, Any] = {
        "standard": {
            "aggregated": standard_metrics,
            "per_emotion": {},
        },
        "per_pair_aggregated": None,
    }

    if metadata is not None and "subjects" in metadata and "recordings" in metadata:
        try:
            pair_pred, pair_true = _aggregate_pairs(
                y_pred_proba=y_pred_proba_np,
                y_true=y_true_np,
                metadata=metadata,
            )
            pair_metrics = compute_multiclass_metrics(
                y_pred_proba=pair_pred,
                y_true=pair_true,
                class_labels=class_labels,
            )
            result["per_pair_aggregated"] = {
                "aggregated": pair_metrics,
                "per_emotion": {},
            }
            result["pair_aggregated"] = result["per_pair_aggregated"]
        except Exception:  # pragma: no cover - defensive fallback
            result["pair_aggregated"] = None

    return result
