"""
Shared metric computation for emotion prediction models.

Provides unified metric computation for both GNN and baseline models.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from typing import Optional, Callable, List, Tuple, Dict, Any


def compute_ccc(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Concordance Correlation Coefficient (CCC).

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        eps: Small value for numerical stability

    Returns:
        CCC value
    """
    pred_mean = np.mean(y_pred, axis=0)
    true_mean = np.mean(y_true, axis=0)
    pred_var = np.var(y_pred, axis=0)
    true_var = np.var(y_true, axis=0)

    if pred_var < eps or true_var < eps:
        return 0.0

    covariance = np.mean((y_pred - pred_mean) * (y_true - true_mean), axis=0)
    ccc = (2 * covariance) / (pred_var + true_var + (pred_mean - true_mean) ** 2)

    return float(ccc)


def _to_numpy(data: Any) -> np.ndarray:
    """Convert various data types to numpy array.

    Handles: torch.Tensor, pandas DataFrame/Series, numpy array
    """
    if hasattr(data, 'cpu'):  # torch.Tensor
        return data.cpu().numpy()
    elif hasattr(data, 'values'):  # pandas DataFrame/Series
        return data.values
    else:
        return np.array(data)


def compute_metrics(
    y_pred: Any,
    y_true: Any,
    emotion_names: Optional[List[str]] = None,
    metadata: Optional[List[Tuple[str, str]]] = None,
    pair_aggregation_fn: Callable = np.mean,
    eps: float = 1e-8,
    min_samples_per_pair: int = 5
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Computes metrics in two ways:
    1. 'standard': Concatenate all predictions/targets, then compute
    2. 'per_pair_aggregated': Compute per (subject, recording) pair, then aggregate

    Args:
        y_pred: Predictions (torch.Tensor, np.array, or pd.DataFrame)
        y_true: Ground truth (torch.Tensor, np.array, or pd.DataFrame)
        emotion_names: List of emotion names (optional)
        metadata: List of (subject, recording) tuples, same length as data (optional)
        pair_aggregation_fn: Function to aggregate per-pair metrics (default: np.mean)
        eps: Small value for numerical stability
        min_samples_per_pair: Minimum samples required per pair (default: 5)

    Returns:
        dict: {'standard': {...}, 'per_pair_aggregated': {...} or None}
    """
    # Convert to numpy
    y_pred_array = _to_numpy(y_pred)
    y_true_array = _to_numpy(y_true)

    # Extract emotion names from DataFrame if available
    if emotion_names is None and hasattr(y_true, 'columns'):
        emotion_names = list(y_true.columns)
    elif emotion_names is None:
        num_emotions = y_pred_array.shape[1] if len(y_pred_array.shape) > 1 else 1
        emotion_names = [f'emotion_{i}' for i in range(num_emotions)]

    # Validate metadata length
    if metadata is not None and len(metadata) != len(y_pred_array):
        raise ValueError(
            f"Metadata length ({len(metadata)}) doesn't match data length ({len(y_pred_array)})"
        )

    num_emotions = y_pred_array.shape[1] if len(y_pred_array.shape) > 1 else 1

    # ========== STANDARD APPROACH ==========
    standard_metrics = _compute_standard_metrics(
        y_pred_array, y_true_array, emotion_names, num_emotions, eps
    )

    # ========== PER-PAIR AGGREGATION ==========
    per_pair_metrics = None
    if metadata is not None and len(metadata) > 0:
        per_pair_metrics = _compute_per_pair_metrics(
            y_pred_array, y_true_array, emotion_names, num_emotions,
            metadata, pair_aggregation_fn, eps, min_samples_per_pair
        )

    return {
        'standard': standard_metrics,
        'per_pair_aggregated': per_pair_metrics
    }


def _compute_standard_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    emotion_names: List[str],
    num_emotions: int,
    eps: float
) -> Dict[str, Any]:
    """Compute standard metrics (concatenate all, then compute)."""
    # Aggregated metrics
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    spearman_val = 0.0
    if np.std(y_true_flat) > eps and np.std(y_pred_flat) > eps:
        spearman_val, _ = spearmanr(y_true_flat, y_pred_flat)

    aggregated = {
        'mse': float(mean_squared_error(y_true_flat, y_pred_flat)),
        'mae': float(mean_absolute_error(y_true_flat, y_pred_flat)),
        'sd_error': float(np.std(y_true_flat - y_pred_flat)),
        'spearman': float(spearman_val),
        'ccc': compute_ccc(y_pred_flat, y_true_flat, eps)
    }

    # Per-emotion metrics
    per_emotion = {}
    for i, emo_name in enumerate(emotion_names[:num_emotions]):
        y_pred_emo = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred
        y_true_emo = y_true[:, i] if len(y_true.shape) > 1 else y_true

        spearman_emo = 0.0
        if np.std(y_true_emo) > eps and np.std(y_pred_emo) > eps:
            spearman_emo, _ = spearmanr(y_true_emo, y_pred_emo)

        per_emotion[emo_name] = {
            'mse': float(mean_squared_error(y_true_emo, y_pred_emo)),
            'mae': float(mean_absolute_error(y_true_emo, y_pred_emo)),
            'sd_error': float(np.std(y_true_emo - y_pred_emo)),
            'spearman': float(spearman_emo),
            'ccc': compute_ccc(y_pred_emo, y_true_emo, eps)
        }

    return {
        'aggregated': aggregated,
        'per_emotion': per_emotion
    }


def _compute_per_pair_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    emotion_names: List[str],
    num_emotions: int,
    metadata: List[Tuple[str, str]],
    pair_aggregation_fn: Callable,
    eps: float,
    min_samples: int
) -> Dict[str, Any]:
    """Compute per-pair metrics, then aggregate."""
    # Group by (subject, recording) pairs
    pairs = {}
    for idx, (subj, rec) in enumerate(metadata):
        key = (subj, rec)
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(idx)

    # Compute metrics for each pair
    pair_aggregated_metrics = []
    pair_per_emotion_metrics = {emo: [] for emo in emotion_names[:num_emotions]}
    skipped_pairs = 0

    for pair_key, indices in pairs.items():
        # Skip pairs with too few samples
        if len(indices) < min_samples:
            skipped_pairs += 1
            continue

        y_pred_pair = y_pred[indices]
        y_true_pair = y_true[indices]

        # Aggregated metrics for this pair
        y_pred_pair_flat = y_pred_pair.flatten()
        y_true_pair_flat = y_true_pair.flatten()

        spearman_pair = 0.0
        if np.std(y_true_pair_flat) > eps and np.std(y_pred_pair_flat) > eps:
            spearman_pair, _ = spearmanr(y_true_pair_flat, y_pred_pair_flat)

        pair_metrics = {
            'mse': float(mean_squared_error(y_true_pair_flat, y_pred_pair_flat)),
            'mae': float(mean_absolute_error(y_true_pair_flat, y_pred_pair_flat)),
            'sd_error': float(np.std(y_true_pair_flat - y_pred_pair_flat)),
            'spearman': float(spearman_pair),
            'ccc': compute_ccc(y_pred_pair_flat, y_true_pair_flat, eps)
        }
        pair_aggregated_metrics.append(pair_metrics)

        # Per-emotion metrics for this pair
        for i, emo_name in enumerate(emotion_names[:num_emotions]):
            y_pred_emo_pair = y_pred_pair[:, i] if len(y_pred_pair.shape) > 1 else y_pred_pair
            y_true_emo_pair = y_true_pair[:, i] if len(y_true_pair.shape) > 1 else y_true_pair

            spearman_emo_pair = 0.0
            if np.std(y_true_emo_pair) > eps and np.std(y_pred_emo_pair) > eps:
                spearman_emo_pair, _ = spearmanr(y_true_emo_pair, y_pred_emo_pair)

            emo_metrics = {
                'mse': float(mean_squared_error(y_true_emo_pair, y_pred_emo_pair)),
                'mae': float(mean_absolute_error(y_true_emo_pair, y_pred_emo_pair)),
                'sd_error': float(np.std(y_true_emo_pair - y_pred_emo_pair)),
                'spearman': float(spearman_emo_pair),
                'ccc': compute_ccc(y_pred_emo_pair, y_true_emo_pair, eps)
            }
            pair_per_emotion_metrics[emo_name].append(emo_metrics)

    if skipped_pairs > 0:
        print(f"        Warning: Skipped {skipped_pairs} pairs with <{min_samples} samples")

    if not pair_aggregated_metrics:
        return None

    # Aggregate across pairs
    per_pair_aggregated = {}
    for metric_name in ['mse', 'mae', 'sd_error', 'spearman', 'ccc']:
        values = [pm[metric_name] for pm in pair_aggregated_metrics]
        per_pair_aggregated[metric_name] = float(pair_aggregation_fn(values))

    per_pair_per_emotion = {}
    for emo_name in emotion_names[:num_emotions]:
        per_pair_per_emotion[emo_name] = {}
        for metric_name in ['mse', 'mae', 'sd_error', 'spearman', 'ccc']:
            values = [pm[metric_name] for pm in pair_per_emotion_metrics[emo_name]]
            per_pair_per_emotion[emo_name][metric_name] = float(pair_aggregation_fn(values))

    return {
        'aggregated': per_pair_aggregated,
        'per_emotion': per_pair_per_emotion
    }
