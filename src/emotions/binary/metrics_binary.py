"""
Binary classification metrics for emotion recognition.

Provides standard classification metrics: accuracy, precision, recall, F1, AUC-ROC.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Any, Optional, List


def _to_numpy(data: Any) -> np.ndarray:
    """Convert various data types to numpy array."""
    if hasattr(data, 'cpu'):  # torch.Tensor
        return data.cpu().numpy()
    elif hasattr(data, 'values'):  # pandas DataFrame/Series
        return data.values
    else:
        return np.array(data)


def compute_binary_metrics(
    y_pred: Any,
    y_true: Any,
    threshold: float = 0.5,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_pred: Predicted probabilities (will be thresholded at 0.5)
        y_true: Ground truth binary labels (0 or 1)
        threshold: Decision threshold for converting probabilities to predictions
        average: Averaging strategy for precision/recall/f1
        
    Returns:
        Dictionary of metric values
    """
    y_pred = _to_numpy(y_pred).flatten()
    y_true = _to_numpy(y_true).flatten()
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = y_true.astype(int)
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    
    # Precision, Recall, F1
    # Handle cases where a class might not be present
    try:
        metrics['precision'] = precision_score(
            y_true_binary, y_pred_binary, 
            average=average, zero_division=0
        )
        metrics['recall'] = recall_score(
            y_true_binary, y_pred_binary,
            average=average, zero_division=0
        )
        metrics['f1'] = f1_score(
            y_true_binary, y_pred_binary,
            average=average, zero_division=0
        )
    except ValueError as e:
        # If only one class is present in y_true
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
    
    # AUC-ROC (requires probabilities, not binary predictions)
    try:
        # Check if we have both classes in ground truth
        if len(np.unique(y_true_binary)) > 1:
            metrics['auc'] = roc_auc_score(y_true_binary, y_pred)
        else:
            metrics['auc'] = 0.0
    except (ValueError, IndexError):
        metrics['auc'] = 0.0
    
    # Confusion matrix components
    # Specify labels=[0, 1] to ensure proper shape even if one class is missing
    try:
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    except ValueError:
        # Handle edge case where confusion matrix can't be computed
        metrics['true_negatives'] = 0
        metrics['false_positives'] = 0
        metrics['false_negatives'] = 0
        metrics['true_positives'] = 0
    
    return metrics


def evaluate_binary_classification(
    y_pred: Any,
    y_true: Any,
    metadata: Optional[Dict[str, List]] = None,
    emotion_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    pair_aggregation_fn = np.mean
) -> Dict[str, Any]:
    """
    Comprehensive binary classification evaluation with subject/recording aggregation.
    
    Args:
        y_pred: Predicted probabilities
        y_true: Ground truth binary labels
        metadata: Optional dict with 'subjects' and 'recordings' lists
        emotion_names: Optional list with single emotion name
        threshold: Decision threshold
        pair_aggregation_fn: Function to aggregate metrics across subjects/recordings
        
    Returns:
        Dictionary with standard and aggregated metrics
    """
    y_pred = _to_numpy(y_pred)
    y_true = _to_numpy(y_true)
    
    # Ensure 1D arrays
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    
    # Standard metrics (sample-level)
    standard_metrics = compute_binary_metrics(y_pred, y_true, threshold=threshold)
    
    result = {
        'standard': {
            'aggregated': standard_metrics,
            'per_emotion': {
                emotion_names[0] if emotion_names else 'emotion_0': standard_metrics
            }
        }
    }
    
    # Pair-aggregated metrics if metadata provided
    if metadata is not None and 'subjects' in metadata and 'recordings' in metadata:
        subjects = metadata['subjects']
        recordings = metadata['recordings']
        
        # Group predictions by (subject, recording) pairs
        pairs = {}
        for i, (subj, rec) in enumerate(zip(subjects, recordings)):
            key = (subj, rec)
            if key not in pairs:
                pairs[key] = {'y_pred': [], 'y_true': []}
            pairs[key]['y_pred'].append(y_pred[i])
            pairs[key]['y_true'].append(y_true[i])
        
        # Compute metrics for each pair, then aggregate
        pair_metrics = []
        for key, data in pairs.items():
            pred_agg = pair_aggregation_fn(data['y_pred'])
            true_agg = pair_aggregation_fn(data['y_true'])
            
            # Single-sample metrics for this pair
            pair_metric = compute_binary_metrics(
                np.array([pred_agg]),
                np.array([true_agg]),
                threshold=threshold
            )
            pair_metrics.append(pair_metric)
        
        # Aggregate across pairs
        aggregated = {}
        for metric_name in pair_metrics[0].keys():
            if metric_name not in ['true_negatives', 'false_positives', 
                                    'false_negatives', 'true_positives']:
                values = [pm[metric_name] for pm in pair_metrics]
                aggregated[metric_name] = float(np.mean(values))
            else:
                # Sum confusion matrix components
                values = [pm[metric_name] for pm in pair_metrics]
                aggregated[metric_name] = int(np.sum(values))
        
        result['pair_aggregated'] = {
            'aggregated': aggregated,
            'per_emotion': {
                emotion_names[0] if emotion_names else 'emotion_0': aggregated
            }
        }
    
    return result
