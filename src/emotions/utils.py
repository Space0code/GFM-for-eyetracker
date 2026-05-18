"""
Shared utilities for emotion prediction training.

Includes: logging, config management, splitter creation, results export.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from emotions.splits import (
    CombinedLOOSplitter,
    SubjectKFoldSplitter,
    RecordingKFoldSplitter,
    RecordingLOOSplitter,
    SubjectLOOSplitter,
)


class Logger:
    """Logger that writes to both console and a timestamped file."""
    
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
        self.log_writer = TimestampedLineWriter(self.log)
    
    def write(self, message: str):
        self.terminal.write(message)
        self.log_writer.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class TimestampedLineWriter:
    """Write text fragments with one HH:MM:SS prefix per output row."""

    def __init__(self, handle):
        self.handle = handle
        self._log_at_line_start = True

    def write(self, message: str) -> None:
        """Write a possibly partial message, timestamping each new row."""
        for chunk in message.splitlines(keepends=True):
            if self._log_at_line_start:
                self.handle.write(f"[{datetime.now().strftime('%H:%M:%S')}] ")
            self.handle.write(chunk)
            self._log_at_line_start = chunk.endswith(("\n", "\r"))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file with validation.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {e}")
    
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]):
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If validation fails
    """
    # Check required top-level keys
    required_keys = ['dataset', 'cross_validation', 'logging', 'metrics']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
    
    # Validate dataset
    dataset_cfg = config['dataset']
    if 'data_dir' not in dataset_cfg and 'data_filepath' not in dataset_cfg:
        raise ValueError("Missing 'data_dir' or 'data_filepath' in dataset config")
    
    # Validate dropping_emotion_threshold
    threshold = dataset_cfg.get('dropping_emotion_threshold', -1)
    if not isinstance(threshold, (int, float)):
        raise ValueError(
            f"dropping_emotion_threshold must be numeric, got: {type(threshold)}"
        )
    
    # Validate CV strategies
    cv_cfg = config['cross_validation']
    strategies = cv_cfg.get('strategies', [])
    if isinstance(strategies, str):
        strategies = [strategies]
    
    valid_strategies = [
        'subject_loo',
        'recording_loo',
        'combined_loo',
        'subject_kfold',
        'recording_kfold',
    ]
    for strategy in strategies:
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid CV strategy: {strategy}. "
                f"Valid options: {', '.join(valid_strategies)}"
            )
    
    # Validate baseline models if present
    if 'baselines' in config:
        baseline_cfg = config['baselines']
        if 'models' in baseline_cfg:
            # Import here to avoid circular dependency. Multiclass baselines have
            # their own factory because they expose predict_proba classifiers.
            if 'multiclass_task' in config:
                from emotions.multiclass.baseline_model_multiclass import (
                    get_multiclass_baseline_by_name as get_baseline_by_name,
                )
            else:
                from emotions.baseline_model import get_baseline_by_name
            
            for model_name in baseline_cfg['models']:
                try:
                    # Test that model exists
                    get_baseline_by_name(model_name)
                except ValueError as e:
                    raise ValueError(f"Invalid baseline model in config: {e}")
    
    # # Validate metrics
    # valid_metrics = ['mse', 'mae', 'sd_error', 'spearman', 'ccc']
    # for metric in config['metrics']:
    #     if metric not in valid_metrics:
    #         raise ValueError(
    #             f"Invalid metric: {metric}. "
    #             f"Valid options: {', '.join(valid_metrics)}"
    #         )


def create_splitter(
    strategy: str,
    samples,
    val_size: int = 1,
    random_state: Optional[int] = None,
    n_splits: int = 3,
):
    """Create cross-validation splitter.
    
    Args:
        strategy: CV strategy ('subject_loo', 'recording_loo', 'combined_loo',
            'subject_kfold', 'recording_kfold')
        samples: Dataset or list of samples with .subject and .recording attributes
        val_size: Number of subjects/recordings for validation
        random_state: Random seed
        n_splits: Number of folds for k-fold strategies
        
    Returns:
        Splitter instance
        
    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == 'subject_loo':
        return SubjectLOOSplitter(samples, val_size, random_state)
    elif strategy == 'recording_loo':
        return RecordingLOOSplitter(samples, val_size, random_state)
    elif strategy == 'combined_loo':
        return CombinedLOOSplitter(samples, val_size, random_state)
    elif strategy == "subject_kfold":
        return SubjectKFoldSplitter(
            samples,
            n_splits=n_splits,
            val_size=val_size,
            shuffle=True,
            random_state=random_state,
        )
    elif strategy == "recording_kfold":
        return RecordingKFoldSplitter(
            samples,
            n_splits=n_splits,
            val_size=val_size,
            shuffle=True,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown CV strategy: {strategy}. "
            "Valid options: subject_loo, recording_loo, combined_loo, "
            "subject_kfold, recording_kfold"
        )


def _resolve_approach_block(
    fold_result: Dict[str, Any],
    approach: str,
) -> Optional[Dict[str, Any]]:
    """Resolve metric block for one fold with backward-compatible key aliases."""
    if approach in fold_result and fold_result[approach] is not None:
        return fold_result[approach]

    aliases = {
        "per_pair_aggregated": ["pair_aggregated"],
        "pair_aggregated": ["per_pair_aggregated"],
    }
    for alias in aliases.get(approach, []):
        if alias in fold_result and fold_result[alias] is not None:
            return fold_result[alias]
    return None


def _collect_metric_values(
    model_results: Dict[str, Any],
    approach: str,
    metric_name: str,
) -> List[float]:
    """Collect one metric across folds with missing-key tolerance."""
    values: List[float] = []
    for fold_result in model_results.values():
        block = _resolve_approach_block(fold_result, approach)
        if block is None:
            continue
        aggregated = block.get("aggregated", {})
        values.append(float(aggregated.get(metric_name, np.nan)))
    return values


def _safe_nanmean(values: List[float]) -> float:
    """Compute mean over finite values only, returning NaN when undefined.

    Avoids RuntimeWarning spam from np.nanmean on all-NaN slices.
    """
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def save_comparison_csv(results: Dict[str, Dict[str, Any]],
                       metric_names: List[str],
                       output_path: str,
                       approach: str = 'standard'):
    """Save cross-validation comparison results to CSV.
    
    Args:
        results: Dict mapping model/strategy names to their fold results
        metric_names: List of metrics to include
        output_path: Path to save CSV file
        approach: 'standard' or 'per_pair_aggregated'
    """
    rows = []
    
    for model_name, model_results in results.items():
        avg_agg = {}
        for metric in metric_names:
            values = _collect_metric_values(model_results, approach, metric)
            avg_agg[metric] = _safe_nanmean(values)

        rows.append({'model': model_name, 'metric_type': 'aggregated', **avg_agg})

        # Per-emotion metrics (if available)
        first_block = None
        for fold_result in model_results.values():
            first_block = _resolve_approach_block(fold_result, approach)
            if first_block is not None:
                break

        if not first_block:
            continue

        per_emotion = first_block.get("per_emotion") or {}
        for emo_name in per_emotion.keys():
            avg_emo = {}
            for metric in metric_names:
                values = []
                for fold_result in model_results.values():
                    block = _resolve_approach_block(fold_result, approach)
                    if block is None:
                        continue
                    metric_value = (
                        block.get("per_emotion", {})
                        .get(emo_name, {})
                        .get(metric, np.nan)
                    )
                    values.append(float(metric_value))
                avg_emo[metric] = _safe_nanmean(values)

            rows.append({
                'model': model_name,
                'metric_type': f'emotion_{emo_name}',
                **avg_emo
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison to: {output_path}")


def save_fold_metrics_csv(results: Dict[str, Dict[str, Any]],
                          metric_names: List[str],
                          output_path: str,
                          approach: str = 'standard'):
    """Save per-fold cross-validation metrics to CSV.

    Args:
        results: Dict mapping model/strategy names to their fold results
        metric_names: List of metrics to include
        output_path: Path to save CSV file
        approach: 'standard' or 'per_pair_aggregated'
    """
    rows = []

    for model_name, model_results in results.items():
        for fold_id, fold_result in model_results.items():
            block = _resolve_approach_block(fold_result, approach)
            if block is None:
                continue

            aggregated = block.get("aggregated", {})
            rows.append({
                "model": model_name,
                "fold_id": fold_id,
                "metric_type": "aggregated",
                **{metric: float(aggregated.get(metric, np.nan)) for metric in metric_names},
            })

            per_emotion = block.get("per_emotion") or {}
            for emo_name, emotion_metrics in per_emotion.items():
                rows.append({
                    "model": model_name,
                    "fold_id": fold_id,
                    "metric_type": f"emotion_{emo_name}",
                    **{metric: float(emotion_metrics.get(metric, np.nan)) for metric in metric_names},
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved fold metrics to: {output_path}")


def print_comparison_table(results: Dict[str, Dict[str, Any]], 
                          metric_names: List[str],
                          title: str = "Results"):
    """Print comparison table for cross-validation results.
    
    Args:
        results: Dict mapping model names to their fold results
        metric_names: List of metrics to display
        title: Table title
    """
    print("\n" + "="*100)
    print(title)
    print("="*100)
    
    has_pair_metrics = False
    for model_results in results.values():
        for fold_result in model_results.values():
            if _resolve_approach_block(fold_result, "per_pair_aggregated") is not None:
                has_pair_metrics = True
                break
        if has_pair_metrics:
            break
    
    # Standard metrics
    print("\n[STANDARD: Concatenate predictions, then compute]")
    print(f"{'Model':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
    print("-"*100)
    
    for model_name, model_results in results.items():
        avg_metrics = {}
        for metric in metric_names:
            values = _collect_metric_values(model_results, "standard", metric)
            avg_metrics[metric] = _safe_nanmean(values)
        metric_str = " | ".join([f"{avg_metrics[m]:<10.4f}" for m in metric_names])
        print(f"{model_name:<20} | {metric_str}")
    
    # Per-pair aggregated metrics
    if has_pair_metrics:
        print("\n[PER-PAIR: Compute per pair, then aggregate]")
        print(f"{'Model':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
        print("-"*100)
        
        for model_name, model_results in results.items():
            avg_metrics = {}
            for metric in metric_names:
                values = _collect_metric_values(model_results, "per_pair_aggregated", metric)
                avg_metrics[metric] = _safe_nanmean(values)
            metric_str = " | ".join([f"{avg_metrics[m]:<10.4f}" for m in metric_names])
            print(f"{model_name:<20} | {metric_str}")
