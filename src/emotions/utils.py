"""
Shared utilities for emotion prediction training.

Includes: logging, config management, splitter creation, results export.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from emotions.splits import SubjectLOOSplitter, RecordingLOOSplitter, CombinedLOOSplitter


class Logger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
    
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


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
    if 'data_dir' not in dataset_cfg:
        raise ValueError("Missing 'data_dir' in dataset config")
    
    data_dir = dataset_cfg['data_dir']
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
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
    
    valid_strategies = ['subject_loo', 'recording_loo', 'combined_loo']
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
            # Import here to avoid circular dependency
            from emotions.baseline_model import get_baseline_by_name
            
            for model_name in baseline_cfg['models']:
                try:
                    # Test that model exists
                    get_baseline_by_name(model_name)
                except ValueError as e:
                    raise ValueError(f"Invalid baseline model in config: {e}")
    
    # Validate metrics
    valid_metrics = ['mse', 'mae', 'sd_error', 'spearman', 'ccc']
    for metric in config['metrics']:
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {metric}. "
                f"Valid options: {', '.join(valid_metrics)}"
            )


def create_splitter(strategy: str, samples, val_size: int = 1, 
                   random_state: Optional[int] = None):
    """Create cross-validation splitter.
    
    Args:
        strategy: CV strategy ('subject_loo', 'recording_loo', 'combined_loo')
        samples: Dataset or list of samples with .subject and .recording attributes
        val_size: Number of subjects/recordings for validation
        random_state: Random seed
        
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
    else:
        raise ValueError(
            f"Unknown CV strategy: {strategy}. "
            f"Valid options: subject_loo, recording_loo, combined_loo"
        )


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
        # Aggregated metrics
        avg_agg = {
            m: np.nanmean([
                model_results[fold][approach]['aggregated'][m] 
                for fold in model_results
            ])
            for m in metric_names
        }
        
        row = {'model': model_name, 'metric_type': 'aggregated', **avg_agg}
        rows.append(row)
        
        # Per-emotion metrics (if available)
        first_fold = next(iter(model_results.values()))
        if 'per_emotion' in first_fold[approach] and first_fold[approach]['per_emotion']:
            for emo_name in first_fold[approach]['per_emotion'].keys():
                avg_emo = {
                    m: np.nanmean([
                        model_results[fold][approach]['per_emotion'][emo_name][m]
                        for fold in model_results
                        if emo_name in model_results[fold][approach]['per_emotion']
                    ])
                    for m in metric_names
                }
                row = {
                    'model': model_name,
                    'metric_type': f'emotion_{emo_name}',
                    **avg_emo
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison to: {output_path}")


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
    
    # Check if we have per_pair_aggregated metrics
    first_model = next(iter(results.values()))
    first_fold = next(iter(first_model.values()))
    has_pair_metrics = (
        'per_pair_aggregated' in first_fold and 
        first_fold['per_pair_aggregated'] is not None
    )
    
    # Standard metrics
    print("\n[STANDARD: Concatenate predictions, then compute]")
    print(f"{'Model':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
    print("-"*100)
    
    for model_name, model_results in results.items():
        avg_metrics = {
            m: np.nanmean([
                model_results[fold]['standard']['aggregated'][m] 
                for fold in model_results
            ])
            for m in metric_names
        }
        metric_str = " | ".join([f"{avg_metrics[m]:<10.4f}" for m in metric_names])
        print(f"{model_name:<20} | {metric_str}")
    
    # Per-pair aggregated metrics
    if has_pair_metrics:
        print("\n[PER-PAIR: Compute per pair, then aggregate]")
        print(f"{'Model':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
        print("-"*100)
        
        for model_name, model_results in results.items():
            avg_metrics = {
                m: np.nanmean([
                    model_results[fold]['per_pair_aggregated']['aggregated'][m]
                    for fold in model_results
                    if model_results[fold]['per_pair_aggregated'] is not None
                ])
                for m in metric_names
            }
            metric_str = " | ".join([f"{avg_metrics[m]:<10.4f}" for m in metric_names])
            print(f"{model_name:<20} | {metric_str}")
