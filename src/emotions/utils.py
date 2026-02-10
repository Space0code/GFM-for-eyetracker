"""
Shared utilities for emotion prediction training.

Includes: logging, config management, data loading, splitter creation, results export.
"""

import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from emotions.splits import SubjectLOOSplitter, RecordingLOOSplitter, CombinedLOOSplitter
from emotions.baseline_model import get_baseline_by_name


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


class TabularWindowSample:
    """Lightweight sample container for tabular windows with metadata."""
    
    def __init__(self, features: Dict[str, float], targets: Dict[str, float], 
                 subject: str, recording: str):
        self.features = features
        self.targets = targets
        self.subject = subject
        self.recording = recording


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
            from emotions.baseline_model import get_baseline_by_name
            
            for model_name in baseline_cfg['models']:
                try:
                    # Test that model exists
                    get_baseline_by_name(model_name)
                except ValueError as e:
                    raise ValueError(f"Invalid baseline model in config: {e}")
    
    # Validate metrics
    valid_metrics = ['mse', 'mae', 'sd_error', 'spearman', 'ccc', 'r2']
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


def parse_subject_recording_from_name(filename: str) -> Tuple[str, str]:
    """Parse subject and recording IDs from filename.
    
    Examples:
        'sample_01_recording_01_merged.csv' -> ('sample_01', 'recording_01')
        's_001.csv' -> ('s_001', 'unknown')
    
    Args:
        filename: CSV filename
        
    Returns:
        Tuple of (subject_id, recording_id)
    """
    name = os.path.basename(filename)
    subject = None
    recording = None
    
    try:
        parts = name.replace('.csv', '').split('_')
        
        for i, part in enumerate(parts):
            if part == 'sample' and i + 1 < len(parts):
                subject = f"sample_{parts[i + 1]}"
            elif part == 'recording' and i + 1 < len(parts):
                recording = f"recording_{parts[i + 1]}"
        
        if subject is None:
            subject = name.replace('.csv', '')
        if recording is None:
            recording = 'unknown'
            
    except Exception:
        subject = name.replace('.csv', '')
        recording = 'unknown'
    
    return subject, recording


def aggregate_window(window_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate window data into statistical features.
    
    Mirrors TabularDataset aggregation logic.
    
    Args:
        window_df: DataFrame of eye-tracking data for one window
        
    Returns:
        Dictionary of feature values and emotion targets
    """
    feats: Dict[str, float] = {}
    
    # Gaze features
    for col in ['x-avg', 'y-avg']:
        if col in window_df.columns:
            feats[f'{col}_mean'] = window_df[col].mean()
            feats[f'{col}_std'] = window_df[col].std()
            feats[f'{col}_min'] = window_df[col].min()
            feats[f'{col}_max'] = window_df[col].max()
    
    # Pupil features
    for col in ['pupil-size-left-avg', 'pupil-size-right-avg']:
        if col in window_df.columns:
            feats[f'{col}_mean'] = window_df[col].mean()
            feats[f'{col}_std'] = window_df[col].std()
    
    # Confidence
    for col in ['confidence-gaze-left', 'confidence-gaze-right']:
        if col in window_df.columns:
            feats[f'{col}_mean'] = window_df[col].mean()
    
    # Emotion targets
    targets: Dict[str, float] = {}
    for col in window_df.columns:
        if 'emotion' in col.lower():
            targets[col] = window_df[col].iloc[-1]
    
    return {**feats, **targets}


def drop_pairs_with_emotions_below_threshold(df: pd.DataFrame, 
                                             emotion_cols: Optional[List[str]] = None,
                                             threshold: float = -1) -> pd.DataFrame:
    """Drop rows where all emotion values are below or equal to threshold.
    
    Args:
        df: DataFrame with emotion columns
        emotion_cols: List of emotion column names (auto-detected if None)
        threshold: Threshold value
        
    Returns:
        Filtered DataFrame
    """
    if emotion_cols is None:
        emotion_cols = [c for c in df.columns if 'emotion' in c.lower()]
    
    all_below = (df[emotion_cols] <= threshold).all(axis=1)
    filtered_df = df[~all_below].reset_index(drop=True)
    
    return filtered_df


def build_tabular_samples(data_dir: str, file_list: Optional[List[str]] = None, 
                         window_length: int = 10, 
                         dropping_emotion_threshold: float = -1) -> List[TabularWindowSample]:
    """Load CSVs and build windowed samples with subject/recording metadata.
    
    Args:
        data_dir: Directory containing CSV files
        file_list: Optional list of specific file names to load
        window_length: Window size in seconds
        dropping_emotion_threshold: Drop pairs where all emotions <= this value
        
    Returns:
        List of TabularWindowSample objects
    """
    root = Path(data_dir)
    files = [root / f for f in file_list] if file_list else list(root.glob('*.csv'))
    samples: List[TabularWindowSample] = []

    for fpath in tqdm(files, desc="Loading data files"):
        df = pd.read_csv(fpath)
        df = df.dropna()
        if len(df) == 0:
            continue
        
        # Drop if all emotions below threshold
        emotion_cols = [c for c in df.columns if 'emotion' in c.lower()]
        if emotion_cols and dropping_emotion_threshold > -np.inf:
            df = drop_pairs_with_emotions_below_threshold(
                df, emotion_cols, dropping_emotion_threshold
            )
            if len(df) == 0:
                continue
        
        subject, recording = parse_subject_recording_from_name(str(fpath))
        time_col = 'time-rel-seconds'
        max_time = df[time_col].max()
        start_time = 0
        
        while start_time < max_time:
            end_time = start_time + window_length
            window_data = df[(df[time_col] >= start_time) & (df[time_col] < end_time)]
            
            if len(window_data) > 10:
                agg = aggregate_window(window_data)
                
                # Separate features and targets
                features = {k: v for k, v in agg.items() if 'emotion' not in k.lower()}
                targets = {k: v for k, v in agg.items() if 'emotion' in k.lower()}
                
                samples.append(TabularWindowSample(features, targets, subject, recording))
            
            start_time += window_length

    return samples


def samples_to_xy(samples: List[TabularWindowSample], 
                  indices: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                  List[Tuple[str, str]], 
                                                  List[str], List[str]]:
    """Convert selected samples to X, y DataFrames with metadata.
    
    Args:
        samples: List of TabularWindowSample objects
        indices: Array of sample indices to select
        
    Returns:
        Tuple of (X, y, metadata, feature_cols, target_cols)
        - X: Feature DataFrame
        - y: Target DataFrame
        - metadata: List of (subject, recording) tuples
        - feature_cols: Feature column names
        - target_cols: Target column names
    """
    sel = [samples[int(i)] for i in indices]
    if not sel:
        raise ValueError("No samples selected")

    feat_cols = sorted(sel[0].features.keys())
    target_cols = sorted(sel[0].targets.keys())

    X = pd.DataFrame(
        [[s.features.get(c, np.nan) for c in feat_cols] for s in sel],
        columns=feat_cols
    )
    y = pd.DataFrame(
        [[s.targets.get(c, np.nan) for c in target_cols] for s in sel],
        columns=target_cols
    )
    metadata = [(s.subject, s.recording) for s in sel]

    return X, y, metadata, feat_cols, target_cols


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


def train_baselines_fold(baseline_cfg: Dict[str, Any], train_idx: np.ndarray, 
                         val_idx: np.ndarray, test_idx: np.ndarray, 
                         tabular_samples: List, fold_dir: str, 
                         metric_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Train all baseline models for one cross-validation fold.
    
    Args:
        baseline_cfg: Baseline configuration dict with 'models' and 'hyperparameters'
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        tabular_samples: List of TabularWindowSample objects
        fold_dir: Directory to save fold results
        metric_names: List of metric names to compute
    
    Returns:
        Dictionary mapping baseline names to their test metrics
    """
    X_train, y_train, metadata_train, feat_cols, target_cols = samples_to_xy(tabular_samples, train_idx)
    X_val, y_val, metadata_val, _, _ = samples_to_xy(tabular_samples, val_idx)
    X_test, y_test, metadata_test, _, _ = samples_to_xy(tabular_samples, test_idx)
    
    # Get baseline models from config
    selected_models = baseline_cfg.get('models', [])
    hyperparams = baseline_cfg.get('hyperparameters', {})
    
    baseline_results = {}
    
    for model_name in selected_models:
        # Get model with hyperparameters
        model_hyperparams = hyperparams.get(model_name, {})
        baseline = get_baseline_by_name(model_name, **model_hyperparams)
        
        # Train
        baseline.fit(X_train, y_train)
        
        # Evaluate
        test_metrics = baseline.evaluate(
            X_test, y_test, 
            emotion_names=target_cols, 
            metadata=metadata_test
        )
        baseline_results[baseline.name] = test_metrics
        
        # Save model
        model_dir = os.path.join(fold_dir, baseline.name)
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(baseline, f)
        
        # Save predictions
        y_pred = baseline.predict(X_test)
        np.save(os.path.join(model_dir, 'y_pred.npy'), y_pred)
        np.save(os.path.join(model_dir, 'y_true.npy'), y_test.to_numpy())
    
    return baseline_results
