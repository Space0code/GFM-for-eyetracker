"""
Baseline model training functions for emotion prediction.

This module contains data preparation, training, and fold-level logic for
baseline models. Use train.py as the main script.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from emotions.baseline_model import get_baseline_by_name


class TabularWindowSample:
    """Lightweight sample container for tabular windows with metadata."""
    
    def __init__(self, features: Dict[str, float], targets: Dict[str, float], 
                 subject: str, recording: str):
        self.features = features
        self.targets = targets
        self.subject = subject
        self.recording = recording


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


def build_tabular_samples(data_dir: str = None, data_filepath: str = None,
                         filter_subjects: list = None, filter_recordings: list = None,
                         file_list: Optional[List[str]] = None, 
                         window_length: int = 10, 
                         dropping_emotion_threshold: float = -1) -> List[TabularWindowSample]:
    """Load CSVs and build windowed samples with subject/recording metadata.
    
    Args:
        data_dir: Directory containing CSV files (mutually exclusive with data_filepath)
        data_filepath: Single CSV file with all data (mutually exclusive with data_dir)
        filter_subjects: List of subject IDs to include (only with data_filepath)
        filter_recordings: List of recording IDs to include (only with data_filepath)
        file_list: Optional list of specific file names to load (only with data_dir)
        window_length: Window size in seconds
        dropping_emotion_threshold: Drop pairs where all emotions <= this value
        
    Returns:
        List of TabularWindowSample objects
    """
    # Validate input
    if (data_dir is None) == (data_filepath is None):
        raise ValueError("Must provide exactly one of: data_dir or data_filepath")
    
    samples: List[TabularWindowSample] = []
    
    if data_filepath is not None:
        # New behavior: load from single CSV file
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"Data file not found: {data_filepath}")
        
        df = pd.read_csv(data_filepath)
        df = df.dropna()
        
        # Check required columns
        if 'subject' not in df.columns or 'recording' not in df.columns:
            raise ValueError(f"CSV must contain 'subject' and 'recording' columns")
        
        # Apply filters
        if filter_subjects is not None:
            df = df[df['subject'].isin(filter_subjects)]
        if filter_recordings is not None:
            df = df[df['recording'].isin(filter_recordings)]
        
        if len(df) == 0:
            raise ValueError("No data remaining after applying filters")
        
        # Group by (subject, recording) and process each group
        grouped = df.groupby(['subject', 'recording'])
        for (subject, recording), group_df in tqdm(grouped, desc="Loading data groups"):
            group_df = group_df.reset_index(drop=True)
            
            # Drop if all emotions below threshold
            emotion_cols = [c for c in group_df.columns if 'emotion' in c.lower()]
            if emotion_cols and dropping_emotion_threshold > -np.inf:
                group_df = drop_pairs_with_emotions_below_threshold(
                    group_df, emotion_cols, dropping_emotion_threshold
                )
                if len(group_df) == 0:
                    continue
            
            time_col = 'time-rel-seconds'
            max_time = group_df[time_col].max()
            start_time = 0
            
            while start_time < max_time:
                end_time = start_time + window_length
                window_data = group_df[(group_df[time_col] >= start_time) & (group_df[time_col] < end_time)]
                
                if len(window_data) > 10:
                    agg = aggregate_window(window_data)
                    
                    # Separate features and targets
                    features = {k: v for k, v in agg.items() if 'emotion' not in k.lower()}
                    targets = {k: v for k, v in agg.items() if 'emotion' in k.lower()}
                    
                    samples.append(TabularWindowSample(features, targets, subject, recording))
                
                start_time += window_length
    
    else:
        # Old behavior: load from directory
        root = Path(data_dir)
        files = [root / f for f in file_list] if file_list else list(root.glob('*.csv'))

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


def train_baselines_fold(baseline_cfg: Dict[str, Any], train_idx: np.ndarray, 
                         val_idx: np.ndarray, test_idx: np.ndarray, 
                         tabular_samples: List[TabularWindowSample], fold_dir: str, 
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
