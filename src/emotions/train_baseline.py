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
from data.data import clean_dataset


class TabularWindowSample:
    """Lightweight sample container for tabular windows with metadata."""
    
    def __init__(self, features: Dict[str, float], targets: Dict[str, float], 
                 subject: str, recording: str):
        self.features = features
        self.targets = targets
        self.subject = subject
        self.recording = recording


def infer_default_target_columns(df: pd.DataFrame) -> List[str]:
    """Infer numeric emotion target columns from a DataFrame."""
    targets: List[str] = []
    for col in df.columns:
        if not col.startswith("emotion-"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            targets.append(col)
    return sorted(targets)


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


def aggregate_window(
    window_df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    target_aggregation: str = "last",
) -> Dict[str, float]:
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
    
    # Target labels
    if target_columns is None:
        target_columns = infer_default_target_columns(window_df)

    targets: Dict[str, float] = {}
    for col in target_columns:
        if col in window_df.columns:
            if target_aggregation == "mean":
                targets[col] = float(window_df[col].mean())
            elif target_aggregation == "last":
                targets[col] = float(window_df[col].iloc[-1])
            else:
                raise ValueError(
                    f"Unsupported target_aggregation='{target_aggregation}'. "
                    "Use 'mean' or 'last'."
                )
    
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
                         window_overlap: float = 0.0,
                         min_samples_per_window: int = 11,
                         dropping_emotion_threshold: float = -1,
                         feature_columns: Optional[List[str]] = None,
                         target_columns: Optional[List[str]] = None,
                         target_aggregation: str = "last",
                         dropna_columns: Optional[List[str]] = None,
                         experiment_type_column: str = "experiment-type",
                         allowed_experiment_types: Optional[List[str]] = None,
                         label_quality_column: Optional[str] = None,
                         allowed_label_quality_values: Optional[List[str]] = None) -> List[TabularWindowSample]:
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
    if target_aggregation not in {"mean", "last"}:
        raise ValueError(
            f"Unsupported target_aggregation='{target_aggregation}'. Use 'mean' or 'last'."
        )
    if not (0 <= window_overlap < 1):
        raise ValueError("window_overlap must be in [0, 1).")
    if min_samples_per_window <= 0:
        raise ValueError("min_samples_per_window must be > 0.")
    feature_columns = feature_columns or [
        "x-avg",
        "y-avg",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
    ]

    def _apply_dataset_filters(df: pd.DataFrame) -> pd.DataFrame:
        """Apply dataset-level filters that do not depend on grouping."""
        if allowed_experiment_types and experiment_type_column in df.columns:
            df = df[df[experiment_type_column].isin(allowed_experiment_types)]
        if label_quality_column and allowed_label_quality_values and label_quality_column in df.columns:
            df = df[df[label_quality_column].isin(allowed_label_quality_values)]
        return df.reset_index(drop=True)

    def _clean_and_dropna(df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning/dropna on one logical sequence (group/file)."""
        if len(df) == 0:
            return df.reset_index(drop=True)

        df = df.sort_values("time-rel-seconds").reset_index(drop=True)
        required_clean_cols = ["time-rel-seconds"] + feature_columns
        missing_required = [col for col in required_clean_cols if col not in df.columns]
        if not missing_required:
            df = clean_dataset(
                df,
                required_cols=required_clean_cols,
                interpolation_cols=feature_columns,
            )

        if dropna_columns is None:
            df = df.dropna()
        else:
            missing = [col for col in dropna_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing configured dropna columns: {missing}")
            df = df.dropna(subset=dropna_columns)
        return df.reset_index(drop=True)

    def _iter_window_slices(group_df: pd.DataFrame):
        """Generate time-based slices aligned with graph dataset windowing."""
        times = group_df["time-rel-seconds"].values
        if len(times) == 0:
            return

        start_time = times[0]
        end_time = times[-1]
        step_size = window_length * (1 - window_overlap)
        current_start = start_time

        while current_start < end_time:
            current_end = min(current_start + window_length, end_time)
            start_idx = np.searchsorted(times, current_start, side="left")
            end_idx = np.searchsorted(times, current_end, side="right")
            if end_idx > start_idx:
                yield slice(start_idx, end_idx)
            current_start += step_size
    
    if data_filepath is not None:
        # New behavior: load from single CSV file
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"Data file not found: {data_filepath}")
        
        df = pd.read_csv(data_filepath)
        df = _apply_dataset_filters(df)
        
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
            group_df = _clean_and_dropna(group_df)
            if len(group_df) == 0:
                continue

            missing_features = [col for col in feature_columns if col not in group_df.columns]
            if missing_features:
                raise ValueError(f"Missing feature columns in group {(subject, recording)}: {missing_features}")
            
            # Drop if all emotions below threshold
            resolved_target_cols = target_columns or infer_default_target_columns(group_df)
            if resolved_target_cols and dropping_emotion_threshold > -np.inf:
                group_df = drop_pairs_with_emotions_below_threshold(
                    group_df, resolved_target_cols, dropping_emotion_threshold
                )
                if len(group_df) == 0:
                    continue
            
            for window_slice in _iter_window_slices(group_df):
                window_data = group_df.iloc[window_slice]
                if len(window_data) < min_samples_per_window:
                    continue

                agg = aggregate_window(
                    window_data,
                    target_columns=resolved_target_cols,
                    target_aggregation=target_aggregation,
                )

                # Separate features and targets
                features = {k: v for k, v in agg.items() if k not in resolved_target_cols}
                targets = {k: v for k, v in agg.items() if k in resolved_target_cols}

                samples.append(TabularWindowSample(features, targets, subject, recording))
    
    else:
        # Old behavior: load from directory
        root = Path(data_dir)
        files = [root / f for f in file_list] if file_list else list(root.glob('*.csv'))

        for fpath in tqdm(files, desc="Loading data files"):
            df = pd.read_csv(fpath)
            df = _apply_dataset_filters(df)
            df = _clean_and_dropna(df)
            if len(df) == 0:
                continue
            
            # Drop if all emotions below threshold
            resolved_target_cols = target_columns or infer_default_target_columns(df)
            if resolved_target_cols and dropping_emotion_threshold > -np.inf:
                df = drop_pairs_with_emotions_below_threshold(
                    df, resolved_target_cols, dropping_emotion_threshold
                )
                if len(df) == 0:
                    continue
            
            if "subject" in df.columns and "recording" in df.columns:
                subject = df["subject"].iloc[0]
                recording = df["recording"].iloc[0]
            else:
                subject, recording = parse_subject_recording_from_name(str(fpath))
            for window_slice in _iter_window_slices(df):
                window_data = df.iloc[window_slice]
                if len(window_data) < min_samples_per_window:
                    continue

                agg = aggregate_window(
                    window_data,
                    target_columns=resolved_target_cols,
                    target_aggregation=target_aggregation,
                )

                # Separate features and targets
                features = {k: v for k, v in agg.items() if k not in resolved_target_cols}
                targets = {k: v for k, v in agg.items() if k in resolved_target_cols}

                samples.append(TabularWindowSample(features, targets, subject, recording))

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
