"""
Train and evaluate baseline models with cross-validation using a YAML config.

Usage:
  python src/emotions/train_baseline_cv.py --config src/emotions/configs/train_config_baseline.yaml
"""

import os
import sys
import argparse
import yaml
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from emotions.splits import SubjectLOOSplitter, RecordingLOOSplitter, CombinedLOOSplitter
from emotions.baseline_model import get_all_baselines


class TabularWindowSample:
    """Lightweight sample container for tabular windows with metadata."""
    def __init__(self, features: Dict[str, float], targets: Dict[str, float], subject: str, recording: str):
        self.features = features
        self.targets = targets
        self.subject = subject
        self.recording = recording


def parse_subject_recording_from_name(filename: str) -> Tuple[str, str]:
    """Parse subject and recording IDs from filename like 'sample_01_recording_01_merged.csv'."""
    name = os.path.basename(filename)
    subject = None
    recording = None
    try:
        # Expected pattern: sample_<S>_recording_<R>_merged.csv
        parts = name.split('_')
        # parts: ['sample', '01', 'recording', '01', 'merged.csv'] or similar
        if len(parts) >= 4 and parts[0] == 'sample' and parts[2] == 'recording':
            subject = parts[1]
            recording = parts[3]
    except Exception:
        pass
    return subject or 'unknown', recording or 'unknown'


def aggregate_window(window_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate window data into statistical features (mirrors TabularDataset)."""
    feats: Dict[str, float] = {}
    # Gaze features
    for col in ['x-avg', 'y-avg']:
        if col in window_df.columns:
            feats[f'{col}_mean'] = float(window_df[col].mean())
            feats[f'{col}_std'] = float(window_df[col].std())
            feats[f'{col}_min'] = float(window_df[col].min())
            feats[f'{col}_max'] = float(window_df[col].max())
    # Pupil features
    for col in ['pupil-size-left-avg', 'pupil-size-right-avg']:
        if col in window_df.columns:
            feats[f'{col}_mean'] = float(window_df[col].mean())
            feats[f'{col}_std'] = float(window_df[col].std())
    # Confidence
    for col in ['confidence-gaze-left', 'confidence-gaze-right']:
        if col in window_df.columns:
            feats[f'{col}_mean'] = float(window_df[col].mean())
    # Emotions (targets)
    targets: Dict[str, float] = {}
    for col in window_df.columns:
        if 'emotion' in col.lower():
            targets[col] = float(window_df[col].mean())
    return {**feats, **targets}

def drop_pairs_with_emotions_below_threshold(df: pd.DataFrame, emotion_cols: List[str] = None, threshold: float = -1) -> pd.DataFrame:
    """Drop (subject, recording) pairs where all emotion values are below or equal to threshold."""
    if emotion_cols is None:
        emotion_cols = [col for col in df.columns if 'emotion' in col.lower()]
    all_zero = (df[emotion_cols] <= threshold).all(axis=1)
    filtered_df = df[~all_zero].reset_index(drop=True)
    return filtered_df


def build_tabular_samples(data_dir: str, file_list: List[str], window_length: int, dropping_emotion_threshold: float = -1) -> List[TabularWindowSample]:
    """Load CSVs and build windowed samples with subject/recording metadata."""
    root = Path(data_dir)
    files = [root / f for f in file_list] if file_list else list(root.glob('*.csv'))
    samples: List[TabularWindowSample] = []

    for fpath in files:
        df = pd.read_csv(fpath).dropna()
        df = drop_pairs_with_emotions_below_threshold(df, threshold=dropping_emotion_threshold) # drop rows with all emotions == 0
        if len(df) == 0:
            continue
        subject, recording = parse_subject_recording_from_name(fpath.name)
        time_col = 'time-rel-seconds'
        if time_col not in df.columns:
            # Skip files without the required time column
            continue
        max_time = float(df[time_col].max())
        start_time = 0.0
        while start_time < max_time:
            end_time = start_time + float(window_length)
            window_df = df[(df[time_col] >= start_time) & (df[time_col] < end_time)]
            if len(window_df) > 10:
                agg = aggregate_window(window_df)
                # Split features/targets
                targets = {k: v for k, v in agg.items() if 'emotion' in k.lower()}
                features = {k: v for k, v in agg.items() if 'emotion' not in k.lower()}
                samples.append(TabularWindowSample(features, targets, subject, recording))
            start_time += float(window_length)

    return samples


def samples_to_xy(samples: List[TabularWindowSample], indices: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Convert selected samples to X, y dataframes and column names."""
    sel = [samples[int(i)] for i in indices]
    if not sel:
        empty = pd.DataFrame()
        return empty, empty, [], []
    feat_cols = sorted(sel[0].features.keys())
    target_cols = sorted(sel[0].targets.keys())
    X = pd.DataFrame([[s.features.get(c, np.nan) for c in feat_cols] for s in sel], columns=feat_cols)
    y = pd.DataFrame([[s.targets.get(c, np.nan) for c in target_cols] for s in sel], columns=target_cols)
    return X, y, feat_cols, target_cols


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_splitter(strategy: str, samples: List[TabularWindowSample], val_size: int, random_state: int = None):
    if strategy == 'subject_loo':
        return SubjectLOOSplitter(samples, val_size=val_size, random_state=random_state)
    elif strategy == 'recording_loo':
        return RecordingLOOSplitter(samples, val_size=val_size, random_state=random_state)
    elif strategy == 'combined_loo':
        return CombinedLOOSplitter(samples, val_size=val_size, random_state=random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline models with cross-validation")
    p.add_argument("--config", type=str, required=True, help="Path to baseline YAML config")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    dataset_cfg = config['dataset']
    cv_cfg = config['cross_validation']
    logging_cfg = config['logging']
    metrics_cfg = config['metrics']

    start_time = datetime.now()

    # Timestamped run dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(logging_cfg['results_dir'], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    # Save the config copy
    config_save_path = os.path.join(run_dir, os.path.basename(args.config))
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved configuration to: {config_save_path}")

    # Build samples
    print("Loading tabular samples...")
    samples = build_tabular_samples(
        data_dir=dataset_cfg['data_dir'],
        file_list=dataset_cfg.get('file_list'),
        window_length=dataset_cfg.get('window_length', 10),
        dropping_emotion_threshold=dataset_cfg.get('dropping_emotion_threshold', -1)
    )
    print(f"Total windows: {len(samples)}")

    strategies = cv_cfg['strategies']
    if isinstance(strategies, str):
        strategies = [strategies]
    print(f"Will run strategies: {', '.join(strategies)}")

    # Baselines
    baselines = get_all_baselines()

    # Accumulate final results
    all_strategies_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for strategy in strategies:
        print("\n" + "="*100)
        print(f"Starting CV strategy: {strategy.upper()}")
        print("="*100)

        splitter = create_splitter(
            strategy=strategy,
            samples=samples,
            val_size=cv_cfg['val_size'],
            random_state=cv_cfg.get('random_state')
        )

        strategy_dir = os.path.join(run_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)

        # Per-baseline metrics averaged across folds
        per_baseline_metrics: Dict[str, List[Dict[str, float]]] = {b.name: [] for b in baselines}

        fold_num = 0
        for train_idx, val_idx, test_idx in splitter.split():
            fold_num += 1
            # Prepare data matrices
            X_train, y_train, feat_cols, target_cols = samples_to_xy(samples, train_idx)
            X_val, y_val, _, _ = samples_to_xy(samples, val_idx)
            X_test, y_test, _, _ = samples_to_xy(samples, test_idx)

            print(f"Fold {fold_num}: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

            # Train and evaluate each baseline
            for baseline in baselines:
                # Fit on train only (minimal)
                baseline.fit(X_train, y_train)
                # Evaluate on test (returns nested dict with aggregated and per_emotion)
                test_metrics = baseline.evaluate(X_test, y_test)
                per_baseline_metrics[baseline.name].append(test_metrics)

                # Optionally save model per fold
                model_dir = os.path.join(strategy_dir, baseline.name, f"fold_{fold_num}")
                os.makedirs(model_dir, exist_ok=True)
                with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
                    pickle.dump(baseline, f)
                # Save predictions
                y_pred = baseline.predict(X_test)
                np.save(os.path.join(model_dir, 'y_pred.npy'), y_pred)
                np.save(os.path.join(model_dir, 'y_true.npy'), y_test.to_numpy())

        # Aggregate across folds for this strategy
        strategy_results: Dict[str, Dict[str, Any]] = {}
        for bname, metrics_list in per_baseline_metrics.items():
            if metrics_list:
                # Aggregated metrics
                agg_metrics = {m: float(np.nanmean([mres['aggregated'][m] for mres in metrics_list])) for m in metrics_cfg}
                
                # Per-emotion metrics
                per_emo = {}
                first_result = metrics_list[0]
                if 'per_emotion' in first_result and first_result['per_emotion']:
                    for emo_name in first_result['per_emotion'].keys():
                        per_emo[emo_name] = {m: float(np.nanmean([mres['per_emotion'][emo_name][m] for mres in metrics_list])) for m in metrics_cfg}
                
                strategy_results[bname] = {
                    'aggregated': agg_metrics,
                    'per_emotion': per_emo
                }
            else:
                strategy_results[bname] = {
                    'aggregated': {m: float('nan') for m in metrics_cfg},
                    'per_emotion': {}
                }

        # Save CSV for this strategy
        df_rows = []
        for bname, result_dict in strategy_results.items():
            row = {'baseline': bname}
            # Aggregated metrics with 'agg_' prefix
            for m in metrics_cfg:
                row[f'agg_{m}'] = result_dict['aggregated'][m]
            # Per-emotion metrics
            if result_dict['per_emotion']:
                for emo_name, emo_metrics in result_dict['per_emotion'].items():
                    for m in metrics_cfg:
                        row[f'{emo_name}_{m}'] = emo_metrics[m]
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        csv_path = os.path.join(strategy_dir, 'summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved strategy summary to: {csv_path}")

        # Store for final comparison
        all_strategies_results[strategy] = strategy_results

        # Print concise summary
        print("\nStrategy Summary (averaged across folds):")
        print("\nAggregated Metrics:")
        print(f"{'Baseline':<20} | " + " | ".join([f"{m.upper():<10}" for m in metrics_cfg]))
        print("-"*100)
        for bname, result_dict in strategy_results.items():
            metric_str = " | ".join([f"{result_dict['aggregated'][m]:<10.4f}" for m in metrics_cfg])
            print(f"{bname:<20} | {metric_str}")
        
        # Per-emotion summary
        if strategy_results:
            first_baseline = next(iter(strategy_results.values()))
            if first_baseline['per_emotion']:
                for emo_name in first_baseline['per_emotion'].keys():
                    print(f"\n{emo_name}:")
                    print(f"{'Baseline':<20} | " + " | ".join([f"{m.upper():<10}" for m in metrics_cfg]))
                    print("-"*100)
                    for bname, result_dict in strategy_results.items():
                        emo_str = " | ".join([f"{result_dict['per_emotion'][emo_name][m]:<10.4f}" for m in metrics_cfg])
                        print(f"{bname:<20} | {emo_str}")

    # Final comparison across strategies
    if len(strategies) > 1:
        print("\n" + "="*100)
        print("FINAL COMPARISON ACROSS STRATEGIES (averaged per baseline)")
        print("="*100)
        
        print("\nAggregated Metrics:")
        print(f"{'Strategy':<20} | {'Baseline':<20} | " + " | ".join([f"{m.upper():<10}" for m in metrics_cfg]))
        print("-"*120)
        rows = []
        for strategy, res in all_strategies_results.items():
            for bname, result_dict in res.items():
                metric_str = " | ".join([f"{result_dict['aggregated'][m]:<10.4f}" for m in metrics_cfg])
                print(f"{strategy:<20} | {bname:<20} | {metric_str}")
                row = {'strategy': strategy, 'baseline': bname}
                for m in metrics_cfg:
                    row[f'agg_{m}'] = result_dict['aggregated'][m]
                # Add per-emotion to row
                if result_dict['per_emotion']:
                    for emo_name, emo_metrics in result_dict['per_emotion'].items():
                        for m in metrics_cfg:
                            row[f'{emo_name}_{m}'] = emo_metrics[m]
                rows.append(row)
        
        # Per-emotion comparison
        if all_strategies_results:
            first_strategy = next(iter(all_strategies_results.values()))
            first_baseline = next(iter(first_strategy.values()))
            if first_baseline['per_emotion']:
                for emo_name in first_baseline['per_emotion'].keys():
                    print(f"\n{emo_name}:")
                    print(f"{'Strategy':<20} | {'Baseline':<20} | " + " | ".join([f"{m.upper():<10}" for m in metrics_cfg]))
                    print("-"*120)
                    for strategy, res in all_strategies_results.items():
                        for bname, result_dict in res.items():
                            emo_str = " | ".join([f"{result_dict['per_emotion'][emo_name][m]:<10.4f}" for m in metrics_cfg])
                            print(f"{strategy:<20} | {bname:<20} | {emo_str}")
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(run_dir, 'all_strategies_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved final comparison to: {csv_path}")

    print(f"\nAll results saved to: {run_dir}")
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Total time taken: {duration}")
    
if __name__ == "__main__":
    main()
