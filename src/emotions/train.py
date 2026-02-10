"""
GNN-specific training functions for emotion prediction.

This module contains training, evaluation, and fold-level logic for
SpatioTemporalHeteroGNN models. Use train_combined.py as the main script.
"""

import sys
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd

from emotions.model import SpatioTemporalHeteroGNN
from emotions.metrics import compute_metrics
from emotions.splits import SubjectLOOSplitter, RecordingLOOSplitter, CombinedLOOSplitter
from emotions.utils import Logger
from data.data import SpacioTemporalDataset


def train_epoch(model, loader, optimizer, device, grad_clip_max_norm=1.0):
    """Train model for one epoch.
    
    Args:
        model: GNN model to train
        loader: DataLoader with training data
        optimizer: Optimizer
        device: Device to train on
        grad_clip_max_norm: Maximum gradient norm for clipping
        
    Returns:
        Average loss over epoch
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        target = data.y.view(-1, 4)
        
        loss = F.mse_loss(out, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device, emotion_names=None, save_outputs=False, 
            save_dir=None, pair_aggregation_fn=np.mean):
    """Evaluate the model and compute comprehensive metrics.

    Args:
        model: GNN model to evaluate
        loader: DataLoader with test data
        device: Device to run evaluation on
        emotion_names: List of emotion names (optional)
        save_outputs: Whether to save outputs
        save_dir: Directory to save outputs
        pair_aggregation_fn: Function to aggregate per-pair metrics (default: np.mean)

    Returns:
        Dictionary with 'standard' and 'per_pair_aggregated' metrics
    """
    model.eval()
    total_loss = 0

    all_outputs = []
    all_targets = []
    all_metadata = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            target = data.y.view(-1, 4)

            loss = F.mse_loss(out, target)
            total_loss += loss.item()

            all_outputs.append(out.cpu())
            all_targets.append(target.cpu())

            # Collect metadata (subject, recording) for each sample in batch
            if hasattr(data, 'idx'):
                batch_indices = data.idx.cpu().numpy().flatten()
                for graph_idx in batch_indices:
                    try:
                        original_graph = loader.dataset[int(graph_idx)]
                        if hasattr(original_graph, 'subject') and hasattr(original_graph, 'recording'):
                            all_metadata.append((original_graph.subject, original_graph.recording))
                        else:
                            all_metadata.append(None)
                    except (AttributeError, IndexError, KeyError):
                        all_metadata.append(None)

    # Concatenate all outputs and targets
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Validate metadata
    if len(all_metadata) == 0:
        print("Warning: No metadata collected. Per-pair metrics will be disabled.")
        all_metadata = None
    elif len(all_metadata) != len(outputs):
        print(f"Warning: metadata length ({len(all_metadata)}) != outputs length ({len(outputs)}). Using None.")
        all_metadata = None
    elif any(m is None for m in all_metadata):
        print("Warning: Some metadata entries are None. Disabling per-pair metrics.")
        all_metadata = None

    # Compute comprehensive metrics
    metrics = compute_metrics(
        outputs,
        targets,
        emotion_names=emotion_names,
        metadata=all_metadata,
        pair_aggregation_fn=pair_aggregation_fn
    )

    # Add loss to metrics
    avg_loss = total_loss / len(loader)
    metrics['standard']['aggregated']['loss'] = avg_loss
    if metrics['per_pair_aggregated'] is not None:
        metrics['per_pair_aggregated']['aggregated']['loss'] = avg_loss

    if save_outputs and save_dir:
        torch.save({
            'outputs': outputs,
            'targets': targets,
            'metadata': all_metadata,
            'metrics': metrics
        }, save_dir)

    return metrics


def train_gnn_fold(config: dict, train_idx: np.ndarray, val_idx: np.ndarray, 
                   test_idx: np.ndarray, dataset, fold_dir: str, 
                   test_name: str, device: torch.device) -> dict:
    """Train GNN model for one cross-validation fold.
    
    Args:
        config: Full configuration dictionary
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        dataset: Graph dataset
        fold_dir: Directory to save fold results
        test_name: Name of test fold for logging
        device: Device to train on
    
    Returns:
        Dictionary of test metrics
    """
    gnn_cfg = config['gnn']
    training_cfg = gnn_cfg['training']
    model_cfg = gnn_cfg['model']
    logging_cfg = config['logging']
    
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    emotion_names = dataset.emotion_names if hasattr(dataset, 'emotion_names') else None
    
    train_loader = DataLoader(train_dataset, batch_size=training_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=training_cfg['batch_size'], shuffle=False)
    
    model = SpatioTemporalHeteroGNN(
        in_channels=model_cfg['in_channels'],
        hidden_channels=model_cfg['hidden_channels'],
        out_channels=model_cfg['out_channels'],
        output_scale=model_cfg.get('output_scale', 10.0),
        use_preprocess_mlp=model_cfg.get('use_preprocess_mlp', True),
        add_self_loops=model_cfg.get('add_self_loops', False),
        dropout_mlp=model_cfg.get('dropout_mlp', 0.1),
        dropout_gnn=model_cfg.get('dropout_gnn', 0.1),
        dropout_head=model_cfg.get('dropout_head', 0.1),
        aggr=model_cfg.get('aggr', 'mean'),
        conv_type=model_cfg.get('conv_type', 'GCNConv')
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])
    
    num_epochs = training_cfg['num_epochs']
    best_val_loss = float('inf')
    
    save_interval = logging_cfg.get('save_outputs_interval', 10)
    save_epochs = set(range(save_interval, num_epochs + 1, save_interval))
    
    gnn_fold_dir = os.path.join(fold_dir, 'gnn')
    os.makedirs(gnn_fold_dir, exist_ok=True)
    
    print(f"Training GNN for {test_name}...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, 
                                training_cfg['grad_clip_max_norm'])
        val_metrics = evaluate(model, val_loader, device, emotion_names=emotion_names)
        val_loss = val_metrics['standard']['aggregated']['loss']
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(gnn_fold_dir, 'best_model.pt'))
        
        if epoch in save_epochs or epoch == num_epochs:
            print(f"  Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(os.path.join(gnn_fold_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device, emotion_names=emotion_names)
    
    # Clean up
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return test_metrics


def save_strategy_results_csv(config_file: str, run_dir: str, 
                               all_strategies_results: dict, metric_names: list) -> str:
    """Save per-config comparison results to CSV.

    Args:
        config_file: Path to config file
        run_dir: Directory to save results
        all_strategies_results: Dictionary of results per strategy
        metric_names: List of metric names to include
    
    Returns:
        Path to saved CSV file
    """
    csv_data = []
    for strategy, test_metrics in all_strategies_results.items():
        row = {'strategy': strategy}

        first_fold = next(iter(test_metrics.values()))
        has_pair_metrics = 'per_pair_aggregated' in first_fold and first_fold['per_pair_aggregated'] is not None

        # ===== STANDARD METRICS =====
        # Aggregated metrics (standard)
        final_metrics_agg_std = {f'std_agg_{metric}': np.nanmean([test_metrics[fold_id]['standard']['aggregated'][metric] for fold_id in test_metrics]) for metric in metric_names}
        row.update(final_metrics_agg_std)

        # Per-emotion metrics (standard)
        if test_metrics:
            if 'per_emotion' in first_fold['standard'] and first_fold['standard']['per_emotion']:
                for emo_name in first_fold['standard']['per_emotion'].keys():
                    emo_metrics_std = {f'std_{emo_name}_{metric}': np.nanmean([test_metrics[fold_id]['standard']['per_emotion'][emo_name][metric] for fold_id in test_metrics]) for metric in metric_names}
                    row.update(emo_metrics_std)

        # ===== PER-PAIR AGGREGATED METRICS =====
        if has_pair_metrics:
            # Aggregated metrics (per-pair)
            final_metrics_agg_pair = {f'pair_agg_{metric}': np.nanmean([test_metrics[fold_id]['per_pair_aggregated']['aggregated'][metric] for fold_id in test_metrics if test_metrics[fold_id]['per_pair_aggregated'] is not None]) for metric in metric_names}
            row.update(final_metrics_agg_pair)

            # Per-emotion metrics (per-pair)
            if 'per_emotion' in first_fold['per_pair_aggregated'] and first_fold['per_pair_aggregated']['per_emotion']:
                for emo_name in first_fold['per_pair_aggregated']['per_emotion'].keys():
                    emo_metrics_pair = {f'pair_{emo_name}_{metric}': np.nanmean([test_metrics[fold_id]['per_pair_aggregated']['per_emotion'][emo_name][metric] for fold_id in test_metrics if test_metrics[fold_id]['per_pair_aggregated'] is not None]) for metric in metric_names}
                    row.update(emo_metrics_pair)

        csv_data.append(row)

    config_basename = os.path.splitext(os.path.basename(config_file))[0]
    csv_filename = f"{config_basename}.csv"
    csv_path = os.path.join(run_dir, csv_filename)
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison metrics to: {csv_path}")
    return csv_path


def save_all_configs_comparison_csv(config_files, all_configs_results):
    """Save final comparison across all configurations to CSV.
    
    Args:
        config_files: List of config file paths
        all_configs_results: Dictionary of results across all configs
    
    Returns:
        Path to saved CSV file
    """
    final_csv_data = []
    for config_file, config_data in all_configs_results.items():
        config_basename = os.path.splitext(os.path.basename(config_file))[0]
        for strategy, metrics in config_data['results'].items():
            row = {
                'config': config_basename,
                'strategy': strategy,
                'run_dir': config_data['run_dir'],
                'timestamp': config_data['timestamp']
            }
            row.update(metrics)
            final_csv_data.append(row)
    
    final_df = pd.DataFrame(final_csv_data)
    
    parent_results_dir = os.path.dirname(all_configs_results[config_files[0]]['run_dir'])
    final_csv_path = os.path.join(parent_results_dir, f"all_configs_comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    final_df.to_csv(final_csv_path, index=False)
    
    print(f"\nSaved final comparison across all configurations to: {final_csv_path}")
    print("\nSummary:")
    print(final_df.to_string(index=False))
    
    return final_csv_path


def print_strategy_summary(strategy, test_metrics, metric_names):
    """Print summary statistics for a single strategy.

    Args:
        strategy: Name of the strategy
        test_metrics: Dictionary of test metrics
        metric_names: List of metric names to display
    """
    print("\n" + "="*100)
    fold_type = "Subjects" if strategy == 'subject_loo' else "Recordings" if strategy == 'recording_loo' else "Subject-Recording Pairs"
    print(f"Summary for {strategy.upper()} (Averaged Across {fold_type})")
    print("="*100)

    # Check if we have per_pair_aggregated metrics
    first_fold = next(iter(test_metrics.values()))
    has_pair_metrics = 'per_pair_aggregated' in first_fold and first_fold['per_pair_aggregated'] is not None

    # ========== STANDARD METRICS ==========
    print("\n[STANDARD APPROACH: Concatenate all predictions, then compute metrics]")

    # Aggregated metrics (standard)
    final_metrics_agg_std = {metric: np.nanmean([test_metrics[fold_id]['standard']['aggregated'][metric] for fold_id in test_metrics]) for metric in metric_names}
    metric_str = " | ".join([f"{metric.upper()}: {final_metrics_agg_std[metric]:.4f}" for metric in metric_names])
    print(f"\nAggregated: {metric_str}")

    # Per-emotion metrics (standard)
    if test_metrics:
        if 'per_emotion' in first_fold['standard'] and first_fold['standard']['per_emotion']:
            print("\nPer-emotion:")
            for emo_name in first_fold['standard']['per_emotion'].keys():
                emo_metrics = {metric: np.nanmean([test_metrics[fold_id]['standard']['per_emotion'][emo_name][metric] for fold_id in test_metrics]) for metric in metric_names}
                emo_str = " | ".join([f"{metric.upper()}: {emo_metrics[metric]:.4f}" for metric in metric_names])
                print(f"  {emo_name}: {emo_str}")

    # ========== PER-PAIR AGGREGATED METRICS ==========
    if has_pair_metrics:
        print("\n" + "-"*100)
        print("[PER-PAIR AGGREGATED: Compute per (subject, recording) pair, then aggregate]")

        # Aggregated metrics (per-pair)
        final_metrics_agg_pair = {metric: np.nanmean([test_metrics[fold_id]['per_pair_aggregated']['aggregated'][metric] for fold_id in test_metrics if test_metrics[fold_id]['per_pair_aggregated'] is not None]) for metric in metric_names}
        metric_str_pair = " | ".join([f"{metric.upper()}: {final_metrics_agg_pair[metric]:.4f}" for metric in metric_names])
        print(f"\nAggregated: {metric_str_pair}")

        # Per-emotion metrics (per-pair)
        if 'per_emotion' in first_fold['per_pair_aggregated'] and first_fold['per_pair_aggregated']['per_emotion']:
            print("\nPer-emotion:")
            for emo_name in first_fold['per_pair_aggregated']['per_emotion'].keys():
                emo_metrics_pair = {metric: np.nanmean([test_metrics[fold_id]['per_pair_aggregated']['per_emotion'][emo_name][metric] for fold_id in test_metrics if test_metrics[fold_id]['per_pair_aggregated'] is not None]) for metric in metric_names}
                emo_str_pair = " | ".join([f"{metric.upper()}: {emo_metrics_pair[metric]:.4f}" for metric in metric_names])
                print(f"  {emo_name}: {emo_str_pair}")



def print_strategies_comparison(all_strategies_results, metric_names):
    """Print comparison table across all strategies.

    Args:
        all_strategies_results: Dictionary of results per strategy
        metric_names: List of metric names to display
    """
    print("\n" + "="*100)
    print("FINAL COMPARISON ACROSS ALL STRATEGIES")
    print("="*100)

    # Check if we have per_pair_aggregated metrics
    first_strategy = next(iter(all_strategies_results.values()))
    first_fold = next(iter(first_strategy.values()))
    has_pair_metrics = 'per_pair_aggregated' in first_fold and first_fold['per_pair_aggregated'] is not None

    # ========== STANDARD METRICS ==========
    print("\n[STANDARD APPROACH: Concatenate all predictions, then compute metrics]")
    print("\nAggregated Metrics:")
    print(f"{'Strategy':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
    print("-"*100)

    for strategy, test_metrics in all_strategies_results.items():
        final_metrics = {metric: np.nanmean([test_metrics[fold_id]['standard']['aggregated'][metric] for fold_id in test_metrics]) for metric in metric_names}
        metric_str = " | ".join([f"{final_metrics[m]:<10.4f}" for m in metric_names])
        print(f"{strategy:<20} | {metric_str}")

    # Per-emotion comparison (standard)
    if all_strategies_results:
        if 'per_emotion' in first_fold['standard'] and first_fold['standard']['per_emotion']:
            emotion_names = list(first_fold['standard']['per_emotion'].keys())
            for emo_name in emotion_names:
                print(f"\n{emo_name}:")
                print(f"{'Strategy':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
                print("-"*100)
                for strategy, test_metrics in all_strategies_results.items():
                    emo_metrics = {metric: np.nanmean([test_metrics[fold_id]['standard']['per_emotion'][emo_name][metric] for fold_id in test_metrics]) for metric in metric_names}
                    emo_str = " | ".join([f"{emo_metrics[m]:<10.4f}" for m in metric_names])
                    print(f"{strategy:<20} | {emo_str}")

    # ========== PER-PAIR AGGREGATED METRICS ==========
    if has_pair_metrics:
        print("\n" + "="*100)
        print("[PER-PAIR AGGREGATED: Compute per (subject, recording) pair, then aggregate]")
        print("\nAggregated Metrics:")
        print(f"{'Strategy':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
        print("-"*100)

        for strategy, test_metrics in all_strategies_results.items():
            final_metrics_pair = {metric: np.nanmean([test_metrics[fold_id]['per_pair_aggregated']['aggregated'][metric] for fold_id in test_metrics if test_metrics[fold_id]['per_pair_aggregated'] is not None]) for metric in metric_names}
            metric_str_pair = " | ".join([f"{final_metrics_pair[m]:<10.4f}" for m in metric_names])
            print(f"{strategy:<20} | {metric_str_pair}")

        # Per-emotion comparison (per-pair)
        if 'per_emotion' in first_fold['per_pair_aggregated'] and first_fold['per_pair_aggregated']['per_emotion']:
            emotion_names_pair = list(first_fold['per_pair_aggregated']['per_emotion'].keys())
            for emo_name in emotion_names_pair:
                print(f"\n{emo_name}:")
                print(f"{'Strategy':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
                print("-"*100)
                for strategy, test_metrics in all_strategies_results.items():
                    emo_metrics_pair = {metric: np.nanmean([test_metrics[fold_id]['per_pair_aggregated']['per_emotion'][emo_name][metric] for fold_id in test_metrics if test_metrics[fold_id]['per_pair_aggregated'] is not None]) for metric in metric_names}
                    emo_str_pair = " | ".join([f"{emo_metrics_pair[m]:<10.4f}" for m in metric_names])
                    print(f"{strategy:<20} | {emo_str_pair}")



def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Use default config path
        config_path = os.path.join(os.path.dirname(__file__), "train_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_splitter(strategy: str, dataset, val_size: int, random_state: int = None):
    """Create a splitter based on strategy name.
    
    Args:
        strategy: Name of splitting strategy ('subject_loo', 'recording_loo', 'combined_loo')
        dataset: Dataset to split
        val_size: Number of subjects/recordings for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Splitter instance
    """
    if strategy == 'subject_loo':
        return SubjectLOOSplitter(dataset, val_size=val_size, random_state=random_state)
    elif strategy == 'recording_loo':
        return RecordingLOOSplitter(dataset, val_size=val_size, random_state=random_state)
    elif strategy == 'combined_loo':
        return CombinedLOOSplitter(dataset, val_size=val_size, random_state=random_state)
    else:
        raise ValueError(f"Unknown cross-validation strategy: {strategy}. "
                        f"Valid options: 'subject_loo', 'recording_loo', 'combined_loo'")


def parse_args():

    parser = argparse.ArgumentParser(description="Train SpatioTemporalHeteroGNN for emotion prediction")
    parser.add_argument("--config", type=str, default=None, 
                       help="(Optional) Path to config YAML file")
    parser.add_argument("--configs_dir", type=str, default=None,
                       help="(Optional) Directory containing multiple config YAML files for batch runs")
    parser.add_argument("--configs", type=str, nargs='*', default=None,
                       help="(Optional) List of config YAML files for multiple runs")
    return parser.parse_args()

def get_config_files(args):
    if args.configs is not None:
        return args.configs
    elif args.configs_dir is not None:
        # List all YAML files in the directory
        config_files = [os.path.join(args.configs_dir, f) for f in os.listdir(args.configs_dir) if f.endswith('.yaml') or f.endswith('.yml')]
        if not config_files:
            raise ValueError(f"No YAML config files found in directory: {args.configs_dir}")
        return config_files
    elif args.config is not None:
        if args.configs_dir is not None:
            return [os.path.join(args.configs_dir, args.config)]
        else:
            return [args.config]
    else:
        raise ValueError("Please provide at least one config file using --config, --configs, or --configs_dir argument.")

def main():
    # Parse command-line arguments
    args = parse_args()

    config_files = get_config_files(args)

    # Dictionary to store results across all config files
    all_configs_results = {}

    outer_start_time = datetime.now()
    print(f"Training started at: {outer_start_time}")
    
    for config_file in config_files:
        print(f"\n{'#'*100}\nStarting training with config: {config_file}\n{'#'*100}\n")

        # Load configuration
        config = load_config(config_file)
        
        # Extract configuration sections
        dataset_cfg = config['dataset']
        model_cfg = config['model']
        training_cfg = config['training']
        cv_cfg = config['cross_validation']
        logging_cfg = config['logging']
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(logging_cfg['results_dir'], timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # Set up logging to file
        log_file = os.path.join(run_dir, 'training_log.txt')
        logger = Logger(log_file)
        sys.stdout = logger
        sys.stderr = logger

        # Save config to run directory for reproducibility
        config_save_path = os.path.join(run_dir, os.path.basename(config_file))
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved configuration to: {config_save_path}")
        print(f"Logging to: {log_file}")

        # Device
        if training_cfg['device'] == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(training_cfg['device'])
        print(f"Using device: {device}")
        print(f"Results will be saved to: {run_dir}")
        
        # Set random seed for reproducibility
        if training_cfg.get('random_seed') is not None:
            torch.manual_seed(training_cfg['random_seed'])
            np.random.seed(training_cfg['random_seed'])
        
        # Load dataset
        print("Loading dataset...")
        dataset = SpacioTemporalDataset(
            root_dir=dataset_cfg['data_dir'],
            file_list=dataset_cfg.get('file_list'),
            recursive=dataset_cfg['recursive'],
            ignore_dirs=dataset_cfg.get('ignore_dirs'),
            window_length=dataset_cfg['window_length'],
            window_overlap=dataset_cfg['window_overlap'],
            kt=dataset_cfg['kt'],
            ks=dataset_cfg['ks'],
            cache_dir=dataset_cfg.get('cache_dir'),
            use_cache=dataset_cfg.get('use_cache', True),
            dropping_emotion_threshold=dataset_cfg.get('dropping_emotion_threshold', -1),
        )

        # Get strategies (can be single string or list)
        strategies = cv_cfg['strategies']
        if isinstance(strategies, str):
            strategies = [strategies]
        
        print(f"\nWill run experiments with {len(strategies)} splitting strateg{'y' if len(strategies) == 1 else 'ies'}: {', '.join(strategies)}")

        
        # Dictionary to store results for all strategies
        all_strategies_results = {}
        
        # Iterate over all splitting strategies
        for strategy in strategies:
            print(f"\n{'='*100}")
            print(f"Starting cross-validation with strategy: {strategy.upper()}")
            print(f"{'='*100}")
            
            # Initialize splitter for this strategy
            splitter = create_splitter(
                strategy=strategy,
                dataset=dataset,
                val_size=cv_cfg['val_size'],
                random_state=cv_cfg.get('random_state')
            )
            
            # Create strategy-specific directory
            strategy_dir = os.path.join(run_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)
            
            test_metrics = {}
            for train_idx, val_idx, test_idx in splitter.split():
                
                # Identify test set by strategy
                if strategy == 'subject_loo':
                    test_subjects = list(set(dataset[i].subject for i in test_idx))
                    if len(test_subjects) != 1:
                        raise ValueError(f"Test split contains multiple subjects: {test_subjects}. "
                                    f"Expected all test samples to be from the same subject.")
                    test_id = test_subjects[0]
                    test_name = f"subject_{test_id}"
                elif strategy == 'recording_loo':
                    test_recordings = list(set(dataset[i].recording for i in test_idx))
                    if len(test_recordings) != 1:
                        raise ValueError(f"Test split contains multiple recordings: {test_recordings}. "
                                    f"Expected all test samples to be from the same recording.")
                    test_id = test_recordings[0]
                    test_name = f"recording_{test_id}"
                elif strategy == 'combined_loo':
                    test_pairs = list(set((dataset[i].subject, dataset[i].recording) for i in test_idx))
                    if len(test_pairs) != 1:
                        raise ValueError(f"Test split contains multiple subject-recording pairs: {test_pairs}. "
                                    f"Expected all test samples to be from the same pair.")
                    test_id = test_pairs[0]
                    test_name = f"subject_{test_id[0]}_recording_{test_id[1]}"
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                # Create test-specific directory within strategy directory
                fold_dir = os.path.join(strategy_dir, test_name)
                fold_data_dir = os.path.join(fold_dir, "data")
                os.makedirs(fold_data_dir, exist_ok=True)

                train_dataset = [dataset[i] for i in train_idx]
                val_dataset = [dataset[i] for i in val_idx]
                test_dataset = [dataset[i] for i in test_idx]
                
                # Get emotion names from dataset
                emotion_names = dataset.emotion_names if hasattr(dataset, 'emotion_names') else None
                
                print(f"Train: {len(train_dataset)} graphs | Val: {len(val_dataset)} graphs | Test: {len(test_dataset)} graphs")
                
                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=training_cfg['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=training_cfg['batch_size'], shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=training_cfg['batch_size'], shuffle=False)
                
                # Initialize model
                model = SpatioTemporalHeteroGNN(
                    in_channels=model_cfg['in_channels'],
                    hidden_channels=model_cfg['hidden_channels'],
                    out_channels=model_cfg['out_channels'],
                    output_scale=model_cfg.get('output_scale', 10.0), 
                    use_preprocess_mlp=model_cfg.get('use_preprocess_mlp', True),
                    add_self_loops=model_cfg.get('add_self_loops', False),
                    dropout_mlp=model_cfg.get('dropout_mlp', 0.1),
                    dropout_gnn=model_cfg.get('dropout_gnn', 0.1),
                    dropout_head=model_cfg.get('dropout_head', 0.1),
                    aggr=model_cfg.get('aggr', 'mean'),
                    conv_type=model_cfg.get('conv_type', 'GCNConv')
                ).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])
                
                # Training loop
                num_epochs = training_cfg['num_epochs']
                best_val_loss = float('inf')
                
                # Determine epochs to save outputs (10% of total, equidistant)
                if logging_cfg.get('save_outputs_interval') is not None:
                    save_interval = logging_cfg['save_outputs_interval']
                else:
                    save_interval = max(1, num_epochs // 10)
                save_epochs = set(range(save_interval, num_epochs + 1, save_interval))
                
                print(f"\nStarting training for {test_name}...")
                fold_start_time = datetime.now()
                for epoch in range(1, num_epochs + 1):
                    epoch_start_time = datetime.now()
                    train_loss = train_epoch(
                        model, train_loader, optimizer, device, 
                        grad_clip_max_norm=training_cfg['grad_clip_max_norm']
                    )
                    
                    # Save outputs for selected epochs
                    save_outputs = logging_cfg.get('save_validation_outputs', False) and epoch in save_epochs
                    save_path = os.path.join(fold_data_dir, f'epoch_{epoch:03d}.pt') if save_outputs else None
                    val_metrics = evaluate(model, val_loader, device, emotion_names=emotion_names, save_outputs=save_outputs, save_dir=save_path)
                    val_loss = val_metrics['standard']['aggregated']['loss']

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if logging_cfg.get('save_best_model', True):
                            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))

                    print_every = logging_cfg.get('print_every', 10)
                    if epoch % print_every == 0 or epoch == 1:
                        agg = val_metrics['standard']['aggregated']
                        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val MSE: {agg['mse']:.4f} | "
                            f"MAE: {agg['mae']:.4f} | Spearman rho: {agg['spearman']:.4f} | CCC: {agg['ccc']:.4f} "
                            f" | Time: {datetime.now() - epoch_start_time}")
                
                test_metrics[test_id] = evaluate(model, test_loader, device, emotion_names=emotion_names, save_outputs=False, save_dir=None) 
                # print(f"Test Metrics for {test_name}: "
                #     f"MSE: {test_metrics[test_id]['mse']:.4f} | "
                #     f"MAE: {test_metrics[test_id]['mae']:.4f} | "
                # print(f"Time taken for {test_name}: {datetime.now() - fold_start_time}\n")
                
                # Clean up GPU memory after each fold
                del model, optimizer, train_loader, val_loader, test_loader
                del train_dataset, val_dataset, test_dataset
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Store results for this strategy
            all_strategies_results[strategy] = test_metrics
            
            # Print summary for this strategy
            metric_names = config['metrics']
            print_strategy_summary(strategy, test_metrics, metric_names)
            
            # Additional cleanup after strategy completes
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Final comparison across all strategies
        if len(strategies) > 1:
            metric_names = config['metrics']
            print_strategies_comparison(all_strategies_results, metric_names)
            
            # Save to CSV
            save_strategy_results_csv(config_file, run_dir, all_strategies_results, metric_names)
        
        # Store results for this config file (both standard and per-pair)
        config_results = {}
        for strategy, test_metrics in all_strategies_results.items():
            # Standard metrics
            final_metrics_std = {f'std_{metric}': np.nanmean([test_metrics[fold_id]['standard']['aggregated'][metric] for fold_id in test_metrics]) for metric in config['metrics']}

            # Per-pair aggregated metrics (if available)
            first_fold = next(iter(test_metrics.values()))
            if 'per_pair_aggregated' in first_fold and first_fold['per_pair_aggregated'] is not None:
                final_metrics_pair = {f'pair_{metric}': np.nanmean([test_metrics[fold_id]['per_pair_aggregated']['aggregated'][metric] for fold_id in test_metrics if test_metrics[fold_id]['per_pair_aggregated'] is not None]) for metric in config['metrics']}
                final_metrics_std.update(final_metrics_pair)

            config_results[strategy] = final_metrics_std
        all_configs_results[config_file] = {
            'run_dir': run_dir,
            'timestamp': timestamp,
            'results': config_results
        }
        
        print(f"\nTraining complete!")
        print(f"Results saved to: {run_dir}")
        
        # Close logger and restore stdout
        sys.stdout = logger.terminal
        sys.stderr = sys.__stderr__
        logger.close()
    
    # After all configs are processed, create a final comparison CSV
    if len(config_files) > 1:
        print("\n" + "="*100)
        print("FINAL COMPARISON ACROSS ALL CONFIGURATIONS")
        print("="*100)
        
        save_all_configs_comparison_csv(config_files, all_configs_results)
    
    outer_end_time = datetime.now()
    print(f"\nAll training complete!")
    print(f"Total time taken: {outer_end_time - outer_start_time}")

if __name__ == "__main__":
    main()