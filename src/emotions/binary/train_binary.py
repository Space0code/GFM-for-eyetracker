"""
Binary classification training script for emotion recognition.

Usage:
  python src/emotions/binary/train_binary.py --config src/emotions/binary/configs/train_binary.yaml
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Add src directory to Python path
src_dir = Path(__file__).resolve().parents[2]  # Go up to src/ directory
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data.data import SpacioTemporalDataset
from emotions.train_baseline import build_tabular_samples, samples_to_xy
from emotions.utils import (
    Logger,
    load_config,
    create_splitter,
    save_comparison_csv,
    print_comparison_table
)
from emotions.binary.data_binary import (
    BinarySpacioTemporalDataset,
    wrap_tabular_samples
)
from emotions.binary.model_binary import BinarySpatioTemporalGNN
from emotions.binary.baseline_model_binary import get_binary_baseline_by_name
from emotions.binary.metrics_binary import evaluate_binary_classification


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary classification models")
    parser.add_argument(
        "--config",
        type=str,
        default="src/emotions/binary/configs/train_binary.yaml",
        help="Path to binary config YAML file"
    )
    return parser.parse_args()


def train_gnn_epoch(model, loader, optimizer, device, grad_clip_max_norm=1.0):
    """Train GNN for one epoch with binary cross-entropy loss."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data).squeeze()  # [batch_size]
        target = data.y.squeeze()    # [batch_size]
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(out, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate_gnn(model, loader, device, emotion_name, threshold=0.5):
    """Evaluate GNN binary classifier."""
    model.eval()
    total_loss = 0
    
    all_outputs = []
    all_targets = []
    all_subjects = []
    all_recordings = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            target = data.y.squeeze()
            
            loss = F.binary_cross_entropy(out, target)
            total_loss += loss.item()
            
            all_outputs.append(out.cpu())
            all_targets.append(target.cpu())
            
            # Collect metadata
            if hasattr(data, 'idx'):
                batch_indices = data.idx.cpu().numpy().flatten()
                for graph_idx in batch_indices:
                    try:
                        original_graph = loader.dataset[int(graph_idx)]
                        if hasattr(original_graph, 'subject'):
                            all_subjects.append(original_graph.subject)
                        if hasattr(original_graph, 'recording'):
                            all_recordings.append(original_graph.recording)
                    except (AttributeError, IndexError):
                        pass
    
    # Concatenate predictions
    y_pred = torch.cat(all_outputs).numpy()
    y_true = torch.cat(all_targets).numpy()
    
    # Prepare metadata
    metadata = None
    if all_subjects and all_recordings:
        metadata = {
            'subjects': all_subjects,
            'recordings': all_recordings
        }
    
    # Compute metrics
    metrics = evaluate_binary_classification(
        y_pred, y_true,
        metadata=metadata,
        emotion_names=[emotion_name],
        threshold=threshold
    )
    
    return metrics, total_loss / len(loader)


def train_gnn_fold(config, train_idx, val_idx, test_idx, dataset, fold_dir, 
                   test_name, device):
    """Train GNN for one fold."""
    model_cfg = config['gnn']['model']
    training_cfg = config['gnn']['training']
    emotion_name = config['binary_task']['target_emotion']
    threshold = config['binary_task'].get('threshold', 0.0)
    
    # Create model
    model = BinarySpatioTemporalGNN(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])
    
    # Create data loaders
    train_loader = DataLoader(
        [dataset[i] for i in train_idx],
        batch_size=training_cfg['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        [dataset[i] for i in val_idx],
        batch_size=training_cfg['batch_size']
    )
    test_loader = DataLoader(
        [dataset[i] for i in test_idx],
        batch_size=training_cfg['batch_size']
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"Training GNN for {test_name}...")
    for epoch in range(training_cfg['num_epochs']):
        train_loss = train_gnn_epoch(
            model, train_loader, optimizer, device,
            training_cfg.get('grad_clip_max_norm', 1.0)
        )
        
        val_metrics, val_loss = evaluate_gnn(
            model, val_loader, device, emotion_name, threshold
        )
        
        if (epoch + 1) % 10 == 0 or epoch == training_cfg['num_epochs'] - 1 or epoch == 0:
            val_acc = val_metrics['standard']['aggregated']['accuracy']
            print(f"  Epoch {epoch+1}/{training_cfg['num_epochs']}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
    
    print(f"  Best model at epoch {best_epoch+1}")
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
    test_metrics, test_loss = evaluate_gnn(
        model, test_loader, device, emotion_name, threshold
    )
    
    test_acc = test_metrics['standard']['aggregated']['accuracy']
    print(f"  ❗GNN - Test Accuracy: {test_acc:.4f}")
    
    return test_metrics


def train_baselines_fold(baseline_cfg, train_idx, val_idx, test_idx, 
                         samples, fold_dir, metric_names, emotion_name):
    """Train baseline models for one fold."""
    baseline_dir = os.path.join(fold_dir, 'baselines')
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Convert to X, y
    X_train, y_train, train_meta, feat_cols, targ_cols = samples_to_xy(samples, train_idx)
    X_val, y_val, val_meta, _, _ = samples_to_xy(samples, val_idx)
    X_test, y_test, test_meta, _, _ = samples_to_xy(samples, test_idx)
    
    # Prepare metadata dicts
    train_metadata = {
        'subjects': [m[0] for m in train_meta if m],
        'recordings': [m[1] for m in train_meta if m]
    }
    test_metadata = {
        'subjects': [m[0] for m in test_meta if m],
        'recordings': [m[1] for m in test_meta if m]
    }
    
    results = {}
    
    for model_name in baseline_cfg['models']:
        print(f"  Training {model_name}...")
        
        # Get hyperparameters
        hyperparams = baseline_cfg.get('hyperparameters', {}).get(model_name, {})
        
        # Create and train model
        model = get_binary_baseline_by_name(model_name, **hyperparams)
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_metrics = model.evaluate(
            X_test, y_test,
            emotion_names=[emotion_name],
            metadata=test_metadata,
            threshold=0.5
        )
        
        results[model_name] = test_metrics
        
        # Log accuracy
        test_acc = test_metrics['standard']['aggregated']['accuracy']
        print(f"    ❗{model_name} - Test Accuracy: {test_acc:.4f}")
    
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Extract config sections
    run_experiments = config['run_experiments']
    dataset_cfg = config['dataset']
    binary_task_cfg = config['binary_task']
    cv_cfg = config['cross_validation']
    logging_cfg = config['logging']
    metric_names = config['metrics']
    
    # Get binary task parameters
    target_emotion = binary_task_cfg['target_emotion']
    threshold = binary_task_cfg.get('threshold', 0.0)
    
    print(f"Binary Classification Task:")
    print(f"  Target emotion: {target_emotion}")
    print(f"  Threshold: {threshold}")
    print(f"  Labels: <=threshold -> 0, >threshold -> 1")
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(logging_cfg['results_dir'], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(run_dir, 'training_log.txt')
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"\nTraining started at: {datetime.now()}")
    print(f"Results will be saved to: {run_dir}")
    print(f"Run baselines: {run_experiments['baselines']}")
    print(f"Run GNN: {run_experiments['gnn']}")
    
    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set device for GNN
    if run_experiments['gnn']:
        training_cfg = config['gnn']['training']
        if training_cfg['device'] == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(training_cfg['device'])
        print(f"Using device: {device}")
        
        # Set random seed
        if training_cfg.get('random_seed') is not None:
            torch.manual_seed(training_cfg['random_seed'])
            np.random.seed(training_cfg['random_seed'])
    
    # Load datasets
    print("\nLoading datasets...")
    
    if run_experiments['gnn']:
        print("Loading graph dataset for GNN...")
        base_gnn_dataset = SpacioTemporalDataset(
            root_dir=dataset_cfg.get('data_dir'),
            data_filepath=dataset_cfg.get('data_filepath'),
            filter_subjects=dataset_cfg.get('filter_subjects'),
            filter_recordings=dataset_cfg.get('filter_recordings'),
            file_list=dataset_cfg.get('file_list'),
            recursive=dataset_cfg['recursive'],
            ignore_dirs=dataset_cfg.get('ignore_dirs', []),
            window_length=dataset_cfg['window_length'],
            window_overlap=dataset_cfg['window_overlap'],
            kt=dataset_cfg['kt'],
            ks=dataset_cfg['ks'],
            cache_dir=dataset_cfg.get('cache_dir'),
            use_cache=dataset_cfg.get('use_cache', True),
            dropping_emotion_threshold=dataset_cfg.get('dropping_emotion_threshold', -1),
        )
        
        # Wrap with binary dataset
        gnn_dataset = BinarySpacioTemporalDataset(
            base_gnn_dataset,
            target_emotion=target_emotion,
            threshold=threshold
        )
        print(f"Loaded {len(gnn_dataset)} graph samples")
    
    if run_experiments['baselines']:
        print("Loading tabular samples for baselines...")
        base_tabular_samples = build_tabular_samples(
            data_dir=dataset_cfg.get('data_dir'),
            data_filepath=dataset_cfg.get('data_filepath'),
            filter_subjects=dataset_cfg.get('filter_subjects'),
            filter_recordings=dataset_cfg.get('filter_recordings'),
            file_list=dataset_cfg.get('file_list'),
            window_length=dataset_cfg.get('window_length', 10),
            dropping_emotion_threshold=dataset_cfg.get('dropping_emotion_threshold', -1)
        )
        
        # Wrap with binary samples
        tabular_samples = wrap_tabular_samples(
            base_tabular_samples,
            target_emotion=target_emotion,
            threshold=threshold
        )
        print(f"Loaded {len(tabular_samples)} tabular samples")
    
    # Get CV strategies
    strategies = cv_cfg['strategies']
    if isinstance(strategies, str):
        strategies = [strategies]
    
    print(f"\nWill run experiments with {len(strategies)} strategy(ies): {', '.join(strategies)}")
    
    # Storage for all results
    all_strategies_results = {}
    
    # Run experiments for each strategy
    for strategy in strategies:
        print("\n" + "="*100)
        print(f"Starting cross-validation with strategy: {strategy.upper()}")
        print("="*100)
        
        strategy_dir = os.path.join(run_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Create splitters
        if run_experiments['baselines']:
            baseline_splitter = create_splitter(
                strategy=strategy,
                samples=tabular_samples,
                val_size=cv_cfg['val_size'],
                random_state=cv_cfg.get('random_state')
            )
        
        if run_experiments['gnn']:
            gnn_splitter = create_splitter(
                strategy=strategy,
                samples=gnn_dataset,
                val_size=cv_cfg['val_size'],
                random_state=cv_cfg.get('random_state')
            )
        
        # Reference dataset for fold identification
        reference_splitter = gnn_splitter if run_experiments['gnn'] else baseline_splitter
        reference_dataset = gnn_dataset if run_experiments['gnn'] else tabular_samples
        
        # Storage for this strategy
        baseline_results_all_folds = {name: {} for name in config['baselines']['models']} if run_experiments['baselines'] else {}
        gnn_results_all_folds = {}
        
        # Get splits
        if run_experiments['baselines'] and run_experiments['gnn']:
            baseline_splits = list(baseline_splitter.split())
            gnn_splits = list(gnn_splitter.split())
            num_folds = len(baseline_splits)
        elif run_experiments['baselines']:
            baseline_splits = list(baseline_splitter.split())
            num_folds = len(baseline_splits)
        else:
            gnn_splits = list(gnn_splitter.split())
            num_folds = len(gnn_splits)
        
        for fold_num in range(num_folds):
            # Get indices
            if run_experiments['baselines']:
                baseline_train_idx, baseline_val_idx, baseline_test_idx = baseline_splits[fold_num]
            if run_experiments['gnn']:
                gnn_train_idx, gnn_val_idx, gnn_test_idx = gnn_splits[fold_num]
            
            # Identify test fold
            if run_experiments['gnn']:
                ref_test_idx = gnn_test_idx
            else:
                ref_test_idx = baseline_test_idx
            
            if strategy == 'subject_loo':
                test_subjects = sorted(set(reference_dataset[i].subject for i in ref_test_idx))
                test_id = f"s_{'_'.join(map(str, test_subjects))}"
                test_name = f"Subjects {', '.join(map(str, test_subjects))}"
            elif strategy == 'recording_loo':
                test_recordings = sorted(set(reference_dataset[i].recording for i in ref_test_idx))
                test_id = f"r_{'_'.join(map(str, test_recordings))}"
                test_name = f"Recordings {', '.join(map(str, test_recordings))}"
            elif strategy == 'combined_loo':
                test_pairs = sorted(set((reference_dataset[i].subject, reference_dataset[i].recording) for i in ref_test_idx))
                test_id = f"sr_{'_'.join([f'{s}_{r}' for s, r in test_pairs])}"
                test_name = f"Pairs {', '.join([f'({s}, {r})' for s, r in test_pairs])}"
            else:
                test_id = f"fold_{fold_num}"
                test_name = f"Fold {fold_num}"
            
            fold_dir = os.path.join(strategy_dir, test_id)
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"\n{test_name}")
            
            # Train baselines
            if run_experiments['baselines']:
                print("Training baselines...")
                baseline_results = train_baselines_fold(
                    config['baselines'], baseline_train_idx, baseline_val_idx,
                    baseline_test_idx, tabular_samples, fold_dir,
                    metric_names, target_emotion
                )
                for model_name, metrics in baseline_results.items():
                    baseline_results_all_folds[model_name][test_id] = metrics
            
            # Train GNN
            if run_experiments['gnn']:
                gnn_metrics = train_gnn_fold(
                    config, gnn_train_idx, gnn_val_idx, gnn_test_idx,
                    gnn_dataset, fold_dir, test_name, device
                )
                gnn_results_all_folds[test_id] = gnn_metrics
        
        # Combine results
        combined_results = {}
        if run_experiments['baselines']:
            combined_results.update(baseline_results_all_folds)
        if run_experiments['gnn']:
            combined_results['GNN'] = gnn_results_all_folds
        
        all_strategies_results[strategy] = combined_results
        
        # Print and save comparison
        print_comparison_table(combined_results, metric_names, strategy)
        csv_path = os.path.join(strategy_dir, 'summary.csv')
        save_comparison_csv(combined_results, metric_names, csv_path)
    
    print(f"\n{'='*100}")
    print("Training complete!")
    print(f"All results saved to: {run_dir}")
    
    # Restore stdout
    sys.stdout = logger.terminal
    sys.stderr = sys.__stderr__
    logger.close()


if __name__ == "__main__":
    main()
