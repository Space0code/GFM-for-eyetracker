"""
Unified training script for GNN and baseline models with cross-validation.

Usage:
  python src/emotions/train_combined.py --config src/emotions/configs/train_config_combined.yaml
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data import SpacioTemporalDataset
from emotions.train_gnn import train_gnn_fold
from emotions.utils import (
    Logger,
    load_config,
    validate_config,
    create_splitter,
    build_tabular_samples,
    train_baselines_fold,
    save_comparison_csv,
    print_comparison_table
)








def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN and baseline models with cross-validation")
    parser.add_argument("--config", type=str, required=True, help="Path to unified config YAML file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Extract config sections
    run_experiments = config['run_experiments']
    dataset_cfg = config['dataset']
    cv_cfg = config['cross_validation']
    logging_cfg = config['logging']
    metric_names = config['metrics']
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(logging_cfg['results_dir'], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(run_dir, 'training_log.txt')
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"Training started at: {datetime.now()}")
    print(f"Results will be saved to: {run_dir}")
    print(f"Run baselines: {run_experiments['baselines']}")
    print(f"Run GNN: {run_experiments['gnn']}")
    
    # Save config copy
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved configuration to: {config_save_path}")
    
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
        gnn_dataset = SpacioTemporalDataset(
            root_dir=dataset_cfg['data_dir'],
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
        print(f"Loaded {len(gnn_dataset)} graph samples")
    
    if run_experiments['baselines']:
        print("Loading tabular samples for baselines...")
        tabular_samples = build_tabular_samples(
            data_dir=dataset_cfg['data_dir'],
            file_list=dataset_cfg.get('file_list'),
            window_length=dataset_cfg.get('window_length', 10),
            dropping_emotion_threshold=dataset_cfg.get('dropping_emotion_threshold', -1)
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
        
        # Create splitter (use GNN dataset if available, else tabular)
        dataset_for_split = gnn_dataset if run_experiments['gnn'] else tabular_samples
        splitter = create_splitter(
            strategy=strategy,
            samples=dataset_for_split,
            val_size=cv_cfg['val_size'],
            random_state=cv_cfg.get('random_state')
        )
        
        # Storage for this strategy
        baseline_results_all_folds = {name: {} for name in config['baselines']['models']} if run_experiments['baselines'] else {}
        gnn_results_all_folds = {}
        
        fold_num = 0
        for train_idx, val_idx, test_idx in splitter.split():
            fold_num += 1
            
            # Identify test fold
            if strategy == 'subject_loo':
                test_subjects = sorted(set(dataset_for_split[i].subject for i in test_idx))
                test_id = f"s_{'_'.join(test_subjects)}"
                test_name = f"Subjects {', '.join(test_subjects)}"
            elif strategy == 'recording_loo':
                test_recordings = sorted(set(dataset_for_split[i].recording for i in test_idx))
                test_id = f"r_{'_'.join(test_recordings)}"
                test_name = f"Recordings {', '.join(test_recordings)}"
            elif strategy == 'combined_loo':
                test_pairs = sorted(set((dataset_for_split[i].subject, dataset_for_split[i].recording) for i in test_idx))
                test_id = f"sr_{'_'.join([f'{s}_{r}' for s, r in test_pairs])}"
                test_name = f"Pairs {', '.join([f'({s}, {r})' for s, r in test_pairs])}"
            else:
                test_id = f"fold_{fold_num}"
                test_name = f"Fold {fold_num}"
            
            fold_dir = os.path.join(strategy_dir, test_id)
            os.makedirs(fold_dir, exist_ok=True)
            
            print(f"\n{test_name}: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
            
            # Train baselines
            if run_experiments['baselines']:
                print("Training baselines...")
                baseline_fold_dir = os.path.join(fold_dir, 'baselines')
                os.makedirs(baseline_fold_dir, exist_ok=True)
                baseline_results = train_baselines_fold(
                    config['baselines'], train_idx, val_idx, test_idx,
                    tabular_samples, baseline_fold_dir, metric_names
                )
                for model_name, metrics in baseline_results.items():
                    baseline_results_all_folds[model_name][test_id] = metrics
            
            # Train GNN
            if run_experiments['gnn']:
                gnn_metrics = train_gnn_fold(
                    config, train_idx, val_idx, test_idx,
                    gnn_dataset, fold_dir, test_name, device
                )
                gnn_results_all_folds[test_id] = gnn_metrics
        
        # Combine results for this strategy
        combined_results = {}
        if run_experiments['baselines']:
            combined_results.update(baseline_results_all_folds)
        if run_experiments['gnn']:
            combined_results['GNN'] = gnn_results_all_folds
        
        all_strategies_results[strategy] = combined_results
        
        # Print and save comparison for this strategy
        print_comparison_table(combined_results, metric_names, strategy)
        save_comparison_csv(combined_results, metric_names, strategy, strategy_dir)
    
    # Final comparison across strategies (if multiple)
    if len(strategies) > 1:
        print("\n" + "="*100)
        print("FINAL COMPARISON ACROSS ALL STRATEGIES")
        print("="*100)
        
        for strategy in strategies:
            print(f"\n{strategy.upper()}:")
            results = all_strategies_results[strategy]
            print(f"{'Model':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
            print("-"*100)
            for model_name, model_results in results.items():
                avg_metrics = {m: np.nanmean([model_results[fold]['standard']['aggregated'][m] 
                                              for fold in model_results]) for m in metric_names}
                metric_str = " | ".join([f"{avg_metrics[m]:<10.4f}" for m in metric_names])
                print(f"{model_name:<20} | {metric_str}")
        
        # Save final CSV
        final_rows = []
        for strategy, results in all_strategies_results.items():
            for model_name, model_results in results.items():
                row = {'strategy': strategy, 'model': model_name}
                for m in metric_names:
                    row[f'std_agg_{m}'] = np.nanmean([model_results[fold]['standard']['aggregated'][m] 
                                                       for fold in model_results])
                final_rows.append(row)
        final_df = pd.DataFrame(final_rows)
        final_csv_path = os.path.join(run_dir, 'all_strategies_comparison.csv')
        final_df.to_csv(final_csv_path, index=False)
        print(f"\nSaved final comparison to: {final_csv_path}")
    
    print(f"\n{'='*100}")
    print("Training complete!")
    print(f"All results saved to: {run_dir}")
    print(f"Total time: {datetime.now() - datetime.strptime(timestamp, '%Y-%m-%d_%H-%M-%S')}")
    
    # Restore stdout
    sys.stdout = logger.terminal
    sys.stderr = sys.__stderr__
    logger.close()


if __name__ == "__main__":
    main()
