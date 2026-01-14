# train.py

"""
Example usage:
python src/emotions/train.py
python src/emotions/train.py --config src/emotions/train_config.yaml
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime
import sys
import os
import numpy as np
import yaml
import argparse
import pandas as pd


# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data import SpacioTemporalDataset
from emotions.model import SpatioTemporalHeteroGNN
from emotions.splits import SubjectLOOSplitter, RecordingLOOSplitter, CombinedLOOSplitter


class Logger:
    """Logger that writes to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def save_strategy_results_csv(config_file, run_dir, all_strategies_results, metric_names):
    """Save per-config comparison results to CSV.
    
    Args:
        config_file: Path to config file
        run_dir: Directory to save results
        all_strategies_results: Dictionary of results per strategy
        metric_names: List of metric names to include
    """
    csv_data = []
    for strategy, test_metrics in all_strategies_results.items():
        row = {'strategy': strategy}
        
        # Aggregated metrics
        final_metrics_agg = {f'agg_{metric}': np.nanmean([test_metrics[fold_id]['aggregated'][metric] for fold_id in test_metrics]) for metric in metric_names}
        row.update(final_metrics_agg)
        
        # Per-emotion metrics
        if test_metrics:
            first_fold = next(iter(test_metrics.values()))
            if 'per_emotion' in first_fold and first_fold['per_emotion']:
                for emo_name in first_fold['per_emotion'].keys():
                    emo_metrics = {f'{emo_name}_{metric}': np.nanmean([test_metrics[fold_id]['per_emotion'][emo_name][metric] for fold_id in test_metrics]) for metric in metric_names}
                    row.update(emo_metrics)
        
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
    
    # Aggregated metrics
    final_metrics_agg = {metric: np.nanmean([test_metrics[fold_id]['aggregated'][metric] for fold_id in test_metrics]) for metric in metric_names}
    metric_str = " | ".join([f"{metric.upper()}: {final_metrics_agg[metric]:.4f}" for metric in metric_names])
    print(f"\nAggregated: {metric_str}")
    
    # Per-emotion metrics
    if test_metrics:
        first_fold = next(iter(test_metrics.values()))
        if 'per_emotion' in first_fold and first_fold['per_emotion']:
            print("\nPer-emotion:")
            for emo_name in first_fold['per_emotion'].keys():
                emo_metrics = {metric: np.nanmean([test_metrics[fold_id]['per_emotion'][emo_name][metric] for fold_id in test_metrics]) for metric in metric_names}
                emo_str = " | ".join([f"{metric.upper()}: {emo_metrics[metric]:.4f}" for metric in metric_names])
                print(f"  {emo_name}: {emo_str}")


def print_strategies_comparison(all_strategies_results, metric_names):
    """Print comparison table across all strategies.
    
    Args:
        all_strategies_results: Dictionary of results per strategy
        metric_names: List of metric names to display
    """
    print("\n" + "="*100)
    print("FINAL COMPARISON ACROSS ALL STRATEGIES")
    print("="*100)
    
    print("\nAggregated Metrics:")
    print(f"{'Strategy':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
    print("-"*100)
    
    for strategy, test_metrics in all_strategies_results.items():
        final_metrics = {metric: np.nanmean([test_metrics[fold_id]['aggregated'][metric] for fold_id in test_metrics]) for metric in metric_names}
        metric_str = " | ".join([f"{final_metrics[m]:<10.4f}" for m in metric_names])
        print(f"{strategy:<20} | {metric_str}")
    
    # Per-emotion comparison
    if all_strategies_results:
        first_strategy = next(iter(all_strategies_results.values()))
        first_fold = next(iter(first_strategy.values()))
        if 'per_emotion' in first_fold and first_fold['per_emotion']:
            emotion_names = list(first_fold['per_emotion'].keys())
            for emo_name in emotion_names:
                print(f"\n{emo_name}:")
                print(f"{'Strategy':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
                print("-"*100)
                for strategy, test_metrics in all_strategies_results.items():
                    emo_metrics = {metric: np.nanmean([test_metrics[fold_id]['per_emotion'][emo_name][metric] for fold_id in test_metrics]) for metric in metric_names}
                    emo_str = " | ".join([f"{emo_metrics[m]:<10.4f}" for m in metric_names])
                    print(f"{strategy:<20} | {emo_str}")


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


def compute_metrics(outputs, targets, emotion_names=None):
    """Compute comprehensive evaluation metrics (aggregated and per-emotion).
    
    Args:
        outputs: torch.Tensor of predictions [num_samples, num_emotions]
        targets: torch.Tensor of ground truth [num_samples, num_emotions]
        emotion_names: list of emotion column names (optional)
        
    Returns:
        dict: Dictionary containing aggregated and per-emotion metrics
    """
    # Convert to numpy
    y_pred = outputs.cpu().numpy()
    y_true = targets.cpu().numpy()
    
    # Aggregated metrics (flatten all emotions)
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # Pearson correlation for aggregated
    if np.std(y_true_flat) > 1e-8 and np.std(y_pred_flat) > 1e-8:
        pearson_r= float(pearsonr(y_true_flat, y_pred_flat)[0])
    else:
        pearson_r = 0.0
    
    aggregated = {
        'mse': float(mean_squared_error(y_true_flat, y_pred_flat)),
        'mae': float(mean_absolute_error(y_true_flat, y_pred_flat)),
        'sd_error': float(np.std(y_true_flat - y_pred_flat)),
        'r2': float(r2_score(y_true_flat, y_pred_flat)),
        'pearson_r': pearson_r
    }
    
    # Per-emotion metrics
    per_emotion = {}
    num_emotions = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
    
    if emotion_names is None:
        emotion_names = [f'emotion_{i}' for i in range(num_emotions)]
    
    for i, emo_name in enumerate(emotion_names[:num_emotions]):
        y_pred_emo = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred
        y_true_emo = y_true[:, i] if len(y_true.shape) > 1 else y_true
        
        per_emotion[emo_name] = {
            'mse': float(mean_squared_error(y_true_emo, y_pred_emo)),
            'mae': float(mean_absolute_error(y_true_emo, y_pred_emo)),
            'sd_error': float(np.std(y_true_emo - y_pred_emo)),
            'r2': float(r2_score(y_true_emo, y_pred_emo)),
            'pearson_r': float(pearsonr(y_true_emo, y_pred_emo)[0]) if np.std(y_true_emo) > 1e-8 and np.std(y_pred_emo) > 1e-8 else 0.0
        }
    
    return {
        'aggregated': aggregated,
        'per_emotion': per_emotion
    }


def train_epoch(model, loader, optimizer, device, grad_clip_max_norm=1.0):
    """Train for one epoch."""
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


def evaluate(model, loader, device, emotion_names=None, save_outputs=False, save_dir=None):
    """Evaluate the model and compute comprehensive metrics."""
    model.eval()
    total_loss = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            target = data.y.view(-1, 4)
            
            loss = F.mse_loss(out, target)
            total_loss += loss.item()
            
            all_outputs.append(out.cpu())
            all_targets.append(target.cpu())
    
    # Concatenate all outputs and targets
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute comprehensive metrics
    metrics = compute_metrics(outputs, targets, emotion_names=emotion_names)
    metrics['aggregated']['loss'] = total_loss / len(loader)
    
    if save_outputs and save_dir:
        torch.save({'outputs': outputs, 'targets': targets, 'metrics': metrics}, save_dir)
    
    return metrics

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
            use_cache=dataset_cfg.get('use_cache', True)
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
                    val_loss = val_metrics['aggregated']['loss']
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if logging_cfg.get('save_best_model', True):
                            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
                    
                    print_every = logging_cfg.get('print_every', 10)
                    if epoch % print_every == 0 or epoch == 1:
                        agg = val_metrics['aggregated']
                        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val MSE: {agg['mse']:.4f} | "
                            f"MAE: {agg['mae']:.4f} | R²: {agg['r2']:.4f} | Pearson R: {agg['pearson_r']:.4f}"
                            f" | Time: {datetime.now() - epoch_start_time}")
                
                test_metrics[test_id] = evaluate(model, test_loader, device, emotion_names=emotion_names, save_outputs=False, save_dir=None) 
                # print(f"Test Metrics for {test_name}: "
                #     f"MSE: {test_metrics[test_id]['mse']:.4f} | "
                #     f"MAE: {test_metrics[test_id]['mae']:.4f} | "
                #     f"R²: {test_metrics[test_id]['r2']:.4f} | "
                #     f"Pearson R: {test_metrics[test_id]['pearson_r']:.4f}")
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
        
        # Store results for this config file
        config_results = {}
        for strategy, test_metrics in all_strategies_results.items():
            final_metrics = {metric: np.nanmean([test_metrics[fold_id]['aggregated'][metric] for fold_id in test_metrics]) for metric in config['metrics']}
            config_results[strategy] = final_metrics
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