# train.py

"""
Example usage:
python src/emotions/train.py
python src/emotions/train.py --config src/emotions/train_config.yaml
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime
import sys
import os
import numpy as np
import yaml
import argparse


# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data import SpacioTemporalDataset
from emotions.model import SpatioTemporalHeteroGNN
from emotions.splits import SubjectLOOSplitter, RecordingLOOSplitter, CombinedLOOSplitter


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


def compute_metrics(outputs, targets):
    """Compute comprehensive evaluation metrics.
    
    Args:
        outputs: torch.Tensor of predictions
        targets: torch.Tensor of ground truth
        
    Returns:
        dict: Dictionary containing MSE, MAE, SD, R², and Pearson R
    """
    # Convert to numpy and flatten
    y_pred = outputs.cpu().numpy().flatten()
    y_true = targets.cpu().numpy().flatten()
    
    # MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Standard deviation of error
    errors = y_true - y_pred
    sd_error = np.std(errors)
    
    # R² (coefficient of determination)
    r2 = r2_score(y_true, y_pred)
    

    # Pearson correlation coefficient
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'sd_error': sd_error,
        'r2': r2,
        'pearson_r': pearson_r
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


def evaluate(model, loader, device, save_outputs=False, save_dir=None):
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
    metrics = compute_metrics(outputs, targets)
    metrics['loss'] = total_loss / len(loader)
    
    if save_outputs and save_dir:
        torch.save({'outputs': outputs, 'targets': targets, 'metrics': metrics}, save_dir)
    
    return metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train SpatioTemporalHeteroGNN for emotion prediction")
    parser.add_argument("--config", type=str, default=None, 
                       help="Path to config YAML file (default: src/emotions/train_config.yaml)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
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

    # Save config to run directory for reproducibility
    config_save_path = os.path.join(run_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved configuration to: {config_save_path}")

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
                dropout_mlp=model_cfg.get('dropout_mlp', 0.1)
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
            print(f"Will save outputs at epochs: {sorted(save_epochs)}")
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
                val_metrics = evaluate(model, val_loader, device, save_outputs=save_outputs, save_dir=save_path)
                val_loss = val_metrics['loss']
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if logging_cfg.get('save_best_model', True):
                        torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pt'))
                
                print_every = logging_cfg.get('print_every', 10)
                if epoch % print_every == 0 or epoch == 1:
                    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val MSE: {val_metrics['mse']:.4f} | "
                        f"MAE: {val_metrics['mae']:.4f} | R²: {val_metrics['r2']:.4f} | Pearson R: {val_metrics['pearson_r']:.4f}"
                        f" | Time: {datetime.now() - epoch_start_time}")
            
            test_metrics[test_id] = evaluate(model, test_loader, device, save_outputs=False, save_dir=None) 
            print(f"Test Metrics for {test_name}: "
                  f"MSE: {test_metrics[test_id]['mse']:.4f} | "
                  f"MAE: {test_metrics[test_id]['mae']:.4f} | "
                  f"R²: {test_metrics[test_id]['r2']:.4f} | "
                  f"Pearson R: {test_metrics[test_id]['pearson_r']:.4f}")
            print(f"Time taken for {test_name}: {datetime.now() - fold_start_time}\n")
        
        # Store results for this strategy
        all_strategies_results[strategy] = test_metrics
        
        # Print summary for this strategy
        print("\n" + "="*100)
        fold_type = "Subjects" if strategy == 'subject_loo' else "Recordings" if strategy == 'recording_loo' else "Subject-Recording Pairs"
        print(f"Summary for {strategy.upper()} (Averaged Across {fold_type})")
        print("="*100)
        
        metric_names = config['metrics']
        final_metrics = {metric: np.nanmean([test_metrics[fold_id][metric] for fold_id in test_metrics]) for metric in metric_names}
        
        metric_str = " | ".join([f"{metric.upper()}: {final_metrics[metric]:.4f}" for metric in metric_names])
        print(metric_str)
    
    # Final comparison across all strategies
    if len(strategies) > 1:
        print("\n" + "="*100)
        print("FINAL COMPARISON ACROSS ALL STRATEGIES")
        print("="*100)
        
        metric_names = config['metrics']
        print(f"{'Strategy':<20} | " + " | ".join([f"{m.upper():<10}" for m in metric_names]))
        print("-"*100)
        
        for strategy, test_metrics in all_strategies_results.items():
            final_metrics = {metric: np.nanmean([test_metrics[fold_id][metric] for fold_id in test_metrics]) for metric in metric_names}
            metric_str = " | ".join([f"{final_metrics[m]:<10.4f}" for m in metric_names])
            print(f"{strategy:<20} | {metric_str}")
    
    print(f"\nTraining complete!")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()


