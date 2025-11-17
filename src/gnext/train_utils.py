import math
import os
import random
import sys
import json
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

class Logger:
    """Logger that writes to both terminal and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_experiment(experiment_name):
    """Setup experiment directory structure and logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("results", experiment_name, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    
    log_file = os.path.join(exp_dir, "complete_log.txt")
    sys.stdout = Logger(log_file)
    
    return exp_dir

def set_seed(seed: int):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # if torch.backends.mps.is_available():
    #     torch.mps.manual_seed(seed)


def split_by_sequence(dataset, val_split=0.2, seed=42):
    """Split dataset by sequences (not time steps) into train/validation sets."""
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_split))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    # guard tiny datasets
    if not train_idx:
        train_idx, val_idx = perm[1:], perm[:1]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

@torch.no_grad()
def evaluate_with_metrics(model, loader, device, use_edge_attr=True):
    """Evaluate model with detailed metrics including Pearson r and R²."""
    model.eval()
    all_preds, all_targets = [], []
    mae_vals, rmse_vals, euclidean_errors = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index) if use_edge_attr else model(batch.x)
        mask = batch.mask
        if mask.sum() == 0:
            continue
        
        pred_masked = pred[mask]
        target_masked = batch.y[mask]
        
        all_preds.append(pred_masked.cpu().numpy())
        all_targets.append(target_masked.cpu().numpy())
        
        diff = pred_masked - target_masked
        mae_vals.append(diff.abs().mean().item())
        rmse_vals.append(torch.sqrt((diff ** 2).mean()).item())
        euclidean_errors.extend(torch.norm(diff, dim=1).cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    mae = float(np.mean(mae_vals)) if mae_vals else math.nan
    rmse = float(np.mean(rmse_vals)) if rmse_vals else math.nan
    
    pearson_x = np.corrcoef(all_targets[:, 0], all_preds[:, 0])[0, 1]
    pearson_y = np.corrcoef(all_targets[:, 1], all_preds[:, 1])[0, 1]
    pearson_avg = (pearson_x + pearson_y) / 2
    
    r2_x = pearson_x ** 2
    r2_y = pearson_y ** 2
    r2_avg = (r2_x + r2_y) / 2
    
    euclidean_mean = float(np.mean(euclidean_errors))
    euclidean_std = float(np.std(euclidean_errors))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'pearson_x': float(pearson_x),
        'pearson_y': float(pearson_y),
        'pearson_avg': float(pearson_avg),
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'r2_avg': float(r2_avg),
        'euclidean_mean': euclidean_mean,
        'euclidean_std': euclidean_std
    }

def prepare_data(all_ds, train_cfg, seed, batch_size):
    """Prepare train/val datasets and loaders."""
    if train_cfg['test_set']:
        print(f"Holding out dataset '{train_cfg['test_set']}' for testing only.")
        train_val_indices = [i for i in range(len(all_ds)) if train_cfg['test_set'] not in all_ds[i].dataset_name]
        test_indices = [i for i in range(len(all_ds)) if train_cfg['test_set'] in all_ds[i].dataset_name]
        train_val_ds = Subset(all_ds, train_val_indices)
        test_ds = Subset(all_ds, test_indices)
    else:
        train_val_ds = all_ds
    
    train_ds, val_ds = split_by_sequence(train_val_ds, val_split=train_cfg['val_split'], seed=seed)
    print("Train graphs:", len(train_ds), "| Val graphs:", len(val_ds))
    avg_train_length = np.mean([train_ds[i].num_nodes for i in range(len(train_ds))])
    avg_val_length = np.mean([val_ds[i].num_nodes for i in range(len(val_ds))])
    print(f"Average train graph: ({avg_train_length:.2f}, {train_ds[0].x.shape[1]}) | Average val graph: ({avg_val_length:.2f}, {val_ds[0].x.shape[1]})")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=os.cpu_count()//2, persistent_workers=True)
    print()
    
    return train_loader, val_loader

def save_checkpoint(model, model_cfg, data_cfg, save_path, layer_module=None):
    """Save model checkpoint."""
    torch.save({
        "state_dict": model.state_dict(),
        "in_channels": 4,
        "hidden": model_cfg['hidden_dim'],
        "layers": model_cfg['layers'],
        "model_name": model_cfg['name'],
        "layer_name": layer_module.__name__ if layer_module else None,
        "lookback": data_cfg['lookback']
    }, save_path)

def save_final_results(metrics, exp_dir):
    """Save final evaluation results to CSV."""
    csv_path = os.path.join(exp_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in metrics.items():
            writer.writerow([key, value])

def save_epoch_results(all_results, exp_dir):
    """Save all epoch results to JSON."""
    json_path = os.path.join(exp_dir, "all_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

def save_config(config, exp_dir):
    """Save configuration to YAML file in experiment directory."""
    import yaml
    yaml_path = os.path.join(exp_dir, "config.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

def train_epoch(model, train_loader, optimizer, device, use_edge_attr=True):
    """Train for one epoch."""
    import torch.nn.functional as F
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index) if use_edge_attr else model(batch.x)
        loss = F.mse_loss(pred[batch.mask], batch.y[batch.mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def run_training_loop(model, train_loader, val_loader, optimizer, device, model_cfg, train_cfg, save_path, save_checkpoint_fn):
    """Run the complete training loop."""
    import time
    best_val = float("inf")
    start_time = time.time()
    print("Starting training...")
    patience_counter = 0
    all_results = {'train': [], 'val': []}
    use_edge_attr = model_cfg['use_edge_attr']
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        epoch_start = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, use_edge_attr)
        
        # Evaluation
        val_metrics = evaluate_with_metrics(model, val_loader, device, use_edge_attr)
        train_metrics = evaluate_with_metrics(model, train_loader, device, use_edge_attr)
        
        # Store results
        all_results['train'].append({**train_metrics, 'epoch': epoch, 'loss': train_loss})
        all_results['val'].append({**val_metrics, 'epoch': epoch})
        
        if epoch % train_cfg['log_epochs'] == 0 or epoch == 1 or epoch == train_cfg['epochs']:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:10.2f} "
                  f"| val_mae={val_metrics['mae']:7.2f} | val_rmse={val_metrics['rmse']:7.2f} "
                  f"| epoch_time={time.time() - epoch_start:6.2f}s")

        # Save best model and update early stopping
        if val_metrics['mae'] < best_val:
            patience_counter = 0
            best_val = val_metrics['mae']
            if train_cfg['save']:
                save_checkpoint_fn(save_path)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.get('patience', 10):
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes.")
    
    return all_results

def finalize_training(model, val_loader, device, use_edge_attr, save_path, exp_dir, train_cfg):
    """Load best model and save final results."""
    import torch
    if train_cfg['save']:
        print("\nLoading best model for final evaluation...")
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            final_metrics = evaluate_with_metrics(model, val_loader, device, use_edge_attr)
            print(f"Best checkpoint | val_mae={final_metrics['mae']:.4f} | val_rmse={final_metrics['rmse']:.4f}")
            print(f"Pearson r (x/y/avg): {final_metrics['pearson_x']:.4f}/{final_metrics['pearson_y']:.4f}/{final_metrics['pearson_avg']:.4f}")
            print(f"R² (x/y/avg): {final_metrics['r2_x']:.4f}/{final_metrics['r2_y']:.4f}/{final_metrics['r2_avg']:.4f}")
            print(f"Euclidean: {final_metrics['euclidean_mean']:.4f} ± {final_metrics['euclidean_std']:.4f}")
            print(f"Results saved in: {exp_dir}")
            return final_metrics
        else:
            print("No checkpoint found, skipping final evaluation.")
    return None
