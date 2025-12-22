# train.py

"""
Example usage:
python src/emotions/train.py
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

from emotions.splits import SubjectLOOSplitter

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data import SpacioTemporalDataset
from emotions.model import SpatioTemporalHeteroGNN


def compute_metrics(outputs, targets):
    """Compute comprehensive evaluation metrics.
    
    Args:
        outputs: torch.Tensor of predictions
        targets: torch.Tensor of ground truth
        
    Returns:
        dict: Dictionary containing MSE, MAE, SD, R², D², and Pearson R
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
    
    # D² (fraction of deviance explained)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    d2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Pearson correlation coefficient
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'sd_error': sd_error,
        'r2': r2,
        'd2': d2,
        'pearson_r': pearson_r
    }


def train_epoch(model, loader, optimizer, device):
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    data_dir = "./data/processed/eSEEd_v2/"
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"./results/SpatioTemporalHeteroGNN/eSEEd_v2/{timestamp}/"
    data_save_dir = os.path.join(run_dir, "data")
    os.makedirs(data_save_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Results will be saved to: {run_dir}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = SpacioTemporalDataset(
        root_dir=data_dir,
        # file_list=["sample_01_recording_01_merged.csv",
        #            "sample_01_recording_02_merged.csv",
        #            "sample_01_recording_03_merged.csv",
        #            "sample_01_recording_04_merged.csv"],
        recursive=True,
        window_length=10,
        kt=2,
        ks=2,
    )

    splitter = SubjectLOOSplitter(dataset, val_size=1, random_state=42)
    exit(0)
    
    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    print(f"Train: {len(train_dataset)} graphs | Test: {len(test_dataset)} graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = SpatioTemporalHeteroGNN(
        in_channels=5,
        hidden_channels=64,
        out_channels=4
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    num_epochs = 100
    best_test_loss = float('inf')
    
    # Determine epochs to save outputs (10% of total, equidistant)
    save_interval = max(1, num_epochs // 10)
    save_epochs = set(range(save_interval, num_epochs + 1, save_interval))
    
    print("\nStarting training...")
    print(f"Will save outputs at epochs: {sorted(save_epochs)}")
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Save outputs for selected epochs
        save_outputs = epoch in save_epochs
        save_path = os.path.join(data_save_dir, f'epoch_{epoch:03d}.pt') if save_outputs else None
        test_metrics = evaluate(model, test_loader, device, save_outputs=save_outputs, save_dir=save_path)
        test_loss = test_metrics['loss']
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test MSE: {test_metrics['mse']:.4f} | "
                  f"MAE: {test_metrics['mae']:.4f} | R²: {test_metrics['r2']:.4f} | Pearson R: {test_metrics['pearson_r']:.4f}")
    
    # Final evaluation with all metrics
    print("\n" + "="*100)
    print("Final Test Metrics")
    print("="*100)
    final_metrics = evaluate(model, test_loader, device, save_outputs=False, save_dir=None)
    print(f"MSE: {final_metrics['mse']:.4f} | MAE: {final_metrics['mae']:.4f} | SD_Err: {final_metrics['sd_error']:.4f}")
    print(f"R²: {final_metrics['r2']:.4f} | D²: {final_metrics['d2']:.4f} | Pearson R: {final_metrics['pearson_r']:.4f}")
    
    print(f"\nTraining complete! Best test loss: {best_test_loss:.4f}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()

