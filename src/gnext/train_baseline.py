# train_baseline.py
"""
Training script for baseline models on eye tracking data.
"""
import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data import EyePathDataset
from baseline_models import MLPBaseline, CNNBaseline
from train_utils import set_seed, split_by_sequence, evaluate


def main():
    """Main training loop for baseline models."""
    
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config_baseline.yaml",
                    help="path to config file")
    args = ap.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config values
    data_dir = config['data_dir']
    seed = config['seed']
    model_cfg = config['model']
    train_cfg = config['train']
    data_cfg = config['data']

    # Set checkpoint path
    if train_cfg['save']:
        save_dir = os.path.join(train_cfg['save_dir'], config["experiment_name"])
        save_path = os.path.join(save_dir, f"best{train_cfg['epochs']}.pt")
        os.makedirs(save_dir, exist_ok=True)
    
    set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    IGNORE_DIRS = ["cog-load"]  # for prototyping use cog-load-mini
    
    # LOAD EYE TRACKING SEQUENCES AS GRAPHS
    all_ds = EyePathDataset(data_dir, recursive=True, lookback=data_cfg['lookback'], ignore_dirs=IGNORE_DIRS)
    
    if train_cfg['test_set']:
        print(f"Holding out dataset '{train_cfg['test_set']}' for testing only.")
        train_val_indices = [i for i in range(len(all_ds)) if train_cfg['test_set'] not in all_ds[i].dataset_name]
        test_indices = [i for i in range(len(all_ds)) if train_cfg['test_set'] in all_ds[i].dataset_name]
        train_val_ds = Subset(all_ds, train_val_indices)
        test_ds = Subset(all_ds, test_indices)  # not used for now
    else:
        train_val_ds = all_ds
    
    # DATA SPLITS AND LOADERS
    train_ds, val_ds = split_by_sequence(train_val_ds, val_split=train_cfg['val_split'], seed=seed)
    print("Train graphs:", len(train_ds), "| Val graphs:", len(val_ds))
    avg_train_length = np.mean([train_ds[i].num_nodes for i in range(len(train_ds))])
    avg_val_length = np.mean([val_ds[i].num_nodes for i in range(len(val_ds))])
    print(f"Average train graph: ({avg_train_length:.2f}, {train_ds[0].x.shape[1]}) | Average val graph: ({avg_val_length:.2f}, {val_ds[0].x.shape[1]})")
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=os.cpu_count()//2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], num_workers=os.cpu_count()//2, persistent_workers=True)
    print()
    
    # INITIALIZE MODEL & OPTIMIZER
    input_dim = 4  # (x, y, pupil-left, pupil-right)
    output_dim = 2  # (x_next, y_next)
    model_name = model_cfg['name']
    
    if model_name == "MLPBaseline":
        model = MLPBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[model_cfg['hidden_dim']] * model_cfg['layers']
        ).to(device)
    elif model_name == "CNNBaseline":
        model = CNNBaseline(
            input_channels=input_dim,
            output_dim=output_dim,
            hidden_dims=[model_cfg['hidden_dim']] * model_cfg['layers']
        ).to(device)
    else:
        raise ValueError(f"Unknown baseline model name: {model_name}")
    
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'])
    
    # TRAINING LOOP
    best_val = float("inf")
    start_time = time.time()
    print("Starting training...")
    for epoch in range(1, train_cfg['epochs'] + 1):
        epoch_start = time.time()
        # training phase
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.x)
            
            # loss only on nodes that have a next-step target (mask excludes last node)
            loss = F.mse_loss(pred[batch.mask], batch.y[batch.mask])
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        # validation phase
        val_mae, val_rmse = evaluate(model, val_loader, device, model_cfg['use_edge_attr'])
        if epoch % (train_cfg['epochs'] // 10) == 0 or epoch == 1 or epoch == train_cfg['epochs']:
            print(f"Epoch {epoch:03d} | train_mse={total_loss/len(train_loader):.6f} "
                  f"| val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f} | epoch_time={time.time() - epoch_start:.2f}s")
        
        # save best model based on validation MAE
        if train_cfg['save'] and val_mae < best_val:
            best_val = val_mae
            torch.save({
                "state_dict": model.state_dict(),
                "in_channels": 4,
                "hidden": model_cfg['hidden_dim'],
                "layers": model_cfg['layers'],
                "model_name": model_name,
                "layer_name": None,
                "lookback": data_cfg['lookback']
            }, save_path)
    
    # LOAD BEST MODEL AND REPORT FINAL VAL METRICS
    if train_cfg['save']:
        print("\nLoading best model for final evaluation...")
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            val_mae, val_rmse = evaluate(model, val_loader, device, model_cfg['use_edge_attr'])
            print(f"Best checkpoint | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes.")
    if train_cfg['save']:
        print(f"Best model saved at: {save_path}")


if __name__ == "__main__":
    main()
