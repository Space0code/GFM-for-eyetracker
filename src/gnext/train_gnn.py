# train_gnn.py
"""
Training script for GNN-based next-point prediction on eye tracking data.
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
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GINConv, TransformerConv
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data import EyePathDataset
from model import NextPointGNN
from train_utils import (
    set_seed, prepare_data, save_checkpoint, setup_experiment,
    save_final_results, save_epoch_results, run_training_loop, finalize_training
)


def main():
    """Main training loop for GNN-based next-point prediction."""
    
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml",
                    help="path to config file")
    args = ap.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Layer modules mapping
    layer_modules = {
        "SAGEConv": SAGEConv,
        "GCNConv": GCNConv,
        "GATConv": GATConv,
        "GINConv": GINConv,
        "TransformerConv": TransformerConv
    }
    
    # Extract config values
    data_dir = config['data_dir']
    seed = config['seed']
    model_cfg = config['model']
    train_cfg = config['train']
    data_cfg = config['data']
    
    # Setup experiment directory and logging
    exp_dir = setup_experiment(config['experiment_name'])
    save_path = os.path.join(exp_dir, "best.pt")
    
    set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    IGNORE_DIRS = ["cog-load"]  # for prototyping use cog-load-mini
    
    # LOAD EYE TRACKING SEQUENCES AS GRAPHS
    all_ds = EyePathDataset(data_dir, recursive=True, lookback=data_cfg['lookback'], ignore_dirs=IGNORE_DIRS)
    train_loader, val_loader = prepare_data(all_ds, train_cfg, seed, train_cfg['batch_size'])
    
    # INITIALIZE MODEL & OPTIMIZER
    input_dim = 4  # (x, y, pupil-left, pupil-right)
    output_dim = 2  # (x_next, y_next)
    layer_module = layer_modules[model_cfg['layer']]
    model = NextPointGNN(
        in_channels=input_dim,
        hidden_dim=model_cfg['hidden_dim'],
        output_dim=output_dim,
        num_layers=model_cfg['layers'],
        layer=layer_module
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'])
    
    # TRAINING LOOP
    def save_checkpoint_fn(path):
        save_checkpoint(model, model_cfg, data_cfg, path, layer_module)
    
    all_results = run_training_loop(
        model, train_loader, val_loader, optim, device,
        model_cfg, train_cfg, save_path, save_checkpoint_fn
    )
    
    # FINALIZE AND SAVE RESULTS
    final_metrics = finalize_training(
        model, val_loader, device, model_cfg['use_edge_attr'],
        save_path, exp_dir, train_cfg
    )
    
    if final_metrics:
        save_final_results(final_metrics, exp_dir)
        save_epoch_results(all_results, exp_dir)


if __name__ == "__main__":
    main()
