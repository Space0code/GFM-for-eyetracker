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
from torch.utils.data import Subset, DataLoader
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_tabular import TabularEyePathDataset
from baseline_models import MLPBaseline, CNNBaseline
from train_utils import (
    save_config, set_seed, prepare_data_tabular, save_checkpoint, setup_experiment,
    save_final_results, save_epoch_results, run_training_loop, finalize_training
)


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

    # Setup experiment directory and logging
    exp_dir = setup_experiment(config['experiment_name'])
    save_path = os.path.join(exp_dir, "best.pt")
    
    set_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    IGNORE_DIRS = ["cog-load"]  # for prototyping use cog-load-mini
    
    # LOAD EYE TRACKING SEQUENCES AS TABULAR DATA
    all_ds = TabularEyePathDataset(data_dir, recursive=True, lookback=data_cfg['lookback'], ignore_dirs=IGNORE_DIRS)
    train_loader, val_loader = prepare_data_tabular(all_ds, train_cfg, seed, train_cfg['batch_size'])
    
    # INITIALIZE MODEL & OPTIMIZER
    input_dim = 4 * data_cfg['lookback']  # (x, y, pupil-left, pupil-right) * lookback
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
    def save_checkpoint_fn(path):
        save_checkpoint(model, model_cfg, data_cfg, path)
    
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
        save_config(config, exp_dir)    


if __name__ == "__main__":
    main()
