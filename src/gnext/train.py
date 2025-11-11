# train.py
"""
Training script for next-point prediction on eye tracking data.
Uses GraphSAGE to predict the next gaze coordinate given a temporal sequence.
"""
import argparse
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from data import EyePathDataset
from model import NextPointGNN
from torch_geometric.nn import (
    SAGEConv,      # Current (mean aggregation)
    GCNConv,       # Graph Convolutional Network
    GATConv,       # Graph Attention Network
    GINConv,       # Graph Isomorphism Network
    TransformerConv # Graph Transformer
)

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
def evaluate(model, loader, device):
    """Evaluate model on validation set, computing MAE and RMSE metrics."""
    model.eval()
    mae_vals, rmse_vals = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index)
        mask = batch.mask  # mask excludes last node (no target)
        # skip empty masks (shouldn't happen, but safe)
        if mask.sum() == 0:
            continue
        diff = pred[mask] - batch.y[mask]
        mae_vals.append(diff.abs().mean().item())
        rmse_vals.append(torch.sqrt((diff ** 2).mean()).item())
    # average over graphs/batches
    mae = float(np.mean(mae_vals)) if mae_vals else math.nan
    rmse = float(np.mean(rmse_vals)) if rmse_vals else math.nan
    return mae, rmse

def main():
    """Main training loop for next-point prediction on eye tracking data."""

    layer_modules =     {
        "SAGEConv": SAGEConv,      # Current (mean aggregation)
        "GCNConv": GCNConv,       # Graph Convolutional Network
        "GATConv": GATConv,       # Graph Attention Network
        "GINConv": GINConv,       # Graph Isomorphism Network
        "TransformerConv": TransformerConv # Graph Transformer
    }

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed",
                    help="folder with CSVs (one sequence per CSV)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="checkpoints")
    ap.add_argument("--lookback", type=int, default=2)
    ap.add_argument("--layer_module_name", type=str, default="SAGEConv", choices=layer_modules.keys())
    ap.add_argument("--test_set", type=str, default=None,
                    help="optional name of a dataset to hold out for testing only")
    args = ap.parse_args()

    save_path = os.path.join(args.save, f"best{args.epochs}.pt")

    set_seed(args.seed)
    # don't use mps for now
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu" 
    print(f"Using device: {device}")

    IGNORE_DIRS = ["cog-load"] # for prototyping use cog-load-mini

    # load eye tracking sequences as graphs
    all_ds = EyePathDataset(args.data_dir, recursive=True, lookback=args.lookback, ignore_dirs=IGNORE_DIRS)
    if args.test_set:
        print(f"Holding out dataset '{args.test_set}' for testing only.")
        train_val_indices = [i for i in range(len(all_ds)) if args.test_set not in all_ds[i].seq_name]
        test_indices = [i for i in range(len(all_ds)) if args.test_set in all_ds[i].seq_name]
        train_val_ds = Subset(all_ds, train_val_indices)
        test_ds = Subset(all_ds, test_indices) # not used for now
        train_ds, val_ds = split_by_sequence(train_val_ds, val_split=args.val_split, seed=args.seed)

    else:    
        train_val_ds = all_ds
    
    train_ds, val_ds = split_by_sequence(train_val_ds, val_split=args.val_split, seed=args.seed)
    print("Train graphs length:", len(train_ds), "| Val graphs length:", len(val_ds))
    print(f"Train graphs shapes: {[train_ds[i].x.shape for i in range(len(train_ds))]} | Val graphs shape: {val_ds[0].x.shape}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()//2, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=os.cpu_count()//2, persistent_workers=True)
    print()

    # initialize model and optimizer
    layer_module = layer_modules[args.layer_module_name]
    model = NextPointGNN(in_channels=4, hidden_dim=args.hidden, num_layers=args.layers, layer=layer_module).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    start_time = time.time()
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # training phase
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index)

            # loss only on nodes that have a next-step target (mask excludes last node)
            loss = F.mse_loss(pred[batch.mask], batch.y[batch.mask])

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        # validation phase
        val_mae, val_rmse = evaluate(model, val_loader, device)
        if epoch % (args.epochs // 10) == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | train_mse={total_loss/len(train_loader):.6f} "
                f"| val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f} | epoch_time={time.time() - epoch_start:.2f}s")

        # save best model based on validation MAE
        if val_mae < best_val:
            best_val = val_mae
            torch.save({"state_dict": model.state_dict(),
                        "in_channels": 4,
                        "hidden": args.hidden,
                        "layers": args.layers,
                        "layer_name": layer_module.__name__,
                        "lookback": args.lookback}, 
                        save_path)

    # load best checkpoint and report final performance
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        val_mae, val_rmse = evaluate(model, val_loader, device)
        print(f"Best checkpoint | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}")

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes.")
    print(f"Best model saved at: {save_path}")

if __name__ == "__main__":
    main()