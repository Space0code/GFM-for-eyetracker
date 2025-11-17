import math
import random
import numpy as np
import torch
from torch.utils.data import Subset

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
def evaluate(model, loader, device, use_edge_attr=True):
    """Evaluate model on validation set, computing MAE and RMSE metrics."""
    model.eval()
    mae_vals, rmse_vals = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index) if use_edge_attr else model(batch.x)
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
