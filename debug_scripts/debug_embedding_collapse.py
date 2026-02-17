#!/usr/bin/env python3
"""
Debug script to detect embedding collapse in trained GNN models.

Loads config from YAML, instantiates dataset + DataLoader, loads checkpoint,
and computes intermediate embeddings at each GNN stage to identify collapse.

Key measurements:
- Variance per embedding dimension (indicator of diversity)
- Pairwise cosine similarity (indicator of distinctiveness)
- L2 distances (indicator of spread)
- Within-graph node variance (indicator of oversmoothing)

Outputs:
- Console table showing metrics per stage with collapse flags
- JSON report with per-batch and aggregate metrics (saved to debug_reports/)

Collapse indicators:
- Variance drop >10x vs previous stage
- Cosine similarity >0.98 with L2 distance <0.1 (embeddings nearly identical)
- Constant probabilities (std < 0.01)
"""

import sys
import os
import json
import yaml
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Add src directory to path
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data.data import SpacioTemporalDataset
from emotions.binary.data_binary import BinarySpacioTemporalDataset
from emotions.binary.model_binary import BinarySpatioTemporalGNN
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


# ========================== HARDCODED CONFIG ==========================
CFG_PATH = "src/emotions/binary/configs/train_binary.yaml"
CHKPT_PATH = "results/binary/2026-02-17_11-14-12/recording_loo/r_2/best_model.pt"
DEVICE = "cuda"
NUM_BATCHES = 10
MAX_PAIRS = 512
SEED = 0


# ========================== METRICS COMPUTATION ==========================
def compute_variance_metrics(X: torch.Tensor) -> Dict[str, float]:
    """
    Compute variance-based metrics for embedding tensor X [N, D].
    
    Args:
        X: Tensor of shape [N, D]
        
    Returns:
        Dictionary with var_mean, std_all, norm_mean
    """
    var_per_dim = torch.var(X, dim=0)  # [D]
    var_mean = torch.mean(var_per_dim).item()
    std_all = torch.std(X).item()
    
    # L2 norm per sample
    norms = torch.norm(X, p=2, dim=1)  # [N]
    norm_mean = torch.mean(norms).item()
    
    return {
        "var_mean": var_mean,
        "std_all": std_all,
        "norm_mean": norm_mean,
    }


def compute_similarity_metrics(X: torch.Tensor, max_pairs: int = 512) -> Dict[str, float]:
    """
    Compute pairwise cosine similarity and L2 distance metrics.
    
    Args:
        X: Tensor of shape [N, D]
        max_pairs: Maximum number of pairs to sample
        
    Returns:
        Dictionary with cos_mean, cos_p5, cos_p95, l2_mean, l2_p5, l2_p95
    """
    N = X.shape[0]
    
    # Sample random pairs (limit to max_pairs)
    num_pairs = min(max_pairs, N * (N - 1) // 2)
    
    if num_pairs == 0:
        return {
            "cos_mean": 0.0, "cos_p5": 0.0, "cos_p95": 0.0,
            "l2_mean": 0.0, "l2_p5": 0.0, "l2_p95": 0.0,
        }
    
    # Random pair sampling
    idx_pairs = np.random.choice(N, size=(num_pairs, 2), replace=True)
    idx_pairs = torch.from_numpy(idx_pairs).to(X.device)
    
    X1 = X[idx_pairs[:, 0]]  # [num_pairs, D]
    X2 = X[idx_pairs[:, 1]]  # [num_pairs, D]
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(X1, X2, dim=1)  # [num_pairs]
    cos_mean = torch.mean(cos_sim).item()
    cos_p5 = torch.quantile(cos_sim, 0.05).item()
    cos_p95 = torch.quantile(cos_sim, 0.95).item()
    
    # L2 distance
    l2_dist = torch.norm(X1 - X2, p=2, dim=1)  # [num_pairs]
    l2_mean = torch.mean(l2_dist).item()
    l2_p5 = torch.quantile(l2_dist, 0.05).item()
    l2_p95 = torch.quantile(l2_dist, 0.95).item()
    
    return {
        "cos_mean": cos_mean,
        "cos_p5": cos_p5,
        "cos_p95": cos_p95,
        "l2_mean": l2_mean,
        "l2_p5": l2_p5,
        "l2_p95": l2_p95,
    }


def compute_within_graph_variance(X: torch.Tensor, batch_vec: torch.Tensor) -> Dict[str, float]:
    """
    Compute within-graph node variance.
    
    For each graph in batch, compute mean(var(X_g, dim=0)) where X_g are nodes in that graph.
    Then report mean across all graphs.
    
    Args:
        X: Node embeddings [num_nodes, D]
        batch_vec: Graph assignment [num_nodes]
        
    Returns:
        Dictionary with within_var_mean, within_var_std
    """
    within_vars = []
    
    unique_graphs = torch.unique(batch_vec)
    for gid in unique_graphs:
        mask = batch_vec == gid
        X_g = X[mask]  # [num_nodes_in_graph, D]
        
        if X_g.shape[0] > 1:
            var_g = torch.mean(torch.var(X_g, dim=0)).item()
            within_vars.append(var_g)
    
    if within_vars:
        return {
            "within_var_mean": np.mean(within_vars),
            "within_var_std": np.std(within_vars),
        }
    else:
        return {
            "within_var_mean": 0.0,
            "within_var_std": 0.0,
        }


def compute_head_norms(model) -> Dict[str, float]:
    """
    Compute weight and bias norms for the head layers.
    
    Args:
        model: Binary GNN model
        
    Returns:
        Dictionary with weight and bias norms for each head layer
    """
    norms = {}
    
    for i, module in enumerate(model.head):
        if isinstance(module, torch.nn.Linear):
            w_norm = torch.norm(module.weight).item()
            b_norm = torch.norm(module.bias).item() if module.bias is not None else 0.0
            norms[f"layer_{i}_w_norm"] = w_norm
            norms[f"layer_{i}_b_norm"] = b_norm
    
    return norms


def check_collapse_flags(metrics: Dict[str, float], prev_var_mean: float) -> List[str]:
    """
    Check for embedding collapse based on metrics.
    
    Flags:
    - If var_mean drops >10x vs previous stage
    - If cos_mean > 0.98 and embeddings nearly identical
    - If constant probability (small logits std)
    - If logits std << logits range (prevalence prediction)
    
    Args:
        metrics: Dictionary with computed metrics
        prev_var_mean: Previous stage's var_mean (or None for first stage)
        
    Returns:
        List of flag strings
    """
    flags = []
    
    # Check variance drop
    if prev_var_mean is not None and prev_var_mean > 0:
        var_ratio = prev_var_mean / (metrics.get("var_mean", 1e-6) + 1e-12)
        if var_ratio > 10:
            flags.append("⚠ VARIANCE COLLAPSE (>10x drop)")
    
    # Check cosine similarity
    cos_mean = metrics.get("cos_mean", 0.0)
    l2_mean = metrics.get("l2_mean", float('inf'))
    if cos_mean > 0.98 and l2_mean < 0.1:
        flags.append("⚠ IDENTICAL EMBEDDINGS (cos>0.98, L2<0.1)")
    
    # Check constant probability (if available)
    if "prob_std" in metrics:
        if metrics["prob_std"] < 0.01:
            flags.append("⚠ CONSTANT PROBABILITY (std<0.01)")
    
    # Check constant logits
    if "logits_std" in metrics:
        if metrics["logits_std"] < 0.01:
            flags.append("⚠ CONSTANT LOGITS (std<0.01)")
    
    # Check for prevalence prediction (logits nearly constant)
    if "logits_std" in metrics and "logits_min" in metrics and "logits_max" in metrics:
        logits_range = metrics["logits_max"] - metrics["logits_min"]
        logits_std = metrics["logits_std"]
        
        if logits_range > 1e-6:  # Avoid division by zero
            ratio = logits_std / logits_range
            if ratio < 0.1:  # std is <10% of range
                flags.append("⚠ PREVALENCE PREDICTION (logits nearly constant)")
    
    return flags


# ========================== MAIN PIPELINE ==========================
def main():
    """Main debugging pipeline."""
    
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create output directory
    # debug_dir = Path("debug_reports")
    # debug_dir.mkdir(exist_ok=True)
    
    # Check if config and checkpoint exist
    if not os.path.exists(CFG_PATH):
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    if not os.path.exists(CHKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHKPT_PATH}")
    
    print(f"Loading config from: {CFG_PATH}")
    print(f"Loading checkpoint from: {CHKPT_PATH}")
    print(f"Using device: {DEVICE}")
    print(f"Analyzing first {NUM_BATCHES} batches\n")
    
    # ==================== 1. SETUP ====================
    # Load config
    with open(CFG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_cfg = config['dataset']
    model_cfg = config['gnn']['model']
    training_cfg = config['gnn']['training']
    binary_cfg = config['binary_task']
    
    target_emotion = binary_cfg['target_emotion']
    threshold = binary_cfg.get('threshold', 0.0)
    
    print(f"Target emotion: {target_emotion}, threshold: {threshold}")
    print(f"Model params: in={model_cfg['in_channels']}, hidden={model_cfg['hidden_channels']}")
    
    # ==================== 2. DATA LOADING ====================
    print("\nLoading dataset...")
    base_dataset = SpacioTemporalDataset(
        root_dir=dataset_cfg.get('data_dir'),
        data_filepath=dataset_cfg.get('data_filepath'),
        filter_subjects=dataset_cfg.get('filter_subjects'),
        filter_recordings=dataset_cfg.get('filter_recordings'),
        file_list=dataset_cfg.get('file_list'),
        recursive=dataset_cfg.get('recursive', False),
        ignore_dirs=dataset_cfg.get('ignore_dirs', []),
        window_length=dataset_cfg['window_length'],
        window_overlap=dataset_cfg['window_overlap'],
        kt=dataset_cfg['kt'],
        ks=dataset_cfg['ks'],
        cache_dir=dataset_cfg.get('cache_dir'),
        use_cache=dataset_cfg.get('use_cache', True),
        dropping_emotion_threshold=dataset_cfg.get('dropping_emotion_threshold', -1),
    )
    
    dataset = BinarySpacioTemporalDataset(
        base_dataset,
        target_emotion=target_emotion,
        threshold=threshold
    )
    
    print(f"Loaded {len(dataset)} samples")
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=0,  # Debug mode: single worker
        pin_memory=True if DEVICE == "cuda" else False,
    )
    
    # ==================== 3. MODEL LOADING ====================
    print("\nLoading model...")
    model = BinarySpatioTemporalGNN(**model_cfg).to(DEVICE)
    
    # Load checkpoint
    state_dict = torch.load(CHKPT_PATH, map_location="cpu")
    
    # Handle case where checkpoint is wrapped in 'model_state' key
    if isinstance(state_dict, dict) and 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    
    # Handle torch.compile() wrapping with '_orig_mod.' prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded and set to eval mode")
    
    # ==================== PRINT HEAD NORMS ====================
    print("\n" + "="*80)
    print("HEAD LAYER ANALYSIS (Degeneracy Check)")
    print("="*80)
    head_norms = compute_head_norms(model)
    for layer_name, norm_val in head_norms.items():
        print(f"  {layer_name:<20} = {norm_val:>12.6f}")
    print("="*80)
    
    # ==================== 4. BATCH PROCESSING ====================
    print(f"\nProcessing batches...\n")
    
    all_batch_metrics = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= NUM_BATCHES:
                break
            
            data = data.to(DEVICE)
            batch_size = data["node"].batch.max().item() + 1
            
            print(f"Batch {batch_idx + 1}/{NUM_BATCHES} (size={batch_size})")
            
            # Extract inputs
            x_raw = data.x_dict["node"]  # [num_nodes, in_channels]
            edge_index_dict = data.edge_index_dict
            batch_vec = data["node"].batch  # [num_nodes]
            y_true = data.y.squeeze()  # [batch_size]
            
            pos_rate = (y_true > 0.5).float().mean().item()
            
            batch_metrics = {
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "pos_rate_true": pos_rate,
                "stages": {}
            }
            
            # ========== Raw Input Stage ==========
            stage_name = "input_raw"
            batch_metrics["stages"][stage_name] = {
                "name": stage_name,
                **compute_variance_metrics(x_raw),
                **compute_similarity_metrics(x_raw, MAX_PAIRS),
                **compute_within_graph_variance(x_raw, batch_vec),
            }
            
            # ========== Preprocessing Stage ==========
            if model.use_preprocess_mlp:
                x_pre = model.preprocess_mlp(x_raw)
                stage_name = "preprocess_mlp"
                
                batch_metrics["stages"][stage_name] = {
                    "name": stage_name,
                    **compute_variance_metrics(x_pre),
                    **compute_similarity_metrics(x_pre, MAX_PAIRS),
                    **compute_within_graph_variance(x_pre, batch_vec),
                }
            else:
                x_pre = x_raw
                stage_name = "preprocess_none"
                batch_metrics["stages"][stage_name] = batch_metrics["stages"]["input_raw"]
            
            # ========== Conv1 Stage ==========
            x1_dict = model.conv1({"node": x_pre}, edge_index_dict)
            x1 = model.gnn_dropout(model.gnn_activation(x1_dict["node"]))
            stage_name = "conv1"
            
            batch_metrics["stages"][stage_name] = {
                "name": stage_name,
                **compute_variance_metrics(x1),
                **compute_similarity_metrics(x1, MAX_PAIRS),
                **compute_within_graph_variance(x1, batch_vec),
            }
            
            # ========== Conv2 Stage ==========
            x2_dict = model.conv2({"node": x1}, edge_index_dict)
            x2 = model.gnn_dropout(model.gnn_activation(x2_dict["node"]))
            stage_name = "conv2"
            
            batch_metrics["stages"][stage_name] = {
                "name": stage_name,
                **compute_variance_metrics(x2),
                **compute_similarity_metrics(x2, MAX_PAIRS),
                **compute_within_graph_variance(x2, batch_vec),
            }
            
            # ========== Pooling Stage ==========
            g_pre = global_mean_pool(x_pre, batch_vec)  # [batch_size, D]
            g1 = global_mean_pool(x1, batch_vec)
            g2 = global_mean_pool(x2, batch_vec)
            
            stage_name = "pool_g2"
            batch_metrics["stages"][stage_name] = {
                "name": stage_name,
                **compute_variance_metrics(g2),
                **compute_similarity_metrics(g2, MAX_PAIRS),
            }
            
            # ========== Head Decomposition ==========
            # Head is: Linear -> GELU -> Dropout -> Linear (outputs raw logits)
            h1 = model.head[0](g2)  # Linear
            h_act = F.gelu(h1)  # GELU
            h_act = model.head[2](h_act)  # Dropout (no-op in eval)
            logits = model.head[3](h_act)  # Linear - raw logits
            prob = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
            
            stage_name = "head_logits"
            logits_metrics = compute_variance_metrics(logits)
            logits_metrics["prob_std"] = torch.std(prob).item()
            logits_metrics["logits_std"] = torch.std(logits).item()
            logits_metrics["logits_min"] = torch.min(logits).item()
            logits_metrics["logits_max"] = torch.max(logits).item()
            logits_metrics["logits_range"] = logits_metrics["logits_max"] - logits_metrics["logits_min"]
            
            # Compute how concentrated logits are (std / range ratio)
            if logits_metrics["logits_range"] > 1e-6:
                logits_metrics["logits_std_ratio"] = logits_metrics["logits_std"] / logits_metrics["logits_range"]
            else:
                logits_metrics["logits_std_ratio"] = 0.0
            
            batch_metrics["stages"][stage_name] = {
                "name": stage_name,
                **logits_metrics,
                **compute_similarity_metrics(logits, MAX_PAIRS),
            }
            
            stage_name = "head_prob"
            batch_metrics["stages"][stage_name] = {
                "name": stage_name,
                "var_mean": torch.var(prob).item(),
                "std_all": torch.std(prob).item(),
                "norm_mean": torch.mean(torch.abs(prob)).item(),
                "prob_mean": torch.mean(prob).item(),
                "prob_min": torch.min(prob).item(),
                "prob_max": torch.max(prob).item(),
            }
            
            all_batch_metrics.append(batch_metrics)
            
            # Print batch-level table
            print(f"  {'Stage':<20} {'var_mean':>12} {'cos_mean':>12} {'L2_mean':>12} {'within_var':>12} {'Flags':<50}")
            print(f"  {'-'*140}")
            
            prev_var_mean = None
            for stage_key in batch_metrics["stages"]:
                stage_data = batch_metrics["stages"][stage_key]
                var_mean = stage_data.get("var_mean", 0)
                cos_mean = stage_data.get("cos_mean", 0)
                l2_mean = stage_data.get("l2_mean", 0)
                within_var = stage_data.get("within_var_mean", None)
                
                flags = check_collapse_flags(stage_data, prev_var_mean)
                flag_str = " | ".join(flags) if flags else "✓"
                
                within_var_str = f"{within_var:.2e}" if within_var is not None else "N/A"
                
                print(f"  {stage_key:<20} {var_mean:>12.2e} {cos_mean:>12.4f} {l2_mean:>12.4f} {within_var_str:>12} {flag_str:<50}")
                
                # Extra info for head_logits
                if stage_key == "head_logits":
                    logits_min = stage_data.get("logits_min", 0)
                    logits_max = stage_data.get("logits_max", 0)
                    logits_range = stage_data.get("logits_range", 0)
                    logits_std = stage_data.get("logits_std", 0)
                    logits_std_ratio = stage_data.get("logits_std_ratio", 0)
                    print(f"    └─ logits: min={logits_min:.4f}, max={logits_max:.4f}, range={logits_range:.4f}")
                    print(f"    └─ logits_std={logits_std:.4f}, ratio(std/range)={logits_std_ratio:.4f}")
                
                prev_var_mean = var_mean
            
            print()
    
    # ==================== 5. AGGREGATE METRICS ====================
    print("\n" + "="*120)
    print("AGGREGATE METRICS (mean across batches)")
    print("="*120)
    
    # Compute aggregates
    all_stages = {}
    for batch_metrics in all_batch_metrics:
        for stage_key, stage_data in batch_metrics["stages"].items():
            if stage_key not in all_stages:
                all_stages[stage_key] = []
            all_stages[stage_key].append(stage_data)
    
    print(f"{'Stage':<20} {'var_mean':>12} {'std':>10} {'cos_mean':>12} {'L2_mean':>12}")
    print("-"*80)
    
    aggregate_report = {}
    
    for stage_key in all_stages:
        stage_list = all_stages[stage_key]
        
        var_means = [s.get("var_mean", 0) for s in stage_list]
        std_alls = [s.get("std_all", 0) for s in stage_list]
        cos_means = [s.get("cos_mean", 0) for s in stage_list]
        l2_means = [s.get("l2_mean", 0) for s in stage_list]
        
        var_mean_agg = np.mean(var_means)
        std_all_agg = np.mean(std_alls)
        cos_mean_agg = np.mean(cos_means)
        l2_mean_agg = np.mean(l2_means)
        
        print(f"{stage_key:<20} {var_mean_agg:>12.2e} {std_all_agg:>10.2e} {cos_mean_agg:>12.4f} {l2_mean_agg:>12.4f}")
        
        aggregate_report[stage_key] = {
            "var_mean": var_mean_agg,
            "std_all": std_all_agg,
            "cos_mean": cos_mean_agg,
            "l2_mean": l2_mean_agg,
        }
    
    # Print head logits distribution summary
    if "head_logits" in all_stages:
        print("\n" + "-"*80)
        print("HEAD LOGITS DISTRIBUTION SUMMARY")
        print("-"*80)
        
        head_metrics = all_stages["head_logits"]
        
        logits_mins = [s.get("logits_min", 0) for s in head_metrics]
        logits_maxs = [s.get("logits_max", 0) for s in head_metrics]
        logits_ranges = [s.get("logits_range", 0) for s in head_metrics]
        logits_stds = [s.get("logits_std", 0) for s in head_metrics]
        logits_std_ratios = [s.get("logits_std_ratio", 0) for s in head_metrics]
        
        print(f"  logits_min (mean):        {np.mean(logits_mins):>12.6f}")
        print(f"  logits_max (mean):        {np.mean(logits_maxs):>12.6f}")
        print(f"  logits_range (mean):      {np.mean(logits_ranges):>12.6f}")
        print(f"  logits_std (mean):        {np.mean(logits_stds):>12.6f}")
        print(f"  logits_std/range (mean):  {np.mean(logits_std_ratios):>12.6f}")
        
        avg_ratio = np.mean(logits_std_ratios)
        if avg_ratio < 0.1:
            print(f"\n  ⚠️  HEAD DEGENERACY DETECTED:")
            print(f"      Logits are nearly constant (std < 10% of range)")
            print(f"      Model is effectively predicting class prevalence only!")
        else:
            print(f"\n  ✓  Logits are well-distributed (std/range = {avg_ratio:.1%})")
    
    # ==================== 6. SAVE JSON REPORT ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # report_path = debug_dir / f"collapse_{timestamp}.json"
    
    report = {
        "metadata": {
            "timestamp": timestamp,
            "config_path": CFG_PATH,
            "checkpoint_path": CHKPT_PATH,
            "device": DEVICE,
            "num_batches_analyzed": len(all_batch_metrics),
            "num_batches_total": len(loader),
        },
        "config_snapshot": {
            "dataset": dataset_cfg,
            "model": model_cfg,
            "training": training_cfg,
            "binary_task": binary_cfg,
        },
        "per_batch_metrics": all_batch_metrics,
        "aggregate_metrics": aggregate_report,
        "pos_rate_summary": {
            "mean": np.mean([b["pos_rate_true"] for b in all_batch_metrics]),
            "std": np.std([b["pos_rate_true"] for b in all_batch_metrics]),
            "min": np.min([b["pos_rate_true"] for b in all_batch_metrics]),
            "max": np.max([b["pos_rate_true"] for b in all_batch_metrics]),
        },
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nJSON report saved to: {report_path}")
    print("\n" + "="*120)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*120)


if __name__ == "__main__":
    main()
