"""
Binary classification training script for emotion recognition.

Usage:
  python src/emotions/binary/train_binary.py --config src/emotions/binary/configs/train_binary.yaml
"""

import os
import sys
import argparse
import yaml
import warnings
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
# resolve warnings
torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import joblib

# Add src directory only when executed as a script (not as installed package).
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from data.data import SpacioTemporalDataset
from emotions.common.cv_utils import (
    build_split_entries as build_common_split_entries,
    describe_fold as describe_common_fold,
    split_group_tokens as split_common_group_tokens,
    validate_non_empty_train_splits as validate_common_non_empty_train_splits,
)
from emotions.common.dataset_config import (
    build_graph_dataset_kwargs,
    build_tabular_samples_kwargs,
    resolve_dropna_columns,
    resolve_feature_columns,
    resolve_min_samples_per_window,
)
from emotions.train_baseline import build_tabular_samples, samples_to_xy
from emotions.utils import (
    Logger,
    load_config,
    create_splitter,
    save_comparison_csv,
    print_comparison_table
)
from emotions.binary.model_binary import BinarySpatioTemporalGNN
from emotions.binary.baseline_model_binary import get_binary_baseline_by_name
from emotions.binary.metrics_binary import evaluate_binary_classification
from emotions.binary.results_plotting import generate_and_save_binary_results_plots



def parse_args():
    parser = argparse.ArgumentParser(description="Train binary classification models")
    parser.add_argument(
        "--config",
        type=str,
        default="src/emotions/binary/configs/train_binary.yaml",
        help="Path to binary config YAML file"
    )
    return parser.parse_args()


def resolve_target_column(binary_task_cfg: Dict[str, Any]) -> str:
    """Resolve a single target column with backward-compatible config keys."""
    target_column = binary_task_cfg.get("target_column", binary_task_cfg.get("target_emotion"))
    if not target_column:
        raise ValueError("Binary task requires 'target_column' (or legacy 'target_emotion').")
    return target_column


def resolve_threshold_value(
    threshold_spec: Any,
    train_values: List[float],
) -> float:
    """Resolve a fold threshold from numeric value or train-based statistic."""
    if isinstance(threshold_spec, str):
        mode = threshold_spec.strip().lower()
        if mode in {"median", "mean"}:
            series = pd.to_numeric(pd.Series(train_values), errors="coerce").dropna()
            if series.empty:
                raise ValueError("Cannot compute threshold from empty train targets.")
            if mode == "median":
                return float(series.median())
            return float(series.mean())
        try:
            return float(mode)
        except ValueError as exc:
            raise ValueError(
                f"Invalid threshold '{threshold_spec}'. Use float, 'median', or 'mean'."
            ) from exc
    return float(threshold_spec)


def validate_non_empty_train_splits(
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    strategy: str,
    dataset_label: str,
) -> None:
    """Backward-compatible wrapper around shared split validation helper."""
    validate_common_non_empty_train_splits(
        splits=splits,
        strategy=strategy,
        dataset_label=dataset_label,
    )


def collect_graph_target_values(
    dataset: SpacioTemporalDataset,
    indices: np.ndarray,
    target_column: str,
) -> List[float]:
    """Collect continuous target values from selected graph samples."""
    values: List[float] = []
    for idx in indices:
        data = dataset[int(idx)]
        names = getattr(data, "emotion_names", getattr(dataset, "emotion_names", []))
        if target_column not in names:
            raise ValueError(f"Target column '{target_column}' not found in graph targets.")
        target_idx = names.index(target_column)
        values.append(float(data.y[target_idx].item()))
    return values


def collect_tabular_target_values(
    samples: List[Any],
    indices: np.ndarray,
    target_column: str,
) -> List[float]:
    """Collect continuous target values from selected tabular samples."""
    values: List[float] = []
    for idx in indices:
        sample = samples[int(idx)]
        if target_column not in sample.targets:
            raise ValueError(f"Target column '{target_column}' missing in tabular sample.")
        values.append(float(sample.targets[target_column]))
    return values


def split_group_tokens(
    strategy: str,
    dataset: List[Any],
    indices: np.ndarray,
) -> Tuple[str, ...]:
    """Backward-compatible wrapper for split-token generation."""
    return split_common_group_tokens(
        strategy=strategy,
        dataset=dataset,
        indices=indices,
        combined_id_style="pipe",
    )


def describe_fold(
    strategy: str,
    dataset: List[Any],
    test_idx: np.ndarray,
    fold_num: int,
) -> Tuple[str, str, Tuple[str, Tuple[str, ...]]]:
    """Backward-compatible wrapper for fold identity derivation."""
    fold = describe_common_fold(
        strategy=strategy,
        dataset=dataset,
        test_idx=test_idx,
        fold_num=fold_num,
        combined_id_style="pipe",
    )
    return fold.test_id, fold.test_name, fold.fold_key


def build_split_entries(
    strategy: str,
    dataset: List[Any],
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper around shared split-entry construction."""
    return build_common_split_entries(
        strategy=strategy,
        dataset=dataset,
        splits=splits,
        combined_id_style="pipe",
    )


def fit_graph_feature_scaler(
    dataset: SpacioTemporalDataset,
    train_idx: np.ndarray,
) -> StandardScaler:
    """Fit StandardScaler on node features from train graphs only."""
    train_x = []
    for idx in train_idx:
        data = dataset[int(idx)]
        train_x.append(data["node"].x.detach().cpu().numpy())
    scaler = StandardScaler()
    scaler.fit(np.vstack(train_x))
    return scaler


def build_binary_graph_subset(
    dataset: SpacioTemporalDataset,
    indices: np.ndarray,
    target_column: str,
    threshold_value: float,
    scaler: StandardScaler | None = None,
) -> List[Any]:
    """Build a list of graphs with binary targets and optional standardized features."""
    graphs = []
    for idx in indices:
        data = dataset[int(idx)].clone()
        names = getattr(data, "emotion_names", getattr(dataset, "emotion_names", []))
        if target_column not in names:
            raise ValueError(f"Target column '{target_column}' not found in graph targets.")
        target_idx = names.index(target_column)
        target_value = float(data.y[target_idx].item())
        binary_label = 1.0 if target_value > threshold_value else 0.0
        data.y = torch.tensor([binary_label], dtype=torch.float32)
        if scaler is not None:
            x_scaled = scaler.transform(data["node"].x.detach().cpu().numpy())
            data["node"].x = torch.tensor(x_scaled, dtype=torch.float32)
        graphs.append(data)
    return graphs


def train_gnn_epoch(model, loader, optimizer, device, grad_clip_max_norm=1.0):
    """Train GNN for one epoch with binary cross-entropy with logits loss.
    
    Model outputs raw logits; loss function includes sigmoid internally.
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        out = model(data).reshape(-1)   # [batch_size] logits; safe when batch_size==1
        target = data.y.reshape(-1)     # [batch_size]
        
        # Binary cross-entropy with logits (includes sigmoid internally)
        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate_gnn(model, loader, device, emotion_name, decision_threshold=0.5):
    """Evaluate GNN binary classifier."""
    model.eval()
    total_loss = 0
    
    all_outputs = []
    all_targets = []
    all_subjects = []
    all_recordings = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).reshape(-1)   # [batch_size] logits; safe when batch_size==1
            target = data.y.reshape(-1)     # [batch_size]
            
            # Loss uses logits directly
            loss = F.binary_cross_entropy_with_logits(out, target)
            total_loss += loss.item()
            
            # Convert logits to probabilities for metrics
            prob = torch.sigmoid(out)
            all_outputs.append(prob.cpu())
            all_targets.append(target.cpu())
            
            # Collect metadata directly from batch if available
            batch_subjects = getattr(data, "subject", None)
            batch_recordings = getattr(data, "recording", None)
            if isinstance(batch_subjects, (list, tuple)):
                all_subjects.extend(batch_subjects)
            elif batch_subjects is not None:
                all_subjects.append(batch_subjects)
            if isinstance(batch_recordings, (list, tuple)):
                all_recordings.extend(batch_recordings)
            elif batch_recordings is not None:
                all_recordings.append(batch_recordings)
    
    # Concatenate predictions
    y_pred = torch.cat(all_outputs).numpy()
    y_true = torch.cat(all_targets).numpy()
    
    # Prepare metadata
    metadata = None
    if all_subjects and all_recordings:
        metadata = {
            'subjects': all_subjects,
            'recordings': all_recordings
        }
    
    # Compute metrics
    metrics = evaluate_binary_classification(
        y_pred, y_true,
        metadata=metadata,
        emotion_names=[emotion_name],
        threshold=decision_threshold
    )
    
    return metrics, total_loss / len(loader), y_pred, y_true


def train_gnn_fold(
    config,
    train_idx,
    val_idx,
    test_idx,
    dataset,
    fold_dir,
    test_name,
    device,
    target_column: str,
    threshold_value: float,
    standardize_features: bool = False,
    verbose: bool = True,
):
    """Train GNN for one fold."""
    model_cfg = config['gnn']['model']
    training_cfg = config['gnn']['training']
    decision_threshold = config['binary_task'].get('decision_threshold', 0.5)
    use_compile = training_cfg.get('use_torch_compile', True)

    def _build_loader_kwargs(*, safe_mode: bool) -> Dict[str, Any]:
        """Build DataLoader kwargs; safe_mode disables multiprocessing/pin-memory."""
        if safe_mode:
            return {
                "batch_size": training_cfg["batch_size"],
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            }

        num_workers = int(training_cfg.get("num_workers", 4))
        pin_memory = bool(training_cfg.get("pin_memory", True)) if device.type == "cuda" else False
        persistent_workers = bool(training_cfg.get("persistent_workers", True)) and num_workers > 0
        return {
            "batch_size": training_cfg["batch_size"],
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }

    def _is_loader_thread_error(exc: RuntimeError) -> bool:
        """Return True for known DataLoader pin-memory/IPC worker failures."""
        message = str(exc)
        return (
            "Pin memory thread exited unexpectedly" in message
            or "received 0 items of ancdata" in message
        )
    
    scaler = None
    if standardize_features:
        scaler = fit_graph_feature_scaler(dataset, train_idx)
        joblib.dump(scaler, os.path.join(fold_dir, "gnn_feature_scaler.pkl"))

    train_graphs = build_binary_graph_subset(
        dataset=dataset,
        indices=train_idx,
        target_column=target_column,
        threshold_value=threshold_value,
        scaler=scaler,
    )
    val_graphs = build_binary_graph_subset(
        dataset=dataset,
        indices=val_idx,
        target_column=target_column,
        threshold_value=threshold_value,
        scaler=scaler,
    )
    test_graphs = build_binary_graph_subset(
        dataset=dataset,
        indices=test_idx,
        target_column=target_column,
        threshold_value=threshold_value,
        scaler=scaler,
    )

    def _run_one_attempt(*, safe_loader_mode: bool) -> Dict[str, Any]:
        """Run one full fold training/evaluation attempt."""
        # Create model per attempt so retries start from a clean state.
        model = BinarySpatioTemporalGNN(**model_cfg).to(device)
        if use_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="default")

        optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])
        loader_kwargs = _build_loader_kwargs(safe_mode=safe_loader_mode)
        train_loader = DataLoader(train_graphs, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_graphs, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_graphs, shuffle=False, **loader_kwargs)

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_epochs = 0
        early_stopped = False
        start_time = time.time()
        early_stopping_enabled = bool(training_cfg.get("early_stopping_enabled", False))
        early_stopping_patience = int(training_cfg.get("early_stopping_patience", 7))
        early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
        early_stopping_restore_best = bool(training_cfg.get("early_stopping_restore_best", True))

        if early_stopping_enabled and early_stopping_patience < 1:
            raise ValueError("gnn.training.early_stopping_patience must be >= 1 when early stopping is enabled.")

        mode_note = " [safe-loader mode]" if safe_loader_mode else ""
        print(f"Training GNN for {test_name}{mode_note}...")

        for epoch in range(training_cfg["num_epochs"]):
            train_loss = train_gnn_epoch(
                model,
                train_loader,
                optimizer,
                device,
                training_cfg.get("grad_clip_max_norm", 1.0),
            )

            val_metrics, val_loss, _, _ = evaluate_gnn(
                model, val_loader, device, target_column, decision_threshold
            )

            if verbose and ((epoch + 1) % 10 == 0 or epoch == training_cfg["num_epochs"] - 1 or epoch == 0):
                val_acc = val_metrics["standard"]["aggregated"]["accuracy"]
                print(
                    f"  Epoch {epoch+1}/{training_cfg['num_epochs']}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

            if val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_epochs = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))
            else:
                no_improve_epochs += 1

            if early_stopping_enabled and no_improve_epochs >= early_stopping_patience:
                early_stopped = True
                if verbose:
                    print(
                        f"  Early stopping at epoch {epoch+1}: "
                        f"no val_loss improvement for {early_stopping_patience} epoch(s)."
                    )
                break

        print(f"  Best model at epoch {best_epoch+1}")
        if early_stopped:
            print("  Early stopping triggered.")
        if not early_stopping_restore_best and verbose:
            print(
                "  NOTE: early_stopping_restore_best=false requested, "
                "but evaluation still uses the best checkpoint for comparability."
            )
        print(f"  GNN train time: {time.time() - start_time:.2f} seconds")

        model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pt")))
        test_metrics, _, test_pred, test_true = evaluate_gnn(
            model, test_loader, device, target_column, decision_threshold
        )

        np.save(os.path.join(fold_dir, "test_predictions.npy"), test_pred)
        np.save(os.path.join(fold_dir, "test_targets.npy"), test_true)

        test_acc = test_metrics["standard"]["aggregated"]["accuracy"]
        if verbose:
            print(f"  ❗GNN - Test Accuracy: {test_acc:.4f}")
        return test_metrics

    try:
        return _run_one_attempt(safe_loader_mode=False)
    except RuntimeError as exc:
        if not _is_loader_thread_error(exc):
            raise
        # Persist safe DataLoader settings for subsequent folds in this run.
        training_cfg["num_workers"] = 0
        training_cfg["pin_memory"] = False
        training_cfg["persistent_workers"] = False
        print(
            "  Warning: DataLoader pin-memory/multiprocessing failed "
            f"({exc}). Retrying fold in safe-loader mode and applying safe loader "
            "settings for subsequent folds."
        )
        return _run_one_attempt(safe_loader_mode=True)


def train_baselines_fold(
    baseline_cfg,
    train_idx,
    val_idx,
    test_idx,
    samples,
    fold_dir,
    metric_names,
    target_column: str,
    threshold_value: float,
    standardize_features: bool = False,
    feature_columns: List[str] | None = None,
    verbose: bool = True,
):
    """Train baseline models for one fold and save them with predictions."""
    baselines_dir = os.path.join(fold_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)

    X_train, y_train, train_meta, feat_cols, targ_cols = samples_to_xy(samples, train_idx)
    X_val, y_val, val_meta, _, _ = samples_to_xy(samples, val_idx)
    X_test, y_test, test_meta, _, _ = samples_to_xy(samples, test_idx)

    # Keep baseline features aligned with configured signal columns.
    if feature_columns:
        selected_feature_cols: List[str] = []
        for col in feature_columns:
            if col in X_train.columns:
                selected_feature_cols.append(col)
            elif f"{col}_mean" in X_train.columns:
                selected_feature_cols.append(f"{col}_mean")

        if not selected_feature_cols:
            raise ValueError(
                "Configured feature columns were not found in tabular baseline features. "
                f"Configured: {feature_columns}. Available: {feat_cols[:20]}..."
            )

        X_train = X_train[selected_feature_cols].copy()
        X_val = X_val[selected_feature_cols].copy()
        X_test = X_test[selected_feature_cols].copy()

    if target_column not in y_train.columns:
        raise ValueError(f"Target column '{target_column}' is missing in tabular labels.")

    y_train_cont = pd.to_numeric(y_train[target_column], errors="coerce")
    y_val_cont = pd.to_numeric(y_val[target_column], errors="coerce")
    y_test_cont = pd.to_numeric(y_test[target_column], errors="coerce")

    # Drop rows with NaN in either features or targets per split.
    train_mask = (~X_train.isna().any(axis=1)) & (~y_train_cont.isna())
    val_mask = (~X_val.isna().any(axis=1)) & (~y_val_cont.isna())
    test_mask = (~X_test.isna().any(axis=1)) & (~y_test_cont.isna())

    dropped_train = int((~train_mask).sum())
    dropped_val = int((~val_mask).sum())
    dropped_test = int((~test_mask).sum())
    if dropped_train or dropped_val or dropped_test:
        print(
            "Dropped NaN rows for baselines "
            f"(train={dropped_train}, val={dropped_val}, test={dropped_test})."
        )

    X_train = X_train.loc[train_mask].reset_index(drop=True)
    X_val = X_val.loc[val_mask].reset_index(drop=True)
    X_test = X_test.loc[test_mask].reset_index(drop=True)
    y_train_cont = y_train_cont.loc[train_mask].reset_index(drop=True)
    y_val_cont = y_val_cont.loc[val_mask].reset_index(drop=True)
    y_test_cont = y_test_cont.loc[test_mask].reset_index(drop=True)
    test_meta = [m for m, keep in zip(test_meta, test_mask.tolist()) if keep]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "Baseline split became empty after removing NaN rows. "
            "Check feature/dropna configuration."
        )

    y_train = (y_train_cont > threshold_value).astype(float).to_frame(name=target_column)
    y_val = (y_val_cont > threshold_value).astype(float).to_frame(name=target_column)
    y_test = (y_test_cont > threshold_value).astype(float).to_frame(name=target_column)

    scaler = None
    if standardize_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    test_metadata = {
        "subjects": [m[0] for m in test_meta if m],
        "recordings": [m[1] for m in test_meta if m],
    }

    results = {}
    for model_name in baseline_cfg["models"]:
        if verbose:
            print(f"  Training {model_name}...")

        model_dir = os.path.join(baselines_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        hyperparams = baseline_cfg.get("hyperparameters", {}).get(model_name, {})
        model = get_binary_baseline_by_name(model_name, **hyperparams)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*Stochastic Optimizer.*",
            )
            model.fit(X_train, y_train)

        test_metrics = model.evaluate(
            X_test,
            y_test,
            emotion_names=[target_column],
            metadata=test_metadata,
            threshold=0.5,
        )

        test_pred = model.predict_proba(X_test)
        test_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        if scaler is not None:
            joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.pkl"))

        np.save(os.path.join(model_dir, "test_predictions.npy"), test_pred)
        np.save(os.path.join(model_dir, "test_targets.npy"), test_true)
        results[model_name] = test_metrics

        test_acc = test_metrics["standard"]["aggregated"]["accuracy"]
        if verbose:
            print(f"    ❗{model_name} - Test Accuracy: {test_acc:.4f}")

    return results


def run_training_from_config(config_path: str) -> str:
    """Run full binary training pipeline from one YAML config path.

    Args:
        config_path: Path to binary training configuration file.

    Returns:
        Absolute/relative run directory where artifacts were saved.
    """
    config = load_config(config_path)
    
    # Extract config sections
    run_experiments = config['run_experiments']
    dataset_cfg = config['dataset']
    binary_task_cfg = config['binary_task']
    cv_cfg = config['cross_validation']
    logging_cfg = config['logging']
    metric_names = config['metrics']
    verbose = logging_cfg.get('verbose', True)
    
    target_column = resolve_target_column(binary_task_cfg)
    threshold_spec = binary_task_cfg.get("threshold", 0.0)
    standardize_features = dataset_cfg.get("standardize_features", False)
    target_aggregation = dataset_cfg.get("target_aggregation", "mean")
    if target_aggregation not in {"mean", "last"}:
        raise ValueError(
            f"Unsupported dataset.target_aggregation='{target_aggregation}'. "
            "Use 'mean' or 'last'."
        )

    print("Binary Classification Task:")
    print(f"  Target column: {target_column}")
    print(f"  Threshold spec: {threshold_spec}")
    print("  Labels: <=threshold -> 0, >threshold -> 1")
    print(f"  Standardize features: {standardize_features}")
    print(f"  Target aggregation: {target_aggregation}")
    
    # Create timestamped run directory; optional prefix supports suite orchestration.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_prefix = str(logging_cfg.get("run_name_prefix", "")).strip()
    run_name = f"{run_name_prefix}_{timestamp}" if run_name_prefix else timestamp
    run_dir = os.path.join(logging_cfg['results_dir'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(run_dir, 'training_log.txt')
    logger = Logger(log_file)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger

    try:
        print(f"\n{'#'*50} \nTraining started at: {datetime.now()}")
        print(f"Results will be saved to: {run_dir}")
        print(f"Run baselines: {run_experiments['baselines']}")
        print(f"Run GNN: {run_experiments['gnn']}")
        
        # Save config
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        
        # Set device for GNN
        if run_experiments['gnn']:
            training_cfg = config['gnn']['training']
            if training_cfg['device'] == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(training_cfg['device'])
            print(f"Using device: {device}")
            
            # Set random seed
            if training_cfg.get('random_seed') is not None:
                torch.manual_seed(training_cfg['random_seed'])
                np.random.seed(training_cfg['random_seed'])
        
        # Load datasets
        print("\nLoading datasets...")
        target_columns = [target_column]
        feature_columns = resolve_feature_columns(dataset_cfg)
        dropna_columns = resolve_dropna_columns(dataset_cfg, target_columns=target_columns)
        min_samples_per_window = resolve_min_samples_per_window(dataset_cfg)
        
        if run_experiments['gnn']:
            print("Loading graph dataset for GNN...")
            base_gnn_dataset = SpacioTemporalDataset(
                **build_graph_dataset_kwargs(
                    dataset_cfg=dataset_cfg,
                    target_columns=target_columns,
                    feature_columns=feature_columns,
                    dropna_columns=dropna_columns,
                ),
            )
            print(f"Loaded {len(base_gnn_dataset)} graph samples")

        
        if run_experiments['baselines']:
            print("Loading tabular samples for baselines...")
            base_tabular_samples = build_tabular_samples(
                **build_tabular_samples_kwargs(
                    dataset_cfg=dataset_cfg,
                    target_columns=target_columns,
                    feature_columns=feature_columns,
                    dropna_columns=dropna_columns,
                    min_samples_per_window=min_samples_per_window,
                ),
            )
            print(f"Loaded {len(base_tabular_samples)} tabular samples")
            unique_subjects = set(
                sample.subject
                for sample in base_tabular_samples
                if hasattr(sample, "subject") and sample.subject is not None
            )
            unique_recordings = set(
                sample.recording
                for sample in base_tabular_samples
                if hasattr(sample, "recording") and sample.recording is not None
            )
            print(f"Unique subjects in tabular samples: {sorted(unique_subjects)}")
            print(f"Unique recordings in tabular samples: {sorted(unique_recordings)}")
        
        # Get CV strategies
        strategies = cv_cfg['strategies']
        if isinstance(strategies, str):
            strategies = [strategies]
        
        print(f"\nWill run experiments with {len(strategies)} strategy(ies): {', '.join(strategies)}")
        
        # Storage for all results
        all_strategies_results = {}
        
        # Run experiments for each strategy
        for strategy in strategies:
            print("\n" + "="*100)
            print(f"Starting cross-validation with strategy: {strategy.upper()}")
            print("="*100)
            
            strategy_dir = os.path.join(run_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Create splitters
            if run_experiments['baselines']:
                baseline_splitter = create_splitter(
                    strategy=strategy,
                    samples=base_tabular_samples,
                    val_size=cv_cfg['val_size'],
                    random_state=cv_cfg.get('random_state'),
                    n_splits=cv_cfg.get('n_splits', 3),
                )
            
            if run_experiments['gnn']:
                gnn_splitter = create_splitter(
                    strategy=strategy,
                    samples=base_gnn_dataset,
                    val_size=cv_cfg['val_size'],
                    random_state=cv_cfg.get('random_state'),
                    n_splits=cv_cfg.get('n_splits', 3),
                )
            
            # Storage for this strategy
            baseline_results_all_folds = {name: {} for name in config['baselines']['models']} if run_experiments['baselines'] else {}
            gnn_results_all_folds = {}
            
            baseline_entries: List[Dict[str, Any]] = []
            gnn_entries: List[Dict[str, Any]] = []
            entry_plan: List[Dict[str, Any]] = []

            # Get splits and build fold entries
            if run_experiments['baselines'] and run_experiments['gnn']:
                baseline_splits = list(baseline_splitter.split())
                gnn_splits = list(gnn_splitter.split())
                validate_non_empty_train_splits(baseline_splits, strategy, "Baseline")
                validate_non_empty_train_splits(gnn_splits, strategy, "GNN")
                baseline_entries = build_split_entries(
                    strategy=strategy,
                    dataset=base_tabular_samples,
                    splits=baseline_splits,
                )
                gnn_entries = build_split_entries(
                    strategy=strategy,
                    dataset=base_gnn_dataset,
                    splits=gnn_splits,
                )
                entry_plan = gnn_entries
            elif run_experiments['baselines']:
                baseline_splits = list(baseline_splitter.split())
                validate_non_empty_train_splits(baseline_splits, strategy, "Baseline")
                baseline_entries = build_split_entries(
                    strategy=strategy,
                    dataset=base_tabular_samples,
                    splits=baseline_splits,
                )
                entry_plan = baseline_entries
            else:
                gnn_splits = list(gnn_splitter.split())
                validate_non_empty_train_splits(gnn_splits, strategy, "GNN")
                gnn_entries = build_split_entries(
                    strategy=strategy,
                    dataset=base_gnn_dataset,
                    splits=gnn_splits,
                )
                entry_plan = gnn_entries

            baseline_by_key: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
            if run_experiments['baselines']:
                baseline_by_key = {entry["fold_key"]: entry for entry in baseline_entries}
            gnn_by_key: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
            if run_experiments['gnn']:
                gnn_by_key = {entry["fold_key"]: entry for entry in gnn_entries}

            if run_experiments['baselines'] and run_experiments['gnn']:
                baseline_keys = set(baseline_by_key.keys())
                gnn_keys = set(gnn_by_key.keys())
                if baseline_keys != gnn_keys:
                    only_baseline = sorted(baseline_keys - gnn_keys)
                    only_gnn = sorted(gnn_keys - baseline_keys)
                    raise ValueError(
                        f"Split mismatch for strategy '{strategy}': baseline and GNN do not share the same fold identities. "
                        f"Baseline-only folds: {len(only_baseline)}, GNN-only folds: {len(only_gnn)}. "
                        "To ensure model comparability, both must use identical folds."
                    )
                for fold_key in sorted(gnn_keys):
                    baseline_signature = baseline_by_key[fold_key]["split_signature"]
                    gnn_signature = gnn_by_key[fold_key]["split_signature"]
                    if baseline_signature != gnn_signature:
                        raise ValueError(
                            f"Split mismatch for strategy '{strategy}' on fold {fold_key}: "
                            "baseline and GNN have different train/val/test group assignments. "
                            "Exact split equality is required for comparable model evaluation."
                        )
                entry_plan = gnn_entries
            
            for entry in entry_plan:
                test_id = entry["test_id"]
                test_name = entry["test_name"]

                baseline_entry = None
                if run_experiments['baselines']:
                    baseline_entry = baseline_by_key.get(entry["fold_key"])
                    if baseline_entry is not None:
                        baseline_train_idx = baseline_entry["train_idx"]
                        baseline_val_idx = baseline_entry["val_idx"]
                        baseline_test_idx = baseline_entry["test_idx"]
                    elif not run_experiments['gnn']:
                        raise RuntimeError("Unexpected missing baseline fold entry.")

                if run_experiments['gnn']:
                    gnn_entry = gnn_by_key.get(entry["fold_key"], entry)
                    gnn_train_idx = gnn_entry["train_idx"]
                    gnn_val_idx = gnn_entry["val_idx"]
                    gnn_test_idx = gnn_entry["test_idx"]
                
                fold_dir = os.path.join(strategy_dir, test_id)
                os.makedirs(fold_dir, exist_ok=True)
                
                print(f"\n{test_name}")

                if run_experiments["gnn"]:
                    train_values = collect_graph_target_values(
                        base_gnn_dataset,
                        gnn_train_idx,
                        target_column,
                    )
                else:
                    train_values = collect_tabular_target_values(
                        base_tabular_samples,
                        baseline_train_idx,
                        target_column,
                    )
                if len(train_values) == 0:
                    raise ValueError(
                        f"Empty train targets for {test_name} with strategy '{strategy}'. "
                        "This split cannot compute a train-based threshold."
                    )
                fold_threshold = resolve_threshold_value(threshold_spec, train_values)
                print(f"  Fold label threshold ({threshold_spec}): {fold_threshold:.6f}")
                
                # Train baselines
                if run_experiments['baselines']:
                    if baseline_entry is None:
                        raise RuntimeError(
                            "Missing baseline fold after fold intersection; this indicates inconsistent fold mapping."
                        )
                    print("Training baselines...")
                    baseline_results = train_baselines_fold(
                        config['baselines'], baseline_train_idx, baseline_val_idx,
                        baseline_test_idx, base_tabular_samples, fold_dir,
                        metric_names, target_column, fold_threshold,
                        standardize_features=standardize_features,
                        feature_columns=feature_columns,
                        verbose=verbose,
                    )
                    for model_name, metrics in baseline_results.items():
                        baseline_results_all_folds[model_name][test_id] = metrics
                
                # Train GNN
                if run_experiments['gnn']:
                    gnn_metrics = train_gnn_fold(
                        config, gnn_train_idx, gnn_val_idx, gnn_test_idx,
                        base_gnn_dataset, fold_dir, test_name, device,
                        target_column=target_column,
                        threshold_value=fold_threshold,
                        standardize_features=standardize_features,
                        verbose=verbose,
                    )
                    gnn_results_all_folds[test_id] = gnn_metrics
            
            # Combine results
            combined_results = {}
            if run_experiments['baselines']:
                combined_results.update(baseline_results_all_folds)
            if run_experiments['gnn']:
                combined_results['GNN'] = gnn_results_all_folds
            
            all_strategies_results[strategy] = combined_results
            
            # Print and save comparison
            print_comparison_table(combined_results, metric_names, strategy)
            csv_path = os.path.join(strategy_dir, 'summary.csv')
            save_comparison_csv(combined_results, metric_names, csv_path)

        print("\nGenerating result plots...")
        models_for_cm: List[str] = []
        if run_experiments['gnn']:
            models_for_cm.append('GNN')
        if run_experiments['baselines']:
            models_for_cm.extend(config['baselines']['models'])
        models_for_cm = list(dict.fromkeys(models_for_cm))
        try:
            saved_plots = generate_and_save_binary_results_plots(
                run_dir=Path(run_dir),
                decision_threshold=float(binary_task_cfg.get('decision_threshold', 0.5)),
                models_for_cm=models_for_cm if models_for_cm else None,
            )
            for plot_path in saved_plots:
                print(f"Saved plot: {plot_path}")
        except Exception as exc:
            print(f"Warning: failed to generate result plots: {exc}")
        
        print(f"\n{'='*100}")
        print("Training complete!")
        print(f"All results saved to: {run_dir}")
        return run_dir
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.close()


def main() -> None:
    args = parse_args()
    run_dir = run_training_from_config(args.config)
    print(f"Binary run directory: {run_dir}")


if __name__ == "__main__":
    main()
