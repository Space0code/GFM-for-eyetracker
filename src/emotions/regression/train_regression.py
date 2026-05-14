"""Train single-target regression emotion models (GNN + baselines) with CV.

Usage:
  python src/emotions/regression/train_regression.py \
      --config src/emotions/regression/configs/train_regression_hci_tagging.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

# Resolve warnings and matmul precision
torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from data.data import SpacioTemporalDataset
from emotions.common.cv_utils import (
    describe_fold,
    validate_kfold_group_disjointness,
    validate_non_empty_train_splits,
)
from emotions.common.dataset_config import (
    build_graph_dataset_kwargs,
    build_tabular_samples_kwargs,
    resolve_dropna_columns,
    resolve_feature_columns,
    resolve_min_samples_per_window,
    sync_gnn_in_channels,
)
from emotions.baseline_model import get_baseline_by_name
from emotions.model import SpatioTemporalHeteroGNN, SpatioTemporalHeteroGNNV1
from emotions.metrics import compute_metrics
from emotions.train_baseline import build_tabular_samples, samples_to_xy
from emotions.utils import (
    Logger,
    create_splitter,
    load_config,
    print_comparison_table,
    save_comparison_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression models")
    parser.add_argument(
        "--config",
        type=str,
        default="src/emotions/regression/configs/train_regression_hci_tagging.yaml",
        help="Path to regression YAML config file",
    )
    return parser.parse_args()


def _validate_non_empty_train_splits(
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    strategy: str,
    dataset_label: str,
) -> None:
    validate_non_empty_train_splits(
        splits=splits,
        strategy=strategy,
        dataset_label=dataset_label,
    )


def _validate_kfold_group_disjointness(
    splits: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    strategy: str,
    dataset: List[Any],
    dataset_label: str,
) -> None:
    validate_kfold_group_disjointness(
        splits=splits,
        strategy=strategy,
        dataset=dataset,
        dataset_label=dataset_label,
        combined_id_style="underscore",
    )


def _fit_graph_feature_scaler(dataset: SpacioTemporalDataset, train_idx: np.ndarray) -> StandardScaler:
    arrays = []
    for idx in train_idx:
        graph = dataset[int(idx)]
        arrays.append(graph["node"].x.detach().cpu().numpy())
    scaler = StandardScaler()
    scaler.fit(np.vstack(arrays))
    return scaler


def _build_graph_subset(
    dataset: SpacioTemporalDataset,
    indices: np.ndarray,
    target_column: str,
    scaler: StandardScaler | None,
) -> List[Any]:
    graphs: List[Any] = []
    for idx in indices:
        graph = dataset[int(idx)].clone()
        names = getattr(graph, "emotion_names", getattr(dataset, "emotion_names", []))
        if target_column not in names:
            raise ValueError(f"Graph target column '{target_column}' not found.")
        col_idx = names.index(target_column)
        target_value = float(graph.y[col_idx].item())
        graph.y = torch.tensor([target_value], dtype=torch.float32)

        if scaler is not None:
            x_scaled = scaler.transform(graph["node"].x.detach().cpu().numpy())
            graph["node"].x = torch.tensor(x_scaled, dtype=torch.float32)

        graphs.append(graph)
    return graphs


def _train_gnn_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_max_norm: float,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch).reshape(-1)
        targets = batch.y.reshape(-1)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def _evaluate_gnn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_column: str,
) -> Tuple[Dict[str, Any], float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0

    all_outputs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_metadata: List[Tuple[str, str]] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch).reshape(-1)
            targets = batch.y.reshape(-1)
            loss = F.mse_loss(outputs, targets)
            total_loss += float(loss.item())

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

            batch_subjects = getattr(batch, "subject", None)
            batch_recordings = getattr(batch, "recording", None)

            subjects: List[Any]
            recordings: List[Any]

            if isinstance(batch_subjects, (list, tuple)):
                subjects = list(batch_subjects)
            elif batch_subjects is not None:
                subjects = [batch_subjects]
            else:
                subjects = []

            if isinstance(batch_recordings, (list, tuple)):
                recordings = list(batch_recordings)
            elif batch_recordings is not None:
                recordings = [batch_recordings]
            else:
                recordings = []

            if subjects and recordings and len(subjects) == len(recordings):
                all_metadata.extend([(str(s), str(r)) for s, r in zip(subjects, recordings)])

    y_pred = torch.cat(all_outputs).numpy().reshape(-1, 1)
    y_true = torch.cat(all_targets).numpy().reshape(-1, 1)

    metadata = all_metadata if len(all_metadata) == len(y_true) else None

    metrics = compute_metrics(
        y_pred=y_pred,
        y_true=y_true,
        emotion_names=[target_column],
        metadata=metadata,
    )

    avg_loss = total_loss / max(len(loader), 1)
    metrics["standard"]["aggregated"]["loss"] = avg_loss
    if metrics.get("per_pair_aggregated") is not None:
        metrics["per_pair_aggregated"]["aggregated"]["loss"] = avg_loss

    return metrics, avg_loss, y_pred, y_true


def _train_gnn_fold(
    config: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: SpacioTemporalDataset,
    fold_dir: str,
    test_name: str,
    device: torch.device,
    target_column: str,
    standardize_features: bool,
    verbose: bool,
) -> Dict[str, Any]:
    model_cfg = config["gnn"]["model"]
    training_cfg = config["gnn"]["training"]

    model_version = str(model_cfg.get("model_version", "v2")).lower()
    if model_version == "v1":
        model_cls = SpatioTemporalHeteroGNNV1
    elif model_version == "v2":
        model_cls = SpatioTemporalHeteroGNN
    else:
        raise ValueError(f"Unsupported gnn.model.model_version='{model_version}'. Choose 'v1' or 'v2'.")

    model_kwargs = {
        "in_channels": model_cfg["in_channels"],
        "hidden_channels": model_cfg["hidden_channels"],
        "out_channels": 1,
        "output_scale": float(model_cfg.get("output_scale", 1.0)),
        "use_preprocess_mlp": model_cfg.get("use_preprocess_mlp", True),
        "use_edge_weights": config["dataset"].get("use_edge_weights", True),
        "add_self_loops": model_cfg.get("add_self_loops", False),
        "dropout_mlp": model_cfg.get("dropout_mlp", 0.1),
        "dropout_gnn": model_cfg.get("dropout_gnn", 0.1),
        "dropout_head": model_cfg.get("dropout_head", 0.1),
        "aggr": model_cfg.get("aggr", "mean"),
        "conv_type": model_cfg.get("conv_type", "GCNConv"),
        "num_layers": model_cfg.get("num_layers", 2),
        "pooling": model_cfg.get("pooling", "attention" if model_version == "v2" else "mean_max"),
    }
    if model_version == "v2":
        model_kwargs["head_pooling"] = model_cfg.get("head_pooling")
        model_kwargs["graph_pooling"] = model_cfg.get(
            "graph_pooling",
            model_cfg.get("head_pooling", model_cfg.get("pooling", "attention")),
        )
        model_kwargs["relation_pooling"] = model_cfg.get("relation_pooling", "mlp")
        model_kwargs["edge_weight_mode"] = model_cfg.get(
            "edge_weight_mode",
            config["dataset"].get("edge_weight_mode", "learned_signed"),
        )
        model_kwargs["use_delta_distance_edge_feature"] = model_cfg.get(
            "use_delta_distance_edge_feature",
            bool(config["dataset"].get("use_delta_distance_edge_feature", True)),
        )
        model_kwargs["use_fixation_edges"] = model_cfg.get(
            "use_fixation_edges",
            bool(config["dataset"].get("use_fixation_edges", True)),
        )

    model = model_cls(**model_kwargs).to(device)

    use_compile = training_cfg.get("use_torch_compile", True)
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="default")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))

    loader_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(training_cfg.get("num_workers", 4)),
        "pin_memory": bool(training_cfg.get("pin_memory", True)) if device.type == "cuda" else False,
        "persistent_workers": bool(training_cfg.get("persistent_workers", True)),
    }

    scaler: StandardScaler | None = None
    if standardize_features:
        scaler = _fit_graph_feature_scaler(dataset=dataset, train_idx=train_idx)
        joblib.dump(scaler, os.path.join(fold_dir, "gnn_feature_scaler.pkl"))

    train_graphs = _build_graph_subset(dataset, train_idx, target_column, scaler)
    val_graphs = _build_graph_subset(dataset, val_idx, target_column, scaler)
    test_graphs = _build_graph_subset(dataset, test_idx, target_column, scaler)

    train_loader = DataLoader(train_graphs, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_graphs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_graphs, shuffle=False, **loader_kwargs)

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    print(f"Training regression GNN for {test_name}...")
    for epoch in range(int(training_cfg["num_epochs"])):
        train_loss = _train_gnn_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_max_norm=float(training_cfg.get("grad_clip_max_norm", 1.0)),
        )
        val_metrics, val_loss, _, _ = _evaluate_gnn(
            model=model,
            loader=val_loader,
            device=device,
            target_column=target_column,
        )

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == int(training_cfg["num_epochs"])):
            val_mae = val_metrics["standard"]["aggregated"].get("mae", np.nan)
            print(
                f"  Epoch {epoch + 1}/{training_cfg['num_epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))

    print(f"  Best model at epoch {best_epoch + 1}")
    print(f"  GNN train time: {time.time() - start_time:.2f} seconds")

    model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pt")))
    test_metrics, _, test_pred, test_true = _evaluate_gnn(
        model=model,
        loader=test_loader,
        device=device,
        target_column=target_column,
    )

    np.save(os.path.join(fold_dir, "test_predictions.npy"), test_pred)
    np.save(os.path.join(fold_dir, "test_targets.npy"), test_true)

    if verbose:
        test_mae = test_metrics["standard"]["aggregated"].get("mae", np.nan)
        print(f"  ❗GNN - Test MAE: {test_mae:.4f}")

    return test_metrics


def _prepare_baseline_split(
    samples: List[Any],
    indices: np.ndarray,
    feature_columns: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]]:
    X, y, metadata, _, _ = samples_to_xy(samples, indices)

    selected_cols: List[str] = []
    for col in feature_columns:
        if col in X.columns:
            selected_cols.append(col)
        elif f"{col}_mean" in X.columns:
            selected_cols.append(f"{col}_mean")

    if not selected_cols:
        raise ValueError(
            "Configured feature columns were not found in tabular baseline features. "
            f"Configured: {feature_columns}."
        )

    if target_column not in y.columns:
        raise ValueError(f"Target column '{target_column}' missing in tabular labels.")

    X = X[selected_cols].copy()
    target_series = pd.to_numeric(y[target_column], errors="coerce")

    valid_mask = (~X.isna().any(axis=1)) & (~target_series.isna())
    X = X.loc[valid_mask].reset_index(drop=True)
    y_clean = target_series.loc[valid_mask].reset_index(drop=True).to_frame(name=target_column)
    metadata = [meta for meta, keep in zip(metadata, valid_mask.tolist()) if keep]

    if len(X) == 0:
        raise ValueError("Baseline split became empty after NaN filtering.")

    return X, y_clean, metadata


def _train_baselines_fold(
    baseline_cfg: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    samples: List[Any],
    fold_dir: str,
    target_column: str,
    standardize_features: bool,
    feature_columns: List[str],
    verbose: bool,
) -> Dict[str, Dict[str, Any]]:
    baselines_dir = os.path.join(fold_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)

    X_train, y_train, _ = _prepare_baseline_split(samples, train_idx, feature_columns, target_column)
    X_val, y_val, _ = _prepare_baseline_split(samples, val_idx, feature_columns, target_column)
    X_test, y_test, test_meta = _prepare_baseline_split(samples, test_idx, feature_columns, target_column)

    scaler: StandardScaler | None = None
    if standardize_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    results: Dict[str, Dict[str, Any]] = {}

    for model_name in baseline_cfg["models"]:
        if verbose:
            print(f"  Training {model_name}...")

        model_dir = os.path.join(baselines_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model = get_baseline_by_name(
            model_name,
            **baseline_cfg.get("hyperparameters", {}).get(model_name, {}),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Stochastic Optimizer.*")
            model.fit(X_train, y_train)

        test_metrics = model.evaluate(
            X=X_test,
            y=y_test,
            emotion_names=[target_column],
            metadata=test_meta,
        )
        test_pred = model.predict(X_test)

        with open(os.path.join(model_dir, "model.pkl"), "wb") as handle:
            pickle.dump(model, handle)
        if scaler is not None:
            joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.pkl"))

        np.save(os.path.join(model_dir, "test_predictions.npy"), np.asarray(test_pred).reshape(-1, 1))
        np.save(os.path.join(model_dir, "test_targets.npy"), y_test.values.reshape(-1, 1))

        results[model_name] = test_metrics

        if verbose:
            test_mae = test_metrics["standard"]["aggregated"].get("mae", np.nan)
            print(f"    ❗{model_name} - Test MAE: {test_mae:.4f}")

    return results


def _generate_regression_plots(
    run_dir: str,
    candidate_metrics: Tuple[str, ...] = ("mae", "ccc", "spearman"),
) -> None:
    run_path = Path(run_dir)
    strategy_dirs = sorted(
        [path for path in run_path.iterdir() if path.is_dir() and (path / "summary.csv").exists()]
    )
    if not strategy_dirs:
        return

    frames: List[pd.DataFrame] = []
    for strategy_dir in strategy_dirs:
        df = pd.read_csv(strategy_dir / "summary.csv")
        if "metric_type" in df.columns:
            df = df[df["metric_type"] != "aggregated"]
        df["strategy"] = strategy_dir.name
        frames.append(df)

    results_df = pd.concat(frames, ignore_index=True)
    metrics = [metric for metric in candidate_metrics if metric in results_df.columns]
    if not metrics:
        return

    figures_dir = run_path / "plots"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(strategy_dirs), figsize=(6 * len(strategy_dirs), 4), sharey=True)
    if len(strategy_dirs) == 1:
        axes = [axes]

    for ax, strategy_dir in zip(axes, strategy_dirs):
        subset = results_df[results_df["strategy"] == strategy_dir.name]
        if subset.empty:
            ax.set_axis_off()
            continue
        subset.plot(x="model", y=metrics, kind="bar", ax=ax)
        ax.set_title(strategy_dir.name)
        ax.set_xlabel("model")
        ax.set_ylabel("metric")

    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_barplots.png", dpi=300)
    plt.close(fig)


def run_training_from_config(config_path: str) -> str:
    """Run full regression training using one YAML config.

    Returns:
        Path to created training run directory.
    """
    config = load_config(config_path)

    run_experiments = config["run_experiments"]
    dataset_cfg = config["dataset"]
    regression_task_cfg = config["regression_task"]
    cv_cfg = config["cross_validation"]
    logging_cfg = config["logging"]
    metric_names = config["metrics"]
    verbose = bool(logging_cfg.get("verbose", True))

    target_column = regression_task_cfg.get("target_column")
    if not target_column:
        raise ValueError("regression_task.target_column is required.")

    standardize_features = bool(dataset_cfg.get("standardize_features", False))
    target_aggregation = dataset_cfg.get("target_aggregation", "mean")
    if target_aggregation not in {"mean", "last"}:
        raise ValueError(
            f"Unsupported dataset.target_aggregation='{target_aggregation}'. Use 'mean' or 'last'."
        )

    print("Regression Task:")
    print(f"  target_column: {target_column}")
    print(f"  standardize_features: {standardize_features}")
    print(f"  target_aggregation: {target_aggregation}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_prefix = str(logging_cfg.get("run_name_prefix", "")).strip()
    run_name = f"{run_name_prefix}_{timestamp}" if run_name_prefix else timestamp
    run_dir = os.path.join(logging_cfg["results_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "training_log.txt")
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

        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        if run_experiments["gnn"]:
            training_cfg = config["gnn"]["training"]
            if training_cfg.get("device", "auto") == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(training_cfg["device"])
            print(f"Using device: {device}")

            if training_cfg.get("random_seed") is not None:
                seed = int(training_cfg["random_seed"])
                torch.manual_seed(seed)
                np.random.seed(seed)

        print("\nLoading datasets...")
        target_columns = [target_column]

        feature_columns = resolve_feature_columns(dataset_cfg)
        sync_gnn_in_channels(config["gnn"]["model"], feature_columns)
        dropna_columns = resolve_dropna_columns(dataset_cfg, target_columns=target_columns)
        min_samples_per_window = resolve_min_samples_per_window(dataset_cfg)
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        base_gnn_dataset = None
        if run_experiments["gnn"]:
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

        base_tabular_samples = None
        if run_experiments["baselines"]:
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

        strategies = cv_cfg["strategies"]
        if isinstance(strategies, str):
            strategies = [strategies]

        print(f"\nWill run {len(strategies)} strategy(ies): {', '.join(strategies)}")

        all_strategies_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for strategy in strategies:
            print("\n" + "=" * 100)
            print(f"Starting CV strategy: {strategy.upper()}")
            print("=" * 100)

            strategy_dir = os.path.join(run_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)

            baseline_splitter = None
            gnn_splitter = None
            if run_experiments["baselines"] and base_tabular_samples is not None:
                baseline_splitter = create_splitter(
                    strategy=strategy,
                    samples=base_tabular_samples,
                    val_size=int(cv_cfg["val_size"]),
                    random_state=cv_cfg.get("random_state"),
                    n_splits=int(cv_cfg.get("n_splits", 3)),
                )
            if run_experiments["gnn"] and base_gnn_dataset is not None:
                gnn_splitter = create_splitter(
                    strategy=strategy,
                    samples=base_gnn_dataset,
                    val_size=int(cv_cfg["val_size"]),
                    random_state=cv_cfg.get("random_state"),
                    n_splits=int(cv_cfg.get("n_splits", 3)),
                )

            reference_splitter = gnn_splitter if gnn_splitter is not None else baseline_splitter
            reference_dataset = base_gnn_dataset if gnn_splitter is not None else base_tabular_samples
            if reference_splitter is None or reference_dataset is None:
                raise ValueError("No splitter/dataset available for CV.")

            baseline_results_all_folds = (
                {name: {} for name in config["baselines"]["models"]}
                if run_experiments["baselines"]
                else {}
            )
            gnn_results_all_folds: Dict[str, Dict[str, Any]] = {}

            if baseline_splitter is not None and gnn_splitter is not None:
                baseline_splits = list(baseline_splitter.split())
                gnn_splits = list(gnn_splitter.split())
                _validate_non_empty_train_splits(baseline_splits, strategy, "Baseline")
                _validate_non_empty_train_splits(gnn_splits, strategy, "GNN")
                _validate_kfold_group_disjointness(
                    baseline_splits,
                    strategy=strategy,
                    dataset=base_tabular_samples,
                    dataset_label="Baseline",
                )
                _validate_kfold_group_disjointness(
                    gnn_splits,
                    strategy=strategy,
                    dataset=base_gnn_dataset,
                    dataset_label="GNN",
                )
                num_folds = len(baseline_splits)
            elif baseline_splitter is not None:
                baseline_splits = list(baseline_splitter.split())
                _validate_non_empty_train_splits(baseline_splits, strategy, "Baseline")
                _validate_kfold_group_disjointness(
                    baseline_splits,
                    strategy=strategy,
                    dataset=base_tabular_samples,
                    dataset_label="Baseline",
                )
                num_folds = len(baseline_splits)
            else:
                gnn_splits = list(gnn_splitter.split())
                _validate_non_empty_train_splits(gnn_splits, strategy, "GNN")
                _validate_kfold_group_disjointness(
                    gnn_splits,
                    strategy=strategy,
                    dataset=base_gnn_dataset,
                    dataset_label="GNN",
                )
                num_folds = len(gnn_splits)

            for fold_num in range(num_folds):
                if baseline_splitter is not None:
                    baseline_train_idx, baseline_val_idx, baseline_test_idx = baseline_splits[fold_num]
                if gnn_splitter is not None:
                    gnn_train_idx, gnn_val_idx, gnn_test_idx = gnn_splits[fold_num]

                ref_test_idx = gnn_test_idx if gnn_splitter is not None else baseline_test_idx
                fold = describe_fold(
                    strategy=strategy,
                    dataset=reference_dataset,
                    test_idx=ref_test_idx,
                    fold_num=fold_num,
                    combined_id_style="underscore",
                )
                test_id = fold.test_id
                test_name = fold.test_name

                fold_dir = os.path.join(strategy_dir, test_id)
                os.makedirs(fold_dir, exist_ok=True)

                print(f"\n{test_name}")

                if run_experiments["baselines"] and base_tabular_samples is not None:
                    print("Training baselines...")
                    baseline_results = _train_baselines_fold(
                        baseline_cfg=config["baselines"],
                        train_idx=baseline_train_idx,
                        val_idx=baseline_val_idx,
                        test_idx=baseline_test_idx,
                        samples=base_tabular_samples,
                        fold_dir=fold_dir,
                        target_column=target_column,
                        standardize_features=standardize_features,
                        feature_columns=feature_columns,
                        verbose=verbose,
                    )
                    for model_name, metrics in baseline_results.items():
                        baseline_results_all_folds[model_name][test_id] = metrics

                if run_experiments["gnn"] and base_gnn_dataset is not None:
                    gnn_metrics = _train_gnn_fold(
                        config=config,
                        train_idx=gnn_train_idx,
                        val_idx=gnn_val_idx,
                        test_idx=gnn_test_idx,
                        dataset=base_gnn_dataset,
                        fold_dir=fold_dir,
                        test_name=test_name,
                        device=device,
                        target_column=target_column,
                        standardize_features=standardize_features,
                        verbose=verbose,
                    )
                    gnn_results_all_folds[test_id] = gnn_metrics

            combined_results: Dict[str, Dict[str, Any]] = {}
            if run_experiments["baselines"]:
                combined_results.update(baseline_results_all_folds)
            if run_experiments["gnn"]:
                combined_results["GNN"] = gnn_results_all_folds

            all_strategies_results[strategy] = combined_results

            print_comparison_table(combined_results, metric_names, strategy)
            save_comparison_csv(combined_results, metric_names, os.path.join(strategy_dir, "summary.csv"))

        print("\nGenerating regression result plots...")
        try:
            _generate_regression_plots(run_dir)
            print(f"Saved plots under: {os.path.join(run_dir, 'figures')}")
        except Exception as exc:  # pragma: no cover - plotting should not fail full run
            print(f"Warning: failed to generate regression plots: {exc}")

        print("\n" + "=" * 100)
        print("Regression training complete!")
        print(f"All results saved to: {run_dir}")
        return run_dir

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.close()


def main() -> None:
    args = parse_args()
    run_dir = run_training_from_config(args.config)
    print(f"Regression run directory: {run_dir}")


if __name__ == "__main__":
    main()
