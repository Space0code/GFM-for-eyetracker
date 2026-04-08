"""Backfill GNN analysis artifacts for an existing binary run directory.

This script reconstructs fold test graphs from a completed binary run config,
loads each fold checkpoint (`best_model.pt`), runs GNN inference, and writes
`gnn_test_analysis_artifacts.npz` needed for TP/FP/TN/FN raw/embedding plots.

It can also generate/update the fold-aggregated plots after backfilling.

Usage:
  python src/emotions/binary/backfill_gnn_analysis_artifacts.py \
      --run-dir results/binary/<run_name> \
      --generate-plots

Common options:
  --run-dir PATH            Existing binary run directory (required).
  --device auto|cpu|cuda    Inference device (default: auto).
  --skip-existing           Skip folds that already have analysis artifacts.
  --embedding-method pca|tsne
                            Embedding projection for generated plots
                            (default: pca).
  --generate-plots          Generate/update figures after backfill.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from data.data import SpacioTemporalDataset
from emotions.binary.model_binary import BinarySpatioTemporalGNN
from emotions.binary.results_plotting import generate_and_save_binary_results_plots
from emotions.binary.train_binary import (
    build_binary_graph_subset,
    build_split_entries,
    collect_graph_target_values,
    evaluate_gnn,
    fit_graph_feature_scaler,
    resolve_target_column,
    resolve_threshold_value,
    save_gnn_test_analysis_artifacts,
    validate_kfold_group_disjointness,
    validate_non_empty_train_splits,
)
from emotions.common.dataset_config import (
    build_graph_dataset_kwargs,
    resolve_dropna_columns,
    resolve_feature_columns,
)
from emotions.utils import create_splitter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Backfill GNN analysis artifacts for a binary run")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to existing binary run directory (must contain config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folds that already contain gnn_test_analysis_artifacts.npz.",
    )
    parser.add_argument(
        "--embedding-method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Embedding projection method used when --generate-plots is set.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate binary result plots after backfilling.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve torch device from CLI argument."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load run config YAML from binary run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        registry_path = run_dir / "suite_experiment_registry.csv"
        if registry_path.exists():
            raise ValueError(
                f"Received suite root '{run_dir}', but binary backfill expects one trainer run dir "
                "(containing config.yaml). Use "
                "`python src/emotions/suite/backfill_suite_analysis_artifacts.py --suite-run-dir <suite_dir>` "
                "for suite-wide backfill."
            )
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid run config content at {config_path}")
    return payload


def _resolve_data_filepath_for_run(run_dir: Path, configured_path: str) -> str:
    """Resolve dataset CSV path for moved/copied run directories."""
    configured = Path(configured_path)
    suite_experiments_candidate: Path | None = None

    if "experiments" in configured.parts:
        experiments_idx = configured.parts.index("experiments")
        suite_experiments_candidate = run_dir.parent / Path(*configured.parts[experiments_idx:])

    candidates: List[Path] = [configured]
    if not configured.is_absolute():
        candidates.extend([run_dir / configured, Path.cwd() / configured])
    if suite_experiments_candidate is not None:
        candidates.append(suite_experiments_candidate)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return configured_path


def _build_graph_dataset(config: Dict[str, Any], run_dir: Path) -> SpacioTemporalDataset:
    """Build graph dataset exactly as in training config."""
    dataset_cfg = dict(config["dataset"])
    data_filepath = dataset_cfg.get("data_filepath")
    if isinstance(data_filepath, str) and data_filepath.strip():
        resolved_path = _resolve_data_filepath_for_run(
            run_dir=run_dir,
            configured_path=data_filepath,
        )
        if resolved_path != data_filepath:
            print(
                "Resolved dataset.data_filepath for current run directory: "
                f"{data_filepath} -> {resolved_path}"
            )
            dataset_cfg["data_filepath"] = resolved_path

    target_column = resolve_target_column(config["binary_task"])
    target_columns = [target_column]
    feature_columns = resolve_feature_columns(dataset_cfg)
    dropna_columns = resolve_dropna_columns(dataset_cfg, target_columns=target_columns)
    kwargs = build_graph_dataset_kwargs(
        dataset_cfg=dataset_cfg,
        target_columns=target_columns,
        feature_columns=feature_columns,
        dropna_columns=dropna_columns,
    )
    return SpacioTemporalDataset(**kwargs)


def _load_fold_scaler_or_fit(
    fold_dir: Path,
    dataset: SpacioTemporalDataset,
    train_idx: np.ndarray,
    standardize_features: bool,
) -> Any:
    """Load per-fold scaler when available, else fit from train split."""
    if not standardize_features:
        return None
    scaler_path = fold_dir / "gnn_feature_scaler.pkl"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return fit_graph_feature_scaler(dataset=dataset, train_idx=train_idx)


def _build_loader_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create evaluation DataLoader kwargs from training config."""
    training_cfg = config["gnn"]["training"]
    batch_size = int(training_cfg.get("batch_size", 64))
    return {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "shuffle": False,
    }


def _remap_state_dict_keys(state_dict: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Return state_dict with optional key-prefix normalization."""
    remapped: Dict[str, Any] = {}

    def _transform_key(key: str) -> str:
        if mode == "identity":
            return key
        if mode == "strip_orig_mod":
            return key.removeprefix("_orig_mod.")
        if mode == "strip_module":
            return key.removeprefix("module.")
        if mode == "strip_orig_mod_and_module":
            return key.removeprefix("_orig_mod.").removeprefix("module.")
        raise ValueError(f"Unsupported key remap mode: {mode}")

    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("Checkpoint state_dict has non-string key.")
        mapped_key = _transform_key(key)
        if mapped_key in remapped:
            raise ValueError(
                f"Key collision after remapping checkpoint keys (mode={mode}): {mapped_key}"
            )
        remapped[mapped_key] = value

    return remapped


def _load_checkpoint_state_dict(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Load checkpoint into model with robust key remapping for compiled checkpoints."""
    raw_state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(raw_state, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} did not contain a state_dict dictionary.")

    modes = [
        "identity",
        "strip_orig_mod",
        "strip_module",
        "strip_orig_mod_and_module",
    ]
    errors: List[str] = []
    for mode in modes:
        try:
            candidate_state = _remap_state_dict_keys(raw_state, mode=mode)
            model.load_state_dict(candidate_state)
            if mode != "identity":
                print(f"  Loaded checkpoint with key remapping mode: {mode}")
            return
        except Exception as exc:
            errors.append(f"{mode}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        f"Failed to load checkpoint with supported key mappings: {checkpoint_path}\n{joined_errors}"
    )


def backfill_run(run_dir: Path, device: torch.device, skip_existing: bool) -> None:
    """Backfill all strategy/fold GNN analysis artifacts in one run directory."""
    config = _load_run_config(run_dir)
    run_experiments = config.get("run_experiments", {})
    if not bool(run_experiments.get("gnn", False)):
        raise ValueError("Run config has run_experiments.gnn=false; no GNN artifacts to backfill.")

    print(f"Loading dataset for backfill from config: {run_dir / 'config.yaml'}")
    dataset = _build_graph_dataset(config, run_dir=run_dir)
    print(f"Loaded {len(dataset)} graph windows.")

    dataset_cfg = config["dataset"]
    binary_task_cfg = config["binary_task"]
    cv_cfg = config["cross_validation"]
    target_column = resolve_target_column(binary_task_cfg)
    threshold_spec = binary_task_cfg.get("threshold", 0.0)
    decision_threshold = float(binary_task_cfg.get("decision_threshold", 0.5))
    standardize_features = bool(dataset_cfg.get("standardize_features", False))
    loader_kwargs = _build_loader_kwargs(config)

    strategies = cv_cfg["strategies"]
    if isinstance(strategies, str):
        strategies = [strategies]

    for strategy in strategies:
        strategy_dir = run_dir / strategy
        if not strategy_dir.exists():
            print(f"[skip] Strategy directory missing: {strategy_dir}")
            continue

        splitter = create_splitter(
            strategy=strategy,
            samples=dataset,
            val_size=int(cv_cfg.get("val_size", 1)),
            random_state=cv_cfg.get("random_state"),
            n_splits=int(cv_cfg.get("n_splits", 3)),
        )
        splits = list(splitter.split())
        validate_non_empty_train_splits(splits=splits, strategy=strategy, dataset_label="GNN")
        validate_kfold_group_disjointness(
            splits=splits,
            strategy=strategy,
            dataset=dataset,
            dataset_label="GNN",
        )

        entries = build_split_entries(strategy=strategy, dataset=dataset, splits=splits)
        print(f"\nStrategy {strategy}: {len(entries)} fold(s)")

        for entry in entries:
            fold_dir = strategy_dir / str(entry["test_id"])
            if not fold_dir.exists():
                print(f"  [skip] Fold directory missing: {fold_dir.name}")
                continue

            artifact_path = fold_dir / "gnn_test_analysis_artifacts.npz"
            if skip_existing and artifact_path.exists():
                print(f"  [skip] Existing artifacts: {fold_dir.name}")
                continue

            checkpoint_path = fold_dir / "best_model.pt"
            if not checkpoint_path.exists():
                print(f"  [skip] Missing checkpoint: {fold_dir.name}/best_model.pt")
                continue

            train_idx = entry["train_idx"]
            test_idx = entry["test_idx"]

            train_values = collect_graph_target_values(
                dataset=dataset,
                indices=train_idx,
                target_column=target_column,
            )
            if not train_values:
                print(f"  [skip] Empty train targets for fold: {fold_dir.name}")
                continue

            fold_threshold = resolve_threshold_value(threshold_spec, train_values)
            scaler = _load_fold_scaler_or_fit(
                fold_dir=fold_dir,
                dataset=dataset,
                train_idx=train_idx,
                standardize_features=standardize_features,
            )
            test_graphs = build_binary_graph_subset(
                dataset=dataset,
                indices=test_idx,
                target_column=target_column,
                threshold_value=fold_threshold,
                scaler=scaler,
            )
            test_loader = DataLoader(test_graphs, **loader_kwargs)

            model = BinarySpatioTemporalGNN(**config["gnn"]["model"]).to(device)
            _load_checkpoint_state_dict(model=model, checkpoint_path=checkpoint_path, device=device)

            _, _, y_pred, y_true, artifacts = evaluate_gnn(
                model=model,
                loader=test_loader,
                device=device,
                emotion_name=target_column,
                decision_threshold=decision_threshold,
                collect_analysis_artifacts=True,
            )
            save_gnn_test_analysis_artifacts(
                fold_dir=str(fold_dir),
                y_pred=y_pred,
                y_true=y_true,
                decision_threshold=decision_threshold,
                artifacts=artifacts,
            )
            print(f"  [ok] Backfilled: {fold_dir.name}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    device = _resolve_device(args.device)
    print(f"Using device: {device}")
    backfill_run(run_dir=run_dir, device=device, skip_existing=bool(args.skip_existing))

    if args.generate_plots:
        config = _load_run_config(run_dir)
        decision_threshold = float(config["binary_task"].get("decision_threshold", 0.5))
        saved_plots = generate_and_save_binary_results_plots(
            run_dir=run_dir,
            decision_threshold=decision_threshold,
            models_for_cm=["GNN"],
            embedding_method=str(args.embedding_method),
        )
        for path in saved_plots:
            print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
