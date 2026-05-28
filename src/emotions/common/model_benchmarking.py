"""Utilities for recording model scale and timing artifacts.

The helpers in this module intentionally measure practical wall-clock cost
around the existing training pipeline. They do not estimate FLOPs. For CUDA
models, timings synchronize the device before and after each measured block so
asynchronous kernels are included in elapsed time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


DEFAULT_BENCHMARK_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "warmup_runs": 1,
    "timed_runs": 3,
}


def resolve_benchmark_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve optional benchmark settings from a training config."""
    raw = config.get("benchmarking", {}) if isinstance(config, Mapping) else {}
    if raw is None:
        raw = {}
    resolved = dict(DEFAULT_BENCHMARK_CONFIG)
    resolved.update(dict(raw))
    resolved["enabled"] = bool(resolved.get("enabled", True))
    resolved["warmup_runs"] = max(0, int(resolved.get("warmup_runs", 1)))
    resolved["timed_runs"] = max(1, int(resolved.get("timed_runs", 3)))
    return resolved


def unwrap_torch_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module when `torch.compile` wrapped the model."""
    return getattr(model, "_orig_mod", model)


def count_torch_parameters(model: torch.nn.Module | None) -> Dict[str, int | None]:
    """Count trainable and total parameters for a PyTorch module."""
    if model is None:
        return {"trainable_parameters": None, "total_parameters": None}
    module = unwrap_torch_model(model)
    parameters = list(module.parameters())
    return {
        "trainable_parameters": int(sum(param.numel() for param in parameters if param.requires_grad)),
        "total_parameters": int(sum(param.numel() for param in parameters)),
    }


def count_checkpoint_state_parameters(path: Path | str) -> int:
    """Count tensor entries in a checkpoint `model_state_dict` or raw state dict."""
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, Mapping):
        return 0
    return int(
        sum(value.numel() for value in state_dict.values() if isinstance(value, torch.Tensor))
    )


def count_gazemae_encoder_parameters(paths: Sequence[Path | str]) -> int:
    """Count frozen GazeMAE encoder parameters from local encoder checkpoints."""
    return int(sum(count_checkpoint_state_parameters(path) for path in paths))


def maybe_synchronize(device: torch.device | str | None) -> None:
    """Synchronize CUDA work when timing a CUDA-backed block."""
    if device is None:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def measure_repeated_wall_time(
    run_once: Callable[[], Any],
    *,
    n_items: int,
    warmup_runs: int,
    timed_runs: int,
    device: torch.device | str | None = None,
) -> Dict[str, float | int]:
    """Measure repeated wall-clock time for a callable and normalize per item."""
    for _ in range(max(0, int(warmup_runs))):
        run_once()
    maybe_synchronize(device)

    started_ns = time.perf_counter_ns()
    for _ in range(max(1, int(timed_runs))):
        run_once()
    maybe_synchronize(device)
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0

    total_items = int(n_items) * max(1, int(timed_runs))
    seconds_per_item = elapsed_seconds / max(1, total_items)
    return {
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "inference_total_seconds": float(elapsed_seconds),
        "inference_items_per_run": int(n_items),
        "inference_total_item_predictions": int(total_items),
        "inference_seconds_per_window": float(seconds_per_item),
        "inference_ms_per_window": float(seconds_per_item * 1000.0),
    }


def measure_gnn_inference_time(
    *,
    model: torch.nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    n_items: int,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float | int | str]:
    """Measure GNN prediction time over a materialized PyG loader."""
    model.eval()

    def run_once() -> None:
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                _ = torch.softmax(logits, dim=1)

    timing = measure_repeated_wall_time(
        run_once,
        n_items=n_items,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        device=device,
    )
    timing["inference_scope"] = "materialized_graph_loader_forward_softmax"
    return timing


def measure_predict_proba_time(
    *,
    predict_proba: Callable[[], Any],
    n_items: int,
    warmup_runs: int,
    timed_runs: int,
    device: torch.device | str | None = None,
    inference_scope: str,
) -> Dict[str, float | int | str]:
    """Measure a tabular baseline `predict_proba` callable."""
    timing = measure_repeated_wall_time(
        predict_proba,
        n_items=n_items,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        device=device,
    )
    timing["inference_scope"] = inference_scope
    return timing


def measure_gazemae_encoder_head_time(
    *,
    head_model: torch.nn.Module,
    model_pos_path: Path | str,
    model_vel_path: Path | str,
    device: torch.device | str,
    window_seconds: float,
    target_hz: int,
    chunk_seconds: float,
    synthetic_windows_per_run: int,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float | int | str]:
    """Measure frozen GazeMAE encoder + pooling + MLP head forward time.

    The input is synthetic zeros with the same `[x, y]` chunk shape used by the
    MAHNOB-HCI adapter. This isolates model-forward cost and excludes CSV/window
    loading, clipping, interpolation, and velocity construction.
    """
    from emotions.gazemae_model import load_gazemae_encoder

    resolved_device = torch.device(device)
    head_model.eval()
    pos_encoder = load_gazemae_encoder(model_pos_path, device=resolved_device)
    vel_encoder = load_gazemae_encoder(model_vel_path, device=resolved_device)

    target_len = int(round(float(target_hz) * float(window_seconds)))
    chunk_len = int(round(float(target_hz) * float(chunk_seconds)))
    chunks_per_window = max(1, target_len // max(1, chunk_len))
    n_windows = max(1, int(synthetic_windows_per_run))
    n_chunks = n_windows * chunks_per_window
    pos_chunks = torch.zeros((n_chunks, 2, chunk_len), dtype=torch.float32, device=resolved_device)
    vel_chunks = torch.zeros_like(pos_chunks)

    def run_once() -> None:
        with torch.inference_mode():
            pos_encoded = pos_encoder.encode(pos_chunks)[0]
            vel_encoded = vel_encoder.encode(vel_chunks)[0]
            chunk_embeddings = torch.cat([pos_encoded, vel_encoded], dim=1)
            chunk_embeddings = chunk_embeddings.reshape(n_windows, chunks_per_window, -1)
            pooled = torch.cat(
                [
                    chunk_embeddings.mean(dim=1),
                    chunk_embeddings.std(dim=1, unbiased=False),
                ],
                dim=1,
            )
            logits = head_model(pooled)
            _ = torch.softmax(logits, dim=1)

    timing = measure_repeated_wall_time(
        run_once,
        n_items=n_windows,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        device=resolved_device,
    )
    return {
        "encoder_head_inference_scope": "synthetic_gazemae_encoder_pooling_head_forward",
        "encoder_head_synthetic_windows_per_run": int(n_windows),
        "encoder_head_chunks_per_window": int(chunks_per_window),
        "encoder_head_inference_total_seconds": timing["inference_total_seconds"],
        "encoder_head_inference_total_item_predictions": timing["inference_total_item_predictions"],
        "encoder_head_inference_seconds_per_window": timing["inference_seconds_per_window"],
        "encoder_head_inference_ms_per_window": timing["inference_ms_per_window"],
    }


def describe_classical_model_scale(model: Any) -> Dict[str, Any]:
    """Return non-neural model scale proxies for raw benchmark artifacts."""
    estimator = getattr(model, "model", None)
    if estimator is None:
        return {}

    if hasattr(estimator, "support_vectors_"):
        n_support = getattr(estimator, "n_support_", None)
        return {
            "scale_proxy_name": "support_vectors",
            "scale_proxy_value": int(getattr(estimator, "support_vectors_").shape[0]),
            "scale_proxy_details": json.dumps(
                {
                    "n_support_by_class": np.asarray(n_support).astype(int).tolist()
                    if n_support is not None
                    else None,
                    "n_features": int(getattr(estimator, "support_vectors_").shape[1]),
                },
                sort_keys=True,
            ),
        }

    booster = getattr(estimator, "booster_", None)
    if booster is not None:
        details: Dict[str, Any] = {"num_trees": int(booster.num_trees())}
        try:
            dumped = booster.dump_model()
            tree_infos = dumped.get("tree_info", [])
            details["num_leaves_total"] = int(
                sum(int(tree.get("num_leaves", 0)) for tree in tree_infos)
            )
        except Exception:
            pass
        return {
            "scale_proxy_name": "trees",
            "scale_proxy_value": int(details["num_trees"]),
            "scale_proxy_details": json.dumps(details, sort_keys=True),
        }

    return {}


def write_benchmark_record(path: Path | str, record: Mapping[str, Any]) -> None:
    """Write one fold/model benchmark record as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(record), handle, indent=2, sort_keys=True)


def load_benchmark_records(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Load benchmark JSON files into a dataframe."""
    rows: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["benchmark_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_strategy_benchmark_records(strategy_dir: Path | str) -> pd.DataFrame:
    """Collect all fold-level benchmark JSON files under one CV strategy dir."""
    root = Path(strategy_dir)
    paths = sorted(root.glob("*/model_benchmark.json"))
    paths.extend(sorted(root.glob("*/baselines/*/model_benchmark.json")))
    return load_benchmark_records(paths)


def summarize_benchmark_records(
    records: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate fold-level benchmark records into comparable per-model rows."""
    if records.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    numeric_records = records.copy()
    numeric_columns = [
        "fit_seconds",
        "train_windows",
        "train_window_epochs",
        "inference_total_seconds",
        "inference_total_item_predictions",
        "encoder_head_inference_total_seconds",
        "encoder_head_inference_total_item_predictions",
        "trainable_parameters",
        "total_parameters",
        "accuracy",
        "macro_f1",
    ]
    for column in numeric_columns:
        if column in numeric_records.columns:
            numeric_records[column] = pd.to_numeric(numeric_records[column], errors="coerce")

    for key, group in numeric_records.groupby(list(group_columns), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key)}
        row["n_folds"] = int(group["fold_id"].nunique()) if "fold_id" in group.columns else int(len(group))

        for column in ["trainable_parameters", "total_parameters"]:
            if column in group.columns:
                values = group[column].dropna()
                row[column] = int(values.iloc[0]) if not values.empty else np.nan

        fit_seconds = group["fit_seconds"].sum(min_count=1) if "fit_seconds" in group.columns else np.nan
        train_windows = group["train_windows"].sum(min_count=1) if "train_windows" in group.columns else np.nan
        train_window_epochs = (
            group["train_window_epochs"].sum(min_count=1)
            if "train_window_epochs" in group.columns
            else np.nan
        )
        row["fit_seconds_total"] = float(fit_seconds) if pd.notna(fit_seconds) else np.nan
        row["train_windows_total"] = int(train_windows) if pd.notna(train_windows) else np.nan
        row["train_ms_per_window"] = (
            float(fit_seconds / train_windows * 1000.0)
            if pd.notna(fit_seconds) and pd.notna(train_windows) and train_windows > 0
            else np.nan
        )
        row["train_ms_per_window_epoch"] = (
            float(fit_seconds / train_window_epochs * 1000.0)
            if pd.notna(fit_seconds)
            and pd.notna(train_window_epochs)
            and train_window_epochs > 0
            else np.nan
        )

        inference_seconds = (
            group["inference_total_seconds"].sum(min_count=1)
            if "inference_total_seconds" in group.columns
            else np.nan
        )
        inference_items = (
            group["inference_total_item_predictions"].sum(min_count=1)
            if "inference_total_item_predictions" in group.columns
            else np.nan
        )
        row["inference_ms_per_window"] = (
            float(inference_seconds / inference_items * 1000.0)
            if pd.notna(inference_seconds) and pd.notna(inference_items) and inference_items > 0
            else np.nan
        )
        encoder_head_seconds = (
            group["encoder_head_inference_total_seconds"].sum(min_count=1)
            if "encoder_head_inference_total_seconds" in group.columns
            else np.nan
        )
        encoder_head_items = (
            group["encoder_head_inference_total_item_predictions"].sum(min_count=1)
            if "encoder_head_inference_total_item_predictions" in group.columns
            else np.nan
        )
        row["encoder_head_inference_ms_per_window"] = (
            float(encoder_head_seconds / encoder_head_items * 1000.0)
            if pd.notna(encoder_head_seconds)
            and pd.notna(encoder_head_items)
            and encoder_head_items > 0
            else np.nan
        )

        for metric in ["accuracy", "macro_f1"]:
            if metric in group.columns:
                row[metric] = float(group[metric].mean())

        for column in ["inference_scope", "encoder_head_inference_scope", "parameter_scope", "scale_proxy_name"]:
            if column in group.columns:
                values = group[column].dropna().astype(str).unique().tolist()
                row[column] = "; ".join(values)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(list(group_columns)).reset_index(drop=True)


def save_strategy_benchmark_outputs(strategy_dir: Path | str) -> List[Path]:
    """Save raw and summary benchmark CSVs under one strategy directory."""
    root = Path(strategy_dir)
    records = collect_strategy_benchmark_records(root)
    if records.empty:
        return []

    saved: List[Path] = []
    raw_path = root / "model_benchmark_raw.csv"
    records.to_csv(raw_path, index=False)
    saved.append(raw_path)

    group_columns = [
        column
        for column in ["task_name", "cv_strategy", "model", "model_display_name"]
        if column in records.columns
    ]
    summary = summarize_benchmark_records(records, group_columns=group_columns)
    if not summary.empty:
        summary_path = root / "model_benchmark_summary.csv"
        summary.to_csv(summary_path, index=False)
        saved.append(summary_path)
    return saved
