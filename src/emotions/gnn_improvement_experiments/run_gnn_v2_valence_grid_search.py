"""Run the overnight GNN v2 grid search for Table-6 valence.

This script runs a sequential two-phase search for the 3-class Table-6
valence task using only GNN v2 and subject k-fold validation. It is intended
for overnight diploma experiments where each configuration is trained with a
large epoch budget and validation-loss early stopping.

Usage:
  python src/emotions/gnn_improvement_experiments/run_gnn_v2_valence_grid_search.py

Useful options:
  --dry-run
  --only-phase phase1
  --only-variant layers2_hidden64
  --num-epochs 1
  --output-root results/gnn_v2_valence_grid_search

Each run mirrors console output to:
  <output-root>/<timestamp>/gnn_v2_valence_grid_search.log
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, TextIO

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.gnn_improvement_experiments.run_quick_v1_v2_comparison import (
    VALENCE_EXPERIMENT_ID,
    _collect_metrics,
    _plot_test_loss_summary,
    _save_training_history_outputs,
)
from emotions.suite.config_merge import merge_many
from emotions.suite.run_hci_experiment_suite import run_suite
from emotions.utils import TimestampedLineWriter


PHASE1_NUM_LAYERS = [2, 3, 4]
PHASE1_HIDDEN_CHANNELS = [64, 128, 256]
PHASE2_KT_VALUES = [1, 2, 3]
PHASE2_KS_VALUES = [1, 2, 3]
PHASE2_FIXATION_DILATION_K_VALUES = [1, 2, 3, 5]


@dataclass(frozen=True)
class VariantSpec:
    """One concrete GNN v2 grid-search variant."""

    phase: str
    variant_id: str
    description: str
    num_layers: int
    hidden_channels: int
    kt: int
    ks: int
    fixation_dilation_k: int
    overrides: Dict[str, Any]


class _TeeStream:
    """Write console output unchanged while timestamping a mirrored log file."""

    def __init__(self, stream: TextIO, log_handle: TextIO) -> None:
        self.stream = stream
        self.log_handle = log_handle
        self.log_writer = TimestampedLineWriter(log_handle)

    def write(self, message: str) -> int:
        self.stream.write(message)
        self.log_writer.write(message)
        self.log_handle.flush()
        return len(message)

    def flush(self) -> None:
        self.stream.flush()
        self.log_handle.flush()


@contextlib.contextmanager
def _tee_output(log_path: Path) -> Iterator[None]:
    """Mirror stdout and stderr to one command-level timestamped log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as log_handle:
        sys.stdout = _TeeStream(original_stdout, log_handle)
        sys.stderr = _TeeStream(original_stderr, log_handle)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run GNN v2 Table-6 valence grid search")
    parser.add_argument(
        "--base-config",
        type=str,
        default=(
            "src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/"
            "run_hci_experiment_suite_table6_3class.yaml"
        ),
        help="Base quick Table-6 wrapper YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/gnn_v2_valence_grid_search",
        help="Output directory for generated configs, model runs, and summaries.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Subject k-fold split count.")
    parser.add_argument("--val-size", type=int, default=1, help="Validation subject count per fold.")
    parser.add_argument("--num-epochs", type=int, default=500, help="Maximum GNN epoch budget.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="GNN learning rate.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Validation-loss early stopping patience.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="GNN batch size.")
    parser.add_argument(
        "--oom-retry-batch-sizes",
        type=str,
        default="128,64",
        help="Comma-separated smaller batch sizes to try when a variant hits CUDA OOM.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Fast DataLoader worker count.")
    parser.add_argument(
        "--stable-num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for retry after loader failures.",
    )
    parser.add_argument(
        "--only-phase",
        type=str,
        default="all",
        choices=["all", "phase1", "phase2"],
        help="Run all phases or a single phase.",
    )
    parser.add_argument("--only-variant", type=str, default=None, help="Run only one variant id.")
    parser.add_argument(
        "--phase2-num-layers",
        type=int,
        default=None,
        help="Manual architecture for phase2-only runs.",
    )
    parser.add_argument(
        "--phase2-hidden-channels",
        type=int,
        default=None,
        help="Manual architecture for phase2-only runs.",
    )
    parser.add_argument(
        "--max-aggregate-history-variants",
        type=int,
        default=12,
        help="Skip large aggregate training-history plots above this successful-variant count.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write configs only; do not train.")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write a YAML mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _enable_valence_only(wrapper_cfg: Dict[str, Any]) -> None:
    """Enable only the Table-6 valence experiment in the wrapper config."""
    wrapper_cfg["quick_comparison"] = {"table6_tasks": ["valence"]}
    experiments = wrapper_cfg.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError("Expected wrapper config experiments to be a dictionary.")
    for experiment_id, experiment_cfg in experiments.items():
        if isinstance(experiment_cfg, dict):
            experiment_cfg["enabled"] = experiment_id == VALENCE_EXPERIMENT_ID


def _loader_overrides(*, num_workers: int, pin_memory: bool, persistent_workers: bool) -> Dict[str, Any]:
    """Build DataLoader-related GNN training overrides."""
    return {
        "global_overrides": {
            "gnn": {
                "training": {
                    "num_workers": int(num_workers),
                    "pin_memory": bool(pin_memory),
                    "persistent_workers": bool(persistent_workers),
                }
            }
        }
    }


def _training_overrides(**training_values: Any) -> Dict[str, Any]:
    """Build focused GNN training overrides for retry attempts."""
    return {"global_overrides": {"gnn": {"training": dict(training_values)}}}


def _fixed_overrides(args: argparse.Namespace, run_output_dir: Path) -> Dict[str, Any]:
    """Build fixed run-wide overrides shared by all variants."""
    fast_num_workers = int(args.num_workers)
    return {
        "suite": {
            "results_dir": str(run_output_dir / "model_runs"),
            "seed": int(args.seed),
        },
        "global_overrides": {
            "cross_validation": {
                "strategies": ["subject_kfold"],
                "n_splits": int(args.n_splits),
                "val_size": int(args.val_size),
                "random_state": int(args.seed),
            },
            "run_experiments": {"baselines": False, "gnn": True},
            "dataset": {
                "use_cache": True,
                "graph_version": "v2",
                "use_edge_weights": True,
                "edge_weight_mode": "learned_signed",
                "use_relative_time": True,
                "use_distance_avg": True,
                "use_fixation_duration": True,
                "use_delta_distance_edge_feature": True,
                "use_fixation_edges": True,
                "standardize_edge_features": True,
            },
            "gnn": {
                "model": {
                    "model_version": "v2",
                    "conv_type": "GCNConv",
                    "edge_weight_mode": "learned_signed",
                    "head_pooling": "attention",
                    "graph_pooling": "attention",
                    "relation_pooling": "mlp",
                    "pooling": "mean_max",
                },
                "training": {
                    "num_epochs": int(args.num_epochs),
                    "batch_size": int(args.batch_size),
                    "learning_rate": float(args.learning_rate),
                    "early_stopping_enabled": True,
                    "early_stopping_patience": int(args.early_stopping_patience),
                    "early_stopping_min_delta": 0.0,
                    "early_stopping_restore_best": True,
                    "use_torch_compile": False,
                    "num_workers": fast_num_workers,
                    "pin_memory": True,
                    "persistent_workers": fast_num_workers > 0,
                },
            },
        },
    }


def _variant_overrides(
    *,
    num_layers: int,
    hidden_channels: int,
    kt: int,
    ks: int,
    fixation_dilation_k: int,
) -> Dict[str, Any]:
    """Build overrides for one concrete GNN v2 configuration."""
    return {
        "global_overrides": {
            "dataset": {
                "kt": int(kt),
                "ks": int(ks),
                "fixation_dilation_k": int(fixation_dilation_k),
            },
            "gnn": {
                "model": {
                    "num_layers": int(num_layers),
                    "hidden_channels": int(hidden_channels),
                }
            },
        }
    }


def _phase1_variants() -> List[VariantSpec]:
    """Return phase 1 architecture/capacity variants."""
    variants: List[VariantSpec] = []
    for num_layers in PHASE1_NUM_LAYERS:
        for hidden_channels in PHASE1_HIDDEN_CHANNELS:
            variant_id = f"layers{num_layers}_hidden{hidden_channels}"
            variants.append(
                VariantSpec(
                    phase="phase1",
                    variant_id=variant_id,
                    description=(
                        f"GNN v2 architecture search: {num_layers} layers, "
                        f"{hidden_channels} hidden channels."
                    ),
                    num_layers=num_layers,
                    hidden_channels=hidden_channels,
                    kt=2,
                    ks=2,
                    fixation_dilation_k=3,
                    overrides=_variant_overrides(
                        num_layers=num_layers,
                        hidden_channels=hidden_channels,
                        kt=2,
                        ks=2,
                        fixation_dilation_k=3,
                    ),
                )
            )
    return variants


def _phase2_variants(*, num_layers: int, hidden_channels: int) -> List[VariantSpec]:
    """Return phase 2 graph-density variants for the selected architecture."""
    variants: List[VariantSpec] = []
    for kt in PHASE2_KT_VALUES:
        for ks in PHASE2_KS_VALUES:
            for fixation_dilation_k in PHASE2_FIXATION_DILATION_K_VALUES:
                variant_id = f"kt{kt}_ks{ks}_kf{fixation_dilation_k}"
                variants.append(
                    VariantSpec(
                        phase="phase2",
                        variant_id=variant_id,
                        description=(
                            f"GNN v2 graph-density search: kt={kt}, ks={ks}, "
                            f"fixation_dilation_k={fixation_dilation_k}."
                        ),
                        num_layers=num_layers,
                        hidden_channels=hidden_channels,
                        kt=kt,
                        ks=ks,
                        fixation_dilation_k=fixation_dilation_k,
                        overrides=_variant_overrides(
                            num_layers=num_layers,
                            hidden_channels=hidden_channels,
                            kt=kt,
                            ks=ks,
                            fixation_dilation_k=fixation_dilation_k,
                        ),
                    )
                )
    return variants


def _phase2_variant_count() -> int:
    """Return the full phase 2 grid size without needing a chosen architecture."""
    return len(PHASE2_KT_VALUES) * len(PHASE2_KS_VALUES) * len(PHASE2_FIXATION_DILATION_K_VALUES)


def _filter_variants(variants: Sequence[VariantSpec], only_variant: str | None) -> List[VariantSpec]:
    """Filter variants by optional variant id."""
    if only_variant is None:
        return list(variants)
    return [variant for variant in variants if variant.variant_id == only_variant]


def _variant_record(
    variant: VariantSpec,
    *,
    status: str,
    loader_mode: str,
    config_path: Path,
    suite_run_dir: Path | None,
    trainer_run_dir: Path | None,
    runtime_seconds: float | None,
    error: str = "",
) -> Dict[str, Any]:
    """Build one stable summary record."""
    row: Dict[str, Any] = {
        "phase": variant.phase,
        "variant_id": variant.variant_id,
        "description": variant.description,
        "num_layers": int(variant.num_layers),
        "hidden_channels": int(variant.hidden_channels),
        "kt": int(variant.kt),
        "ks": int(variant.ks),
        "fixation_dilation_k": int(variant.fixation_dilation_k),
        "experiment_id": VALENCE_EXPERIMENT_ID,
        "experiment_display_name": "Table-6 Valence",
        "cv_strategy": "subject_kfold",
        "model": variant.variant_id,
        "summary_model_name": "GNN",
        "status": status,
        "loader_mode": loader_mode,
        "runtime_seconds": round(float(runtime_seconds), 3) if runtime_seconds is not None else np.nan,
        "wrapper_config_path": str(config_path),
        "suite_run_dir": str(suite_run_dir) if suite_run_dir is not None else "",
        "trainer_run_dir": str(trainer_run_dir) if trainer_run_dir is not None else "",
        "error": error,
    }
    return row


def _build_payload(
    *,
    base_cfg: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    variant: VariantSpec,
    loader_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Merge base wrapper, fixed overrides, variant overrides, and loader overrides."""
    payload = merge_many(base_cfg, fixed_overrides, variant.overrides, loader_overrides or {})
    _enable_valence_only(payload)
    return payload


def _trainer_run_dir(suite_run_dir: Path) -> Path:
    """Resolve the successful valence trainer directory for one suite run."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Suite registry not found: {registry_path}")

    registry = pd.read_csv(registry_path)
    valence_rows = registry[registry["experiment_id"] == VALENCE_EXPERIMENT_ID]
    if valence_rows.empty:
        raise ValueError(f"No {VALENCE_EXPERIMENT_ID} row in {registry_path}")

    failed = valence_rows[valence_rows["status"] != "success"]
    if not failed.empty:
        error_text = str(failed.iloc[-1].get("error", ""))
        raise RuntimeError(f"Valence suite experiment failed: {error_text}")

    row = valence_rows[valence_rows["status"] == "success"].iloc[-1]
    return Path(str(row["trainer_run_dir"]))


def _is_loader_failure(error_text: str) -> bool:
    """Return true when an error is likely DataLoader multiprocessing related."""
    lowered = error_text.lower()
    indicators = [
        "pin memory",
        "pin-memory",
        "dataloader",
        "data loader",
        "persistent_workers",
        "worker exited",
        "worker process",
        "ancdata",
        "multiprocessing",
        "_pin_memory_loop",
        "broken pipe",
        "bus error",
        "connection reset",
        "file descriptor",
        "killed by signal",
        "shared memory",
        "shm",
        "too many open files",
        "cuda error: initialization error",
    ]
    return any(indicator in lowered for indicator in indicators)


def _is_cuda_oom(error_text: str) -> bool:
    """Return true when an error is a CUDA out-of-memory failure."""
    lowered = error_text.lower()
    return "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered


def _parse_oom_retry_batch_sizes(args: argparse.Namespace) -> List[int]:
    """Parse retry batch sizes smaller than the primary batch size."""
    values: List[int] = []
    for raw_value in str(args.oom_retry_batch_sizes).split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        value = int(raw_value)
        if value <= 0:
            raise ValueError("--oom-retry-batch-sizes must contain positive integers.")
        if value < int(args.batch_size) and value not in values:
            values.append(value)
    return values


def _release_cuda_cache() -> None:
    """Best-effort cleanup between failed CUDA attempts in the same process."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _suite_run_dirs(results_dir: Path) -> set[Path]:
    """Return currently existing suite run directories."""
    if not results_dir.exists():
        return set()
    return {path.resolve() for path in results_dir.iterdir() if path.is_dir()}


def _new_suite_run_dir(results_dir: Path, before: set[Path]) -> Path | None:
    """Return the newest suite run directory created after a failed run_suite call."""
    candidates = [path for path in _suite_run_dirs(results_dir) if path not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_variant_once(
    *,
    base_cfg: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    variant: VariantSpec,
    generated_dir: Path,
    loader_mode: str,
    loader_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run one variant once and return its summary row."""
    suffix = "" if loader_mode == "fast" else f"_{loader_mode}"
    config_path = generated_dir / f"{variant.phase}_{variant.variant_id}{suffix}.yaml"
    attempt_results_dir = generated_dir.parent / "model_runs" / f"{variant.phase}_{variant.variant_id}_{loader_mode}"
    attempt_overrides = {"suite": {"results_dir": str(attempt_results_dir)}}
    payload = _build_payload(
        base_cfg=base_cfg,
        fixed_overrides=fixed_overrides,
        variant=variant,
        loader_overrides=merge_many(attempt_overrides, loader_overrides or {}),
    )
    _write_yaml(config_path, payload)

    started = time.time()
    suite_run_dir: Path | None = None
    suite_dirs_before = _suite_run_dirs(attempt_results_dir)
    try:
        suite_run_dir = Path(run_suite(str(config_path)))
        trainer_run_dir = _trainer_run_dir(suite_run_dir)
        row = _variant_record(
            variant,
            status="success",
            loader_mode=loader_mode,
            config_path=config_path,
            suite_run_dir=suite_run_dir,
            trainer_run_dir=trainer_run_dir,
            runtime_seconds=time.time() - started,
        )
        row.update(
            _collect_metrics(
                suite_run_dir=suite_run_dir,
                cv_strategy="subject_kfold",
                summary_model_name="GNN",
                experiment_id=VALENCE_EXPERIMENT_ID,
            )
        )
        return row
    except Exception as exc:
        error_text = f"{exc}\n{traceback.format_exc()}"
        if suite_run_dir is None:
            suite_run_dir = _new_suite_run_dir(attempt_results_dir, suite_dirs_before)
        if suite_run_dir is not None:
            try:
                trainer_run_dir = _trainer_run_dir(suite_run_dir)
                row = _variant_record(
                    variant,
                    status="success",
                    loader_mode=loader_mode,
                    config_path=config_path,
                    suite_run_dir=suite_run_dir,
                    trainer_run_dir=trainer_run_dir,
                    runtime_seconds=time.time() - started,
                    error=f"Recovered metrics after run_suite raised:\n{error_text}",
                )
                row.update(
                    _collect_metrics(
                        suite_run_dir=suite_run_dir,
                        cv_strategy="subject_kfold",
                        summary_model_name="GNN",
                        experiment_id=VALENCE_EXPERIMENT_ID,
                    )
                )
                return row
            except Exception:
                pass
        return _variant_record(
            variant,
            status="failed",
            loader_mode=loader_mode,
            config_path=config_path,
            suite_run_dir=suite_run_dir,
            trainer_run_dir=None,
            runtime_seconds=time.time() - started,
            error=error_text,
        )


def _run_variant(
    *,
    args: argparse.Namespace,
    base_cfg: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    variant: VariantSpec,
    generated_dir: Path,
) -> Dict[str, Any]:
    """Run one variant, retrying loader failures with stable settings."""
    print("\n" + "=" * 100)
    print(f"Running {variant.phase} | {variant.variant_id}: {variant.description}")
    print(
        "Config: "
        f"layers={variant.num_layers}, hidden={variant.hidden_channels}, "
        f"kt={variant.kt}, ks={variant.ks}, kf={variant.fixation_dilation_k}"
    )

    fast_row = _run_variant_once(
        base_cfg=base_cfg,
        fixed_overrides=fixed_overrides,
        variant=variant,
        generated_dir=generated_dir,
        loader_mode="fast",
    )
    if fast_row["status"] == "success":
        print(f"Completed {variant.variant_id} with fast loader")
        return fast_row

    if _is_cuda_oom(str(fast_row.get("error", ""))):
        last_row = fast_row
        for batch_size in _parse_oom_retry_batch_sizes(args):
            _release_cuda_cache()
            loader_mode = f"oom_batch{batch_size}"
            print(f"Retrying {variant.variant_id} after CUDA OOM with batch_size={batch_size}")
            retry_overrides = merge_many(
                _training_overrides(batch_size=int(batch_size)),
                _loader_overrides(
                    num_workers=int(args.stable_num_workers),
                    pin_memory=False,
                    persistent_workers=False,
                ),
            )
            last_row = _run_variant_once(
                base_cfg=base_cfg,
                fixed_overrides=fixed_overrides,
                variant=variant,
                generated_dir=generated_dir,
                loader_mode=loader_mode,
                loader_overrides=retry_overrides,
            )
            if last_row["status"] == "success":
                print(f"Completed {variant.variant_id} with CUDA OOM retry batch_size={batch_size}")
                return last_row
            if not _is_cuda_oom(str(last_row.get("error", ""))):
                break
        print(f"FAILED {variant.variant_id} after CUDA OOM retries")
        return last_row

    if _is_loader_failure(str(fast_row.get("error", ""))):
        print(f"Retrying {variant.variant_id} with stable DataLoader settings after loader failure")
        _release_cuda_cache()
        stable_overrides = _loader_overrides(
            num_workers=int(args.stable_num_workers),
            pin_memory=False,
            persistent_workers=False,
        )
        stable_row = _run_variant_once(
            base_cfg=base_cfg,
            fixed_overrides=fixed_overrides,
            variant=variant,
            generated_dir=generated_dir,
            loader_mode="stable_retry",
            loader_overrides=stable_overrides,
        )
        if stable_row["status"] == "success":
            print(f"Completed {variant.variant_id} with stable retry")
        else:
            print(f"FAILED {variant.variant_id} after stable retry")
        return stable_row

    print(f"FAILED {variant.variant_id}")
    return fast_row


def _dry_run_row(variant: VariantSpec, config_path: Path) -> Dict[str, Any]:
    """Build one dry-run summary row."""
    return _variant_record(
        variant,
        status="dry_run",
        loader_mode="fast",
        config_path=config_path,
        suite_run_dir=None,
        trainer_run_dir=None,
        runtime_seconds=None,
    )


def _rows_to_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convert summary rows into a stable table."""
    columns = [
        "phase",
        "variant_id",
        "description",
        "num_layers",
        "hidden_channels",
        "kt",
        "ks",
        "fixation_dilation_k",
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
        "summary_model_name",
        "status",
        "loader_mode",
        "runtime_seconds",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "loss",
        "wrapper_config_path",
        "suite_run_dir",
        "trainer_run_dir",
        "error",
    ]
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan if column in {"accuracy", "macro_f1", "balanced_accuracy", "loss"} else ""
    return df[columns].sort_values(["phase", "variant_id"]).reset_index(drop=True)


def _successful_variant_count(rows: Sequence[Dict[str, Any]]) -> int:
    """Count successful variant rows."""
    return sum(1 for row in rows if row.get("status") == "success")


def _select_best(rows: Sequence[Dict[str, Any]], *, phase: str) -> Dict[str, Any] | None:
    """Select the best successful variant by accuracy, macro F1, then loss."""
    df = _rows_to_dataframe(row for row in rows if row.get("phase") == phase and row.get("status") == "success")
    if df.empty:
        return None

    ranking = df.copy()
    ranking["accuracy_rank"] = pd.to_numeric(ranking["accuracy"], errors="coerce").fillna(float("-inf"))
    ranking["macro_f1_rank"] = pd.to_numeric(ranking["macro_f1"], errors="coerce").fillna(float("-inf"))
    ranking["loss_rank"] = pd.to_numeric(ranking["loss"], errors="coerce").fillna(float("inf"))
    ranking = ranking.sort_values(
        ["accuracy_rank", "macro_f1_rank", "loss_rank", "variant_id"],
        ascending=[False, False, True, True],
    )
    return ranking.iloc[0].to_dict()


def _save_phase_summary(summary: pd.DataFrame, output_dir: Path, phase: str) -> None:
    """Save one phase-specific summary CSV."""
    phase_df = summary[summary["phase"] == phase].copy() if not summary.empty else pd.DataFrame()
    phase_df.to_csv(output_dir / f"{phase}_summary.csv", index=False)


def _save_best_yaml(best_row: Dict[str, Any], output_dir: Path, filename: str) -> None:
    """Save the selected best row as a YAML file."""
    serializable = {
        key: (None if pd.isna(value) else value)
        for key, value in best_row.items()
        if key not in {"error"}
    }
    _write_yaml(output_dir / filename, serializable)


def _save_grid_summaries(rows: Sequence[Dict[str, Any]], output_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Save final grid and per-phase summaries."""
    summary_df = _rows_to_dataframe(rows)
    summary_path = output_dir / "grid_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _save_phase_summary(summary_df, output_dir=output_dir, phase="phase1")
    _save_phase_summary(summary_df, output_dir=output_dir, phase="phase2")
    return summary_df, summary_path


def _save_optional_artifacts(
    *,
    rows: Sequence[Dict[str, Any]],
    summary_df: pd.DataFrame,
    output_dir: Path,
    max_aggregate_history_variants: int,
) -> List[str]:
    """Best-effort optional aggregate plots/tables after core summaries are safe."""
    warnings: List[str] = []
    success_count = _successful_variant_count(rows)
    if success_count <= int(max_aggregate_history_variants):
        try:
            for path in _save_training_history_outputs(rows=rows, output_dir=output_dir):
                print(f"Saved training output: {path}")
        except Exception as exc:
            warning = f"Training-history artifact generation failed: {exc}\n{traceback.format_exc()}"
            warnings.append(warning)
            print(f"WARNING: {warning}")
    else:
        warning = (
            "Skipped aggregate training-history plots because "
            f"{success_count} successful variants exceed --max-aggregate-history-variants="
            f"{max_aggregate_history_variants}. Per-fold histories remain in trainer directories."
        )
        warnings.append(warning)
        print(f"WARNING: {warning}")

    try:
        test_loss_path = _plot_test_loss_summary(summary=summary_df, output_dir=output_dir)
        if test_loss_path is not None:
            print(f"Saved test loss plot: {test_loss_path}")
    except Exception as exc:
        warning = f"Test-loss plot generation failed: {exc}\n{traceback.format_exc()}"
        warnings.append(warning)
        print(f"WARNING: {warning}")

    return warnings


def _write_phase2_plan(output_dir: Path) -> Path:
    """Write the dry-run phase 2 plan when phase 1 has no real winner yet."""
    planned = [
        {
            "phase": "phase2",
            "variant_id": f"kt{kt}_ks{ks}_kf{kf}",
            "kt": kt,
            "ks": ks,
            "fixation_dilation_k": kf,
            "requires_phase1_winner": True,
        }
        for kt in PHASE2_KT_VALUES
        for ks in PHASE2_KS_VALUES
        for kf in PHASE2_FIXATION_DILATION_K_VALUES
    ]
    path = output_dir / "phase2_planned_grid.csv"
    pd.DataFrame(planned).to_csv(path, index=False)
    return path


def _run_or_dry_variants(
    *,
    args: argparse.Namespace,
    base_cfg: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    variants: Sequence[VariantSpec],
    generated_dir: Path,
    rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run or dry-run variants and continuously save partial results."""
    for variant in variants:
        config_path = generated_dir / f"{variant.phase}_{variant.variant_id}.yaml"
        payload = _build_payload(base_cfg=base_cfg, fixed_overrides=fixed_overrides, variant=variant)
        _write_yaml(config_path, payload)

        if args.dry_run:
            rows.append(_dry_run_row(variant, config_path=config_path))
        else:
            rows.append(
                _run_variant(
                    args=args,
                    base_cfg=base_cfg,
                    fixed_overrides=fixed_overrides,
                    variant=variant,
                    generated_dir=generated_dir,
                )
            )

        partial_df = _rows_to_dataframe(rows)
        partial_df.to_csv(output_dir / "grid_summary_partial.csv", index=False)
        _save_phase_summary(partial_df, output_dir=output_dir, phase="phase1")
        _save_phase_summary(partial_df, output_dir=output_dir, phase="phase2")


def _print_run_header(args: argparse.Namespace, output_dir: Path, log_path: Path) -> None:
    """Print reproducibility and debugging context at the start of the run."""
    print(f"Logging GNN v2 valence grid-search output to: {log_path}")
    print("GNN v2 valence grid-search arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("Resolved run context:")
    print(f"  cwd: {Path.cwd()}")
    print(f"  python: {sys.executable}")
    print(f"  platform: {platform.platform()}")
    print(f"  command: {' '.join(sys.argv)}")
    print(f"  output_dir: {output_dir}")
    print(f"  phase1_grid_size: {len(_phase1_variants())}")
    print(f"  phase2_grid_size: {_phase2_variant_count()}")
    print("Winner metric order: accuracy desc, macro_f1 desc, loss asc")


def run_grid(args: argparse.Namespace, *, run_timestamp: str | None = None) -> Path:
    """Run the requested grid search and write summary artifacts."""
    base_config_path = Path(args.base_config)
    base_cfg = _load_yaml(base_config_path)

    timestamp = run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_root) / timestamp
    generated_dir = output_dir / "generated_wrapper_configs"
    log_path = output_dir / "gnn_v2_valence_grid_search.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    fixed_overrides = _fixed_overrides(args, run_output_dir=output_dir)
    rows: List[Dict[str, Any]] = []

    _print_run_header(args=args, output_dir=output_dir, log_path=log_path)
    run_warnings: List[str] = []

    phase1_variants = _filter_variants(_phase1_variants(), args.only_variant)
    run_phase1 = args.only_phase in {"all", "phase1"} and bool(phase1_variants)
    if run_phase1:
        print(f"Selected phase 1 variants: {len(phase1_variants)}")
        _run_or_dry_variants(
            args=args,
            base_cfg=base_cfg,
            fixed_overrides=fixed_overrides,
            variants=phase1_variants,
            generated_dir=generated_dir,
            rows=rows,
            output_dir=output_dir,
        )
    elif args.only_phase == "phase1":
        raise ValueError("No phase 1 variants selected.")

    best_phase1: Dict[str, Any] | None = None
    phase2_architecture: tuple[int, int] | None = None
    if run_phase1 and not args.dry_run:
        best_phase1 = _select_best(rows, phase="phase1")
        if best_phase1 is None:
            warning = "No successful phase 1 variants; skipping dependent phase 2."
            run_warnings.append(warning)
            print(f"WARNING: {warning}")
        else:
            _save_best_yaml(best_phase1, output_dir=output_dir, filename="best_phase1.yaml")
            phase2_architecture = (int(best_phase1["num_layers"]), int(best_phase1["hidden_channels"]))
            print(
                "Best phase 1: "
                f"{best_phase1['variant_id']} "
                f"(accuracy={best_phase1.get('accuracy')}, macro_f1={best_phase1.get('macro_f1')})"
            )
    elif args.dry_run and args.only_phase == "all" and args.only_variant is None:
        plan_path = _write_phase2_plan(output_dir)
        print(f"Dry-run phase 2 plan saved to: {plan_path}")

    if args.phase2_num_layers is not None or args.phase2_hidden_channels is not None:
        if args.phase2_num_layers is None or args.phase2_hidden_channels is None:
            raise ValueError("Set both --phase2-num-layers and --phase2-hidden-channels.")
        phase2_architecture = (int(args.phase2_num_layers), int(args.phase2_hidden_channels))

    run_phase2 = args.only_phase in {"all", "phase2"}
    if run_phase2:
        if phase2_architecture is None:
            if args.dry_run and args.only_phase == "all":
                phase2_variants: List[VariantSpec] = []
            elif args.only_phase == "all":
                phase2_variants = []
                warning = "Phase 2 skipped because no phase 1 architecture was available."
                run_warnings.append(warning)
                print(f"WARNING: {warning}")
            else:
                raise ValueError(
                    "Phase 2 needs a phase 1 winner or manual "
                    "--phase2-num-layers and --phase2-hidden-channels."
                )
        else:
            num_layers, hidden_channels = phase2_architecture
            phase2_variants = _filter_variants(
                _phase2_variants(num_layers=num_layers, hidden_channels=hidden_channels),
                args.only_variant,
            )
            if not phase2_variants and args.only_phase == "phase2":
                raise ValueError("No phase 2 variants selected.")
            if not phase2_variants and args.only_phase == "all" and args.only_variant is not None:
                warning = (
                    "No phase 2 variants matched --only-variant. "
                    "Use --only-phase phase2 with a phase2 variant id for phase2-only debugging."
                )
                run_warnings.append(warning)
                print(f"WARNING: {warning}")
            if phase2_variants:
                print(f"Selected phase 2 variants: {len(phase2_variants)}")
                _run_or_dry_variants(
                    args=args,
                    base_cfg=base_cfg,
                    fixed_overrides=fixed_overrides,
                    variants=phase2_variants,
                    generated_dir=generated_dir,
                    rows=rows,
                    output_dir=output_dir,
                )

                if not args.dry_run:
                    best_phase2 = _select_best(rows, phase="phase2")
                    if best_phase2 is None:
                        warning = "No successful phase 2 variants; best_phase2.yaml was not written."
                        run_warnings.append(warning)
                        print(f"WARNING: {warning}")
                    else:
                        _save_best_yaml(best_phase2, output_dir=output_dir, filename="best_phase2.yaml")
                        print(
                            "Best phase 2: "
                            f"{best_phase2['variant_id']} "
                            f"(accuracy={best_phase2.get('accuracy')}, macro_f1={best_phase2.get('macro_f1')})"
                        )

    summary_df, summary_path = _save_grid_summaries(rows=rows, output_dir=output_dir)
    print(f"Saved grid summary: {summary_path}")

    manifest = {
        "created_at": timestamp,
        "base_config": str(base_config_path),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "val_size": int(args.val_size),
        "num_epochs": int(args.num_epochs),
        "learning_rate": float(args.learning_rate),
        "early_stopping_patience": int(args.early_stopping_patience),
        "batch_size": int(args.batch_size),
        "oom_retry_batch_sizes": _parse_oom_retry_batch_sizes(args),
        "fast_num_workers": int(args.num_workers),
        "stable_num_workers": int(args.stable_num_workers),
        "only_phase": args.only_phase,
        "only_variant": args.only_variant,
        "phase2_num_layers": args.phase2_num_layers,
        "phase2_hidden_channels": args.phase2_hidden_channels,
        "dry_run": bool(args.dry_run),
        "summary_csv": str(summary_path),
        "winner_metric_order": ["accuracy_desc", "macro_f1_desc", "loss_asc"],
        "phase1_variant_count": len(phase1_variants) if run_phase1 else 0,
        "successful_row_count": _successful_variant_count(rows),
        "run_warnings": run_warnings,
    }
    _write_yaml(output_dir / "run_manifest.yaml", manifest)

    if not args.dry_run:
        artifact_warnings = _save_optional_artifacts(
            rows=rows,
            summary_df=summary_df,
            output_dir=output_dir,
            max_aggregate_history_variants=int(args.max_aggregate_history_variants),
        )
        if artifact_warnings:
            manifest["artifact_warnings"] = artifact_warnings
            manifest["run_warnings"] = run_warnings + artifact_warnings
            _write_yaml(output_dir / "run_manifest.yaml", manifest)
    return output_dir


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(args.output_root) / timestamp / "gnn_v2_valence_grid_search.log"
    with _tee_output(log_path):
        output_dir = run_grid(args, run_timestamp=timestamp)
        print(f"GNN v2 valence grid-search directory: {output_dir}")


if __name__ == "__main__":
    main()
