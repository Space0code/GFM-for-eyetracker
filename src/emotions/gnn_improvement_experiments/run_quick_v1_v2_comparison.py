"""Run a quick Table-6 arousal comparison for GNN v1, GNN v2, and baselines.

This script generates focused suite-wrapper configs and optionally runs them
sequentially on a small MAHNOB-HCI subset. By default it compares frozen
`GNN_v1`, current `GNN_v2`, and `LightGBM` on the Table-6 three-class arousal
task with proper k-fold splitting.

Example:
  python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py

Useful options:
  python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py \
      --models GNN_v1,GNN_v2,LightGBM,SVM \
      --n-splits 3 \
      --num-epochs 5 \
      --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.suite.config_merge import merge_many
from emotions.suite.run_hci_experiment_suite import run_suite


AROUSAL_EXPERIMENT_ID = "multiclass_table6_arousal_3class"
DEFAULT_MODELS = ["GNN_v1", "GNN_v2", "LightGBM"]
DEFAULT_SUBJECTS = ["P1", "P8", "P5", "P4", "P28", "P2", "P27"]
DEFAULT_RECORDINGS = [
    "69.avi",
    "55.avi",
    "58.avi",
    "earworm_f.avi",
    "53.avi",
    "80.avi",
    "79.avi",
    "73.avi",
    "107.avi",
    "146.avi",
    "30.avi",
    "138.avi",
    "detroit_f.avi",
    "dallas_f.avi",
]
BASELINE_MODELS = {"Mean", "SVM", "LightGBM", "MLP"}


@dataclass(frozen=True)
class QuickVariant:
    """One quick-comparison model variant."""

    model_name: str
    description: str
    overrides: Dict[str, Any]
    summary_model_name: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run quick GNN v1/v2 comparison")
    parser.add_argument(
        "--base-config",
        type=str,
        default="src/emotions/suite/configs/run_hci_experiment_suite_table6_3class.yaml",
        help="Base suite wrapper config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/quick_v1_v2_comparison",
        help="Output directory for generated configs and summary CSVs.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated models: GNN_v1,GNN_v2,Mean,SVM,LightGBM,MLP.",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Number of k-fold splits.")
    parser.add_argument("--val-size", type=int, default=1, help="Validation groups per fold.")
    parser.add_argument("--num-epochs", type=int, default=5, help="GNN epochs for quick runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="recording_kfold",
        choices=["recording_kfold", "subject_kfold"],
        help="Cross-validation strategy for the quick comparison.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only.")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary at {path}.")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write a YAML dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _parse_models(raw_models: str) -> List[str]:
    """Parse and validate requested model names."""
    models = [token.strip() for token in raw_models.split(",") if token.strip()]
    allowed = {"GNN_v1", "GNN_v2"} | BASELINE_MODELS
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Allowed: {sorted(allowed)}")
    if not models:
        raise ValueError("At least one model must be requested.")
    return models


def _enable_only_arousal(wrapper_cfg: Dict[str, Any]) -> None:
    """Enable only the Table-6 arousal experiment in a suite wrapper config."""
    experiments = wrapper_cfg.get("experiments")
    if isinstance(experiments, dict):
        for experiment_id, experiment_cfg in experiments.items():
            if isinstance(experiment_cfg, dict):
                experiment_cfg["enabled"] = experiment_id == AROUSAL_EXPERIMENT_ID
        return
    if isinstance(experiments, list):
        for experiment_cfg in experiments:
            if isinstance(experiment_cfg, dict):
                experiment_cfg["enabled"] = str(experiment_cfg.get("id", "")) == AROUSAL_EXPERIMENT_ID
        return
    raise ValueError("Unsupported experiments format; expected dict or list.")


def build_fixed_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build fixed quick-run overrides shared by all variants."""
    return {
        "suite": {
            "seed": int(args.seed),
            "results_dir": str(Path(args.output_root) / "suite_runs"),
        },
        "global_overrides": {
            "cross_validation": {
                "strategies": [str(args.cv_strategy)],
                "n_splits": int(args.n_splits),
                "val_size": int(args.val_size),
                "random_state": int(args.seed),
            },
            "dataset": {
                "use_cache": True,
                "kt": 2,
                "ks": 2,
                "use_edge_weights": True,
                "filter_subjects": DEFAULT_SUBJECTS,
                "filter_recordings": DEFAULT_RECORDINGS,
                "target_aggregation": "mean",
            },
            "gnn": {
                "model": {
                    "conv_type": "GCNConv",
                    "num_layers": 2,
                },
                "training": {
                    "num_epochs": int(args.num_epochs),
                    "num_workers": 0,
                    "persistent_workers": False,
                    "use_torch_compile": False,
                    "early_stopping_enabled": False,
                },
            },
        },
    }


def build_variant(model_name: str) -> QuickVariant:
    """Build one variant override block."""
    if model_name == "GNN_v1":
        return QuickVariant(
            model_name=model_name,
            description="Frozen v1 GNN with old handcrafted edge weights enabled.",
            summary_model_name="GNN",
            overrides={
                "global_overrides": {
                    "run_experiments": {"baselines": False, "gnn": True},
                    "dataset": {
                        "graph_version": "v1",
                        "edge_weight_mode": "handcrafted",
                        "use_edge_weights": True,
                    },
                    "gnn": {
                        "model": {
                            "model_version": "v1",
                            "pooling": "mean_max",
                        }
                    },
                }
            },
        )
    if model_name == "GNN_v2":
        return QuickVariant(
            model_name=model_name,
            description="Current v2 GNN with split temporal edges, attention pooling, and learned signed weights.",
            summary_model_name="GNN",
            overrides={
                "global_overrides": {
                    "run_experiments": {"baselines": False, "gnn": True},
                    "dataset": {
                        "graph_version": "v2",
                        "edge_weight_mode": "learned_signed",
                        "use_edge_weights": True,
                    },
                    "gnn": {
                        "model": {
                            "model_version": "v2",
                            "pooling": "attention",
                            "edge_weight_mode": "learned_signed",
                        }
                    },
                }
            },
        )
    if model_name in BASELINE_MODELS:
        return QuickVariant(
            model_name=model_name,
            description=f"{model_name} baseline on the same quick subset and folds.",
            summary_model_name=model_name,
            overrides={
                "global_overrides": {
                    "run_experiments": {"baselines": True, "gnn": False},
                    "baselines": {"models": [model_name]},
                }
            },
        )
    raise ValueError(f"Unsupported model_name={model_name}")


def _build_payload(base_cfg: Dict[str, Any], fixed_overrides: Dict[str, Any], variant: QuickVariant) -> Dict[str, Any]:
    """Merge base, fixed, and variant-specific overrides."""
    payload = merge_many(base_cfg, fixed_overrides, variant.overrides)
    _enable_only_arousal(payload)
    return payload


def _collect_metrics(suite_run_dir: Path, cv_strategy: str, summary_model_name: str) -> Dict[str, float]:
    """Collect aggregate metrics from one suite run."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Suite registry not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    row = registry[
        (registry["experiment_id"] == AROUSAL_EXPERIMENT_ID)
        & (registry["status"] == "success")
    ]
    if row.empty:
        raise ValueError(f"No successful {AROUSAL_EXPERIMENT_ID} run found in {registry_path}.")

    trainer_run_dir = Path(str(row.iloc[-1]["trainer_run_dir"]))
    summary_path = trainer_run_dir / cv_strategy / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Trainer summary not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    model_row = summary[
        (summary["model"] == summary_model_name)
        & (summary["metric_type"] == "aggregated")
    ]
    if model_row.empty:
        raise ValueError(f"No aggregated row for model={summary_model_name} in {summary_path}.")

    metrics: Dict[str, float] = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "loss"]:
        if metric in model_row.columns:
            metrics[metric] = float(model_row.iloc[0][metric])
    return metrics


def _rows_to_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Build a stable summary dataframe from result rows."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    sort_cols = [col for col in ["status", "balanced_accuracy", "macro_f1", "model"] if col in df.columns]
    ascending = [True] + [False] * (len(sort_cols) - 1)
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def run_quick_comparison(args: argparse.Namespace) -> Path:
    """Run or generate the quick comparison configs and summary."""
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_root) / timestamp
    generated_dir = output_dir / "generated_wrapper_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(base_config_path)
    fixed_overrides = build_fixed_overrides(args)
    model_names = _parse_models(args.models)
    variants = [build_variant(model_name) for model_name in model_names]

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        payload = _build_payload(base_cfg=base_cfg, fixed_overrides=fixed_overrides, variant=variant)
        config_path = generated_dir / f"{variant.model_name}.yaml"
        _write_yaml(config_path, payload)

        row: Dict[str, Any] = {
            "model": variant.model_name,
            "description": variant.description,
            "status": "dry_run" if args.dry_run else "pending",
            "wrapper_config_path": str(config_path),
            "suite_run_dir": "",
            "runtime_seconds": np.nan,
            "error": "",
        }
        if args.dry_run:
            rows.append(row)
            print(f"dry-run | {variant.model_name} | {config_path}")
            continue

        started = time.time()
        try:
            suite_run_dir = Path(run_suite(str(config_path)))
            row.update(_collect_metrics(suite_run_dir, args.cv_strategy, variant.summary_model_name))
            row["suite_run_dir"] = str(suite_run_dir)
            row["runtime_seconds"] = round(time.time() - started, 3)
            row["status"] = "success"
        except Exception as exc:
            row["runtime_seconds"] = round(time.time() - started, 3)
            row["status"] = "failed"
            row["error"] = f"{exc}\n{traceback.format_exc()}"

        rows.append(row)
        print(f"{row['status']} | {variant.model_name} | balanced_accuracy={row.get('balanced_accuracy', np.nan)}")

    summary = _rows_to_dataframe(rows)
    summary_path = output_dir / "quick_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved quick comparison summary: {summary_path}")
    return output_dir


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    print("Quick v1/v2 comparison arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    run_quick_comparison(args)


if __name__ == "__main__":
    main()
