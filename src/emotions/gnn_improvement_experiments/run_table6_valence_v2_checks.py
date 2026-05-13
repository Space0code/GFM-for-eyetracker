"""Run focused Table-6 valence convergence and GNN v2 comparison checks.

This script runs a conservative, sequential experiment matrix for the current
quick v1/v2 Table-6 setup:

- convergence checks for LightGBM, MLP, GNN v1, and GNN v2;
- GNN v2 depth checks for 1, 3, 5, and 10 layers;
- GNN v2 GCN/GAT and edge-weight checks;
- GNN v2 epoch-budget checks with early stopping disabled.

Usage:
  python src/emotions/gnn_improvement_experiments/run_table6_valence_v2_checks.py

Useful options:
  --base-config src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml
  --output-root results/table6_valence_v2_checks
  --num-workers 0
  --lightgbm-n-jobs 4
  --only-family convergence
  --only-variant depth_3
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
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yaml

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


@dataclass(frozen=True)
class VariantSpec:
    """One concrete sequential experiment variant."""

    variant_id: str
    family: str
    description: str
    display_model: str
    summary_model_names: Sequence[str]
    overrides: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Table-6 valence GNN v2 checks")
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
        default="results/table6_valence_v2_checks",
        help="Output directory for generated configs, model runs, and summaries.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=3, help="Subject k-fold split count.")
    parser.add_argument("--val-size", type=int, default=1, help="Validation group count per fold.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count for GNN runs.")
    parser.add_argument(
        "--lightgbm-n-jobs",
        type=int,
        default=4,
        help="LightGBM CPU worker count. Use a small value to avoid CPU saturation.",
    )
    parser.add_argument(
        "--only-family",
        type=str,
        default=None,
        choices=["convergence", "depth", "architecture", "epochs"],
        help="Run only one family of checks.",
    )
    parser.add_argument(
        "--only-variant",
        type=str,
        default=None,
        help="Run only one variant id.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write configs only; do not train.")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write a YAML mapping to disk."""
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


def _base_overrides(args: argparse.Namespace, run_output_dir: Path) -> Dict[str, Any]:
    """Build run-wide conservative overrides."""
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
            "dataset": {
                "use_cache": True,
                "graph_version": "v2",
                "use_edge_weights": True,
                "edge_weight_mode": "learned_signed",
            },
            "baselines": {
                "hyperparameters": {
                    "LightGBM": {
                        "n_jobs": int(args.lightgbm_n_jobs),
                        "random_state": int(args.seed),
                    },
                    "MLP": {
                        "random_state": int(args.seed),
                    },
                }
            },
            "gnn": {
                "model": {
                    "model_version": "v2",
                    "conv_type": "GCNConv",
                    "num_layers": 10,
                    "head_pooling": "attention",
                    "graph_pooling": "attention",
                    "relation_pooling": "mlp",
                    "pooling": "mean_max",
                },
                "training": {
                    "num_workers": int(args.num_workers),
                    "pin_memory": False,
                    "persistent_workers": False,
                    "use_torch_compile": False,
                    "early_stopping_enabled": True,
                    "early_stopping_patience": 7,
                    "early_stopping_min_delta": 0.0,
                    "early_stopping_restore_best": True,
                },
            },
        },
    }


def _gnn_v2_overrides(
    *,
    num_layers: int = 10,
    conv_type: str = "GCNConv",
    use_edge_weights: bool = True,
    num_epochs: int | None = None,
    early_stopping_enabled: bool = True,
) -> Dict[str, Any]:
    """Build GNN v2 variant overrides."""
    overrides: Dict[str, Any] = {
        "global_overrides": {
            "run_experiments": {"baselines": False, "gnn": True},
            "dataset": {
                "graph_version": "v2",
                "use_edge_weights": bool(use_edge_weights),
                "edge_weight_mode": "learned_signed",
            },
            "gnn": {
                "model": {
                    "model_version": "v2",
                    "edge_weight_mode": "learned_signed",
                    "num_layers": int(num_layers),
                    "conv_type": conv_type,
                },
                "training": {
                    "early_stopping_enabled": bool(early_stopping_enabled),
                },
            },
        },
    }
    if num_epochs is not None:
        overrides["global_overrides"]["gnn"]["training"]["num_epochs"] = int(num_epochs)
    return overrides


def _build_variants() -> List[VariantSpec]:
    """Return the requested focused experiment matrix."""
    variants: List[VariantSpec] = [
        VariantSpec(
            variant_id="convergence_baselines",
            family="convergence",
            description="LightGBM and MLP convergence/baseline check with current non-GNN config.",
            display_model="Baselines",
            summary_model_names=["LightGBM", "MLP"],
            overrides={
                "global_overrides": {
                    "run_experiments": {"baselines": True, "gnn": False},
                    "baselines": {"models": ["LightGBM", "MLP"]},
                }
            },
        ),
        VariantSpec(
            variant_id="convergence_gnn_v1",
            family="convergence",
            description="GNN v1 with current shared quick-comparison settings.",
            display_model="GNN_v1",
            summary_model_names=["GNN"],
            overrides={
                "global_overrides": {
                    "run_experiments": {"baselines": False, "gnn": True},
                    "dataset": {
                        "graph_version": "v1",
                        "use_edge_weights": True,
                        "edge_weight_mode": "handcrafted",
                    },
                    "gnn": {
                        "model": {
                            "model_version": "v1",
                        }
                    },
                }
            },
        ),
        VariantSpec(
            variant_id="convergence_gnn_v2_current",
            family="convergence",
            description="GNN v2 current config: GCNConv, learned signed weights, 10 layers.",
            display_model="GNN_v2",
            summary_model_names=["GNN"],
            overrides=_gnn_v2_overrides(),
        ),
    ]

    for depth in [1, 3, 5, 10]:
        variants.append(
            VariantSpec(
                variant_id=f"depth_{depth}",
                family="depth",
                description=f"GNN v2 with {depth} message-passing layer(s).",
                display_model=f"depth_{depth}",
                summary_model_names=["GNN"],
                overrides=_gnn_v2_overrides(num_layers=depth),
            )
        )

    variants.extend(
        [
            VariantSpec(
                variant_id="arch_gcn_weighted",
                family="architecture",
                description="GNN v2, GCNConv, learned signed edge weights enabled.",
                display_model="GCN_weighted",
                summary_model_names=["GNN"],
                overrides=_gnn_v2_overrides(conv_type="GCNConv", use_edge_weights=True),
            ),
            VariantSpec(
                variant_id="arch_gcn_unweighted",
                family="architecture",
                description="GNN v2, GCNConv, model edge weights disabled.",
                display_model="GCN_unweighted",
                summary_model_names=["GNN"],
                overrides=_gnn_v2_overrides(conv_type="GCNConv", use_edge_weights=False),
            ),
            VariantSpec(
                variant_id="arch_gat_unweighted",
                family="architecture",
                description="GNN v2, GATConv, model edge weights disabled.",
                display_model="GAT_unweighted",
                summary_model_names=["GNN"],
                overrides=_gnn_v2_overrides(conv_type="GATConv", use_edge_weights=False),
            ),
        ]
    )

    for epochs in [1, 5, 10, 30, 50, 200]:
        variants.append(
            VariantSpec(
                variant_id=f"epochs_{epochs}_no_es",
                family="epochs",
                description=f"GNN v2 trained for exactly {epochs} epoch(s), early stopping disabled.",
                display_model=f"epochs_{epochs}",
                summary_model_names=["GNN"],
                overrides=_gnn_v2_overrides(num_epochs=epochs, early_stopping_enabled=False),
            )
        )

    return variants


def _build_payload(base_cfg: Dict[str, Any], fixed_overrides: Dict[str, Any], variant: VariantSpec) -> Dict[str, Any]:
    """Merge base wrapper, fixed scope overrides, and variant overrides."""
    payload = merge_many(base_cfg, fixed_overrides, variant.overrides)
    _enable_valence_only(payload)
    return payload


def _trainer_run_dir(suite_run_dir: Path) -> Path:
    """Resolve the successful valence trainer directory for one suite run."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    registry = pd.read_csv(registry_path)
    row = registry[
        (registry["experiment_id"] == VALENCE_EXPERIMENT_ID)
        & (registry["status"] == "success")
    ]
    if row.empty:
        raise ValueError(f"No successful valence row in {registry_path}")
    return Path(str(row.iloc[-1]["trainer_run_dir"]))


def _variant_rows(
    variant: VariantSpec,
    suite_run_dir: Path,
    config_path: Path,
    runtime_seconds: float,
) -> List[Dict[str, Any]]:
    """Collect one or more summary rows for a completed variant."""
    rows: List[Dict[str, Any]] = []
    trainer_run_dir = _trainer_run_dir(suite_run_dir)
    for summary_model_name in variant.summary_model_names:
        model_label = summary_model_name if variant.display_model == "Baselines" else variant.display_model
        row: Dict[str, Any] = {
            "variant_id": variant.variant_id,
            "family": variant.family,
            "description": variant.description,
            "experiment_id": VALENCE_EXPERIMENT_ID,
            "experiment_display_name": "Table-6 Valence",
            "cv_strategy": "subject_kfold",
            "model": model_label,
            "summary_model_name": summary_model_name,
            "status": "success",
            "wrapper_config_path": str(config_path),
            "suite_run_dir": str(suite_run_dir),
            "trainer_run_dir": str(trainer_run_dir),
            "runtime_seconds": round(runtime_seconds, 3),
            "error": "",
        }
        row.update(
            _collect_metrics(
                suite_run_dir=suite_run_dir,
                cv_strategy="subject_kfold",
                summary_model_name=summary_model_name,
                experiment_id=VALENCE_EXPERIMENT_ID,
            )
        )
        rows.append(row)
    return rows


def _rows_to_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convert result rows into a stable table."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    sort_columns = [column for column in ["family", "variant_id", "model"] if column in df.columns]
    return df.sort_values(sort_columns).reset_index(drop=True)


def run_matrix(args: argparse.Namespace) -> Path:
    """Run the requested sequential matrix and write summary artifacts."""
    base_config_path = Path(args.base_config)
    base_cfg = _load_yaml(base_config_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_root) / timestamp
    generated_dir = output_dir / "generated_wrapper_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    fixed_overrides = _base_overrides(args, run_output_dir=output_dir)
    variants = _build_variants()
    if args.only_family is not None:
        variants = [variant for variant in variants if variant.family == args.only_family]
    if args.only_variant is not None:
        variants = [variant for variant in variants if variant.variant_id == args.only_variant]
    if not variants:
        raise ValueError("No variants selected.")

    print("Table-6 valence v2 checks arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"Selected variants: {len(variants)}")
    print(f"Output directory: {output_dir}")

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        print("\n" + "=" * 100)
        print(f"Running {variant.family} | {variant.variant_id}: {variant.description}")
        payload = _build_payload(base_cfg=base_cfg, fixed_overrides=fixed_overrides, variant=variant)
        config_path = generated_dir / f"{variant.variant_id}.yaml"
        _write_yaml(config_path, payload)

        if args.dry_run:
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "family": variant.family,
                    "description": variant.description,
                    "experiment_id": VALENCE_EXPERIMENT_ID,
                    "experiment_display_name": "Table-6 Valence",
                    "cv_strategy": "subject_kfold",
                    "model": variant.display_model,
                    "summary_model_name": ",".join(variant.summary_model_names),
                    "status": "dry_run",
                    "wrapper_config_path": str(config_path),
                    "suite_run_dir": "",
                    "trainer_run_dir": "",
                    "runtime_seconds": np.nan,
                    "error": "",
                }
            )
            continue

        started = time.time()
        try:
            suite_run_dir = Path(run_suite(str(config_path)))
            rows.extend(
                _variant_rows(
                    variant=variant,
                    suite_run_dir=suite_run_dir,
                    config_path=config_path,
                    runtime_seconds=time.time() - started,
                )
            )
            print(f"Completed {variant.variant_id}")
        except Exception as exc:
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "family": variant.family,
                    "description": variant.description,
                    "experiment_id": VALENCE_EXPERIMENT_ID,
                    "experiment_display_name": "Table-6 Valence",
                    "cv_strategy": "subject_kfold",
                    "model": variant.display_model,
                    "summary_model_name": ",".join(variant.summary_model_names),
                    "status": "failed",
                    "wrapper_config_path": str(config_path),
                    "suite_run_dir": "",
                    "trainer_run_dir": "",
                    "runtime_seconds": round(time.time() - started, 3),
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )
            print(f"FAILED {variant.variant_id}: {exc}")

        summary_df = _rows_to_dataframe(rows)
        summary_df.to_csv(output_dir / "matrix_summary_partial.csv", index=False)

    summary_df = _rows_to_dataframe(rows)
    summary_path = output_dir / "matrix_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved matrix summary: {summary_path}")

    if not args.dry_run:
        training_paths = _save_training_history_outputs(rows=rows, output_dir=output_dir)
        test_loss_path = _plot_test_loss_summary(summary=summary_df, output_dir=output_dir)
        for path in training_paths:
            print(f"Saved training output: {path}")
        if test_loss_path is not None:
            print(f"Saved test loss plot: {test_loss_path}")

    manifest = {
        "created_at": timestamp,
        "base_config": str(base_config_path),
        "output_dir": str(output_dir),
        "selected_variant_count": len(variants),
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "val_size": int(args.val_size),
        "num_workers": int(args.num_workers),
        "lightgbm_n_jobs": int(args.lightgbm_n_jobs),
        "only_family": args.only_family,
        "only_variant": args.only_variant,
        "dry_run": bool(args.dry_run),
        "summary_csv": str(summary_path),
    }
    _write_yaml(output_dir / "run_manifest.yaml", manifest)
    return output_dir


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    output_dir = run_matrix(args)
    print(f"Experiment matrix directory: {output_dir}")


if __name__ == "__main__":
    main()
