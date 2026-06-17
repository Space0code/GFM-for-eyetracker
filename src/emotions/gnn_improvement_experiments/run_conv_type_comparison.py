"""Run the one-time gaze+pupil GNN convolution comparison.

This script compares the homogeneous `BasicGCN` baseline and the weighted
heterogeneous `HeteroGCNMLPWeights` architecture on the low/high Table-6
valence task using only the `gaze_pupil` signal set. By default it preserves the
current quick-comparison data, training, and 7-fold subject-CV settings.

Usage:
  python src/emotions/gnn_improvement_experiments/run_conv_type_comparison.py

Useful options:
  python src/emotions/gnn_improvement_experiments/run_conv_type_comparison.py --dry-run
  python src/emotions/gnn_improvement_experiments/run_conv_type_comparison.py \
      --only-variant BasicGCN_GCNConv --num-epochs 1

Outputs are written under:
  results/conv_type_comparison/<timestamp>/
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

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.gnn_improvement_experiments.run_quick_v1_v2_comparison import (
    EXPERIMENT_DISPLAY_NAMES,
    VALENCE_EXPERIMENT_ID,
    QuickRun,
    _build_run_payload,
    _collect_metrics,
    _get_cv_strategies,
    _get_enabled_experiment_ids,
    _load_yaml,
    _rows_to_dataframe,
    _write_yaml,
    build_fixed_overrides,
    build_signal_set_variant,
)
from emotions.suite.config_merge import merge_many
from emotions.suite.run_hci_experiment_suite import run_suite


@dataclass(frozen=True)
class ConvVariant:
    """One architecture/convolution variant for the comparison."""

    variant_id: str
    architecture: str
    conv_type: str
    summary_conv_type: str
    edge_info_mode: str
    description: str


CONV_VARIANTS: tuple[ConvVariant, ...] = (
    ConvVariant(
        variant_id="BasicGCN_GCNConv",
        architecture="BasicGCN",
        conv_type="GCNConv",
        summary_conv_type="GCNConv",
        edge_info_mode="collapsed_unweighted_structural",
        description="BasicGCN with collapsed v2 relations and structural GCN convolution.",
    ),
    ConvVariant(
        variant_id="BasicGCN_GATConv",
        architecture="BasicGCN",
        conv_type="GATConv",
        summary_conv_type="GATConv",
        edge_info_mode="collapsed_native_attention_no_edge_attr",
        description="BasicGCN with collapsed v2 relations and native GAT attention over structure.",
    ),
    ConvVariant(
        variant_id="BasicGCN_GraphConv",
        architecture="BasicGCN",
        conv_type="GraphConv",
        summary_conv_type="GraphConv",
        edge_info_mode="collapsed_unweighted_graphconv",
        description="BasicGCN with collapsed v2 relations and unweighted GraphConv.",
    ),
    ConvVariant(
        variant_id="BasicGCN_GINConv",
        architecture="BasicGCN",
        conv_type="GINConv",
        summary_conv_type="GINConv",
        edge_info_mode="collapsed_unweighted_gin",
        description="BasicGCN with collapsed v2 relations and unweighted GINConv.",
    ),
    ConvVariant(
        variant_id="HeteroGCNMLPWeights_GCNConv",
        architecture="HeteroGCNMLPWeights",
        conv_type="GCNConv",
        summary_conv_type="GCNConv",
        edge_info_mode="relation_mlp_scalar_edge_weight",
        description="Weighted heterogeneous MLP-fusion architecture with GCNConv.",
    ),
    ConvVariant(
        variant_id="HeteroGCNMLPWeights_GATConv",
        architecture="HeteroGCNMLPWeights",
        conv_type="GATConv",
        summary_conv_type="GATConv",
        edge_info_mode="native_attention_edge_attr_no_scalar_weight",
        description="Weighted heterogeneous MLP-fusion architecture using GAT edge attributes.",
    ),
    ConvVariant(
        variant_id="HeteroGCNMLPWeights_GraphConv",
        architecture="HeteroGCNMLPWeights",
        conv_type="GraphConv",
        summary_conv_type="GraphConv",
        edge_info_mode="relation_mlp_scalar_edge_weight",
        description="Weighted heterogeneous MLP-fusion architecture using weighted GraphConv.",
    ),
    ConvVariant(
        variant_id="HeteroGCNMLPWeights_GINEConv",
        architecture="HeteroGCNMLPWeights",
        conv_type="GINEConv",
        summary_conv_type="GINEConv",
        edge_info_mode="edge_attr_plus_relation_mlp_scalar_weight",
        description="Weighted heterogeneous MLP-fusion architecture using weighted GINEConv.",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the gaze+pupil GNN conv-type comparison")
    parser.add_argument(
        "--base-config",
        type=str,
        default=(
            "src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/"
            "run_hci_experiment_suite_table6_low_high.yaml"
        ),
        help="Base low/high Table-6 wrapper YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/conv_type_comparison",
        help="Output directory for generated configs, suite runs, and summaries.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional suite/CV seed override.")
    parser.add_argument("--n-splits", type=int, default=7, help="Subject k-fold split count.")
    parser.add_argument("--val-size", type=int, default=None, help="Validation subject count per fold.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Optional GNN epoch override.")
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="subject_kfold",
        help="CV strategy override. Defaults to subject_kfold.",
    )
    parser.add_argument(
        "--only-variant",
        type=str,
        default=None,
        help="Run only one variant id, for example BasicGCN_GCNConv.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Enable torch.compile for GNN runs. Disabled by default for PyG robustness.",
    )
    parser.add_argument(
        "--enable-benchmarking",
        action="store_true",
        help="Keep model benchmarking enabled. Disabled by default to focus this experiment on metrics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write configs only; do not train.")
    return parser.parse_args()


def _selected_variants(only_variant: str | None) -> List[ConvVariant]:
    """Return requested conv variants in stable order."""
    variants = list(CONV_VARIANTS)
    if only_variant is None:
        return variants
    selected = [variant for variant in variants if variant.variant_id == only_variant]
    if not selected:
        valid = ", ".join(variant.variant_id for variant in variants)
        raise ValueError(f"Unknown --only-variant '{only_variant}'. Valid values: {valid}")
    return selected


def _fixed_overrides(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """Build run-wide overrides shared by all variants."""
    quick_args = argparse.Namespace(
        output_root=args.output_root,
        seed=args.seed,
        cv_strategy=args.cv_strategy,
        n_splits=args.n_splits,
        val_size=args.val_size,
        num_epochs=args.num_epochs,
        use_torch_compile=args.use_torch_compile,
    )
    overrides = build_fixed_overrides(quick_args, run_output_dir=output_dir)
    overrides = merge_many(
        overrides,
        {
            "quick_comparison": {
                "models": ["BasicGCN", "HeteroGCNMLPWeights"],
                "signal_sets": ["gaze_pupil"],
                "table6_tasks": ["valence"],
            },
            "global_overrides": {
                "benchmarking": {"enabled": bool(args.enable_benchmarking)},
                "run_experiments": {"baselines": False, "gnn": True},
            },
        },
    )
    return overrides


def _quick_run_for_variant(variant: ConvVariant) -> QuickRun:
    """Build the one-model suite invocation for a conv variant."""
    return QuickRun(
        run_name=variant.variant_id,
        description=variant.description,
        model_names=[variant.architecture],
        summary_model_names={variant.architecture: variant.architecture},
        overrides={
            "global_overrides": {
                "dataset": {
                    "graph_version": "v2",
                    "edge_weight_mode": "learned_signed",
                    "use_edge_weights": True,
                },
                "gnn": {
                    "models": [variant.architecture],
                    "model": {
                        "model_version": variant.architecture,
                        "conv_type": variant.conv_type,
                    },
                },
            }
        },
    )


def _manifest_rows(variants: Iterable[ConvVariant]) -> List[Dict[str, str]]:
    """Build human-readable variant metadata rows."""
    return [
        {
            "variant_id": variant.variant_id,
            "architecture": variant.architecture,
            "conv_type": variant.summary_conv_type,
            "configured_conv_type": variant.conv_type,
            "edge_info_mode": variant.edge_info_mode,
            "description": variant.description,
        }
        for variant in variants
    ]


def _write_manifest(output_dir: Path, args: argparse.Namespace, variants: List[ConvVariant]) -> None:
    """Save run arguments and variant metadata."""
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": args.base_config,
        "output_root": args.output_root,
        "dry_run": bool(args.dry_run),
        "only_variant": args.only_variant,
        "n_splits": args.n_splits,
        "val_size": args.val_size,
        "num_epochs": args.num_epochs,
        "cv_strategy": args.cv_strategy,
        "signal_sets": ["gaze_pupil"],
        "table6_tasks": ["valence"],
        "variants": _manifest_rows(variants),
    }
    with (output_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    pd.DataFrame(payload["variants"]).to_csv(output_dir / "variant_manifest.csv", index=False)


def run_conv_type_comparison(args: argparse.Namespace) -> Path:
    """Run or dry-run the convolution comparison and return the output directory."""
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_root) / timestamp
    generated_dir = output_dir / "generated_wrapper_configs" / "gaze_pupil"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(base_config_path)
    fixed_overrides = _fixed_overrides(args, output_dir=output_dir)
    signal_set = build_signal_set_variant("gaze_pupil")
    variants = _selected_variants(args.only_variant)
    _write_manifest(output_dir=output_dir, args=args, variants=variants)

    print("Conv-type comparison arguments:")
    print(f"  base_config: {base_config_path}")
    print(f"  output_dir: {output_dir}")
    print("  signal_sets: gaze_pupil")
    print("  table6_tasks: valence")
    print(f"  variants: {', '.join(variant.variant_id for variant in variants)}")
    print(f"  dry_run: {args.dry_run}")

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        quick_run = _quick_run_for_variant(variant)
        variant_overrides = merge_many(
            fixed_overrides,
            {"suite": {"results_dir": str(output_dir / "model_runs" / variant.variant_id)}},
        )
        payload = _build_run_payload(
            base_cfg=base_cfg,
            fixed_overrides=variant_overrides,
            signal_set=signal_set,
            quick_run=quick_run,
        )
        cv_strategies = _get_cv_strategies(payload)
        experiment_ids = _get_enabled_experiment_ids(payload)
        config_path = generated_dir / f"{variant.variant_id}.yaml"
        _write_yaml(config_path, payload)

        base_row: Dict[str, Any] = {
            "variant_id": variant.variant_id,
            "architecture": variant.architecture,
            "conv_type": variant.summary_conv_type,
            "configured_conv_type": variant.conv_type,
            "edge_info_mode": variant.edge_info_mode,
            "signal_set": "gaze_pupil",
            "signal_set_description": signal_set.description,
            "run_name": quick_run.run_name,
            "description": variant.description,
            "wrapper_config_path": str(config_path),
            "suite_run_dir": "",
            "runtime_seconds": np.nan,
            "error": "",
        }
        if args.dry_run:
            for experiment_id in experiment_ids:
                for cv_strategy in cv_strategies:
                    rows.append(
                        {
                            **base_row,
                            "experiment_id": experiment_id,
                            "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                            "cv_strategy": cv_strategy,
                            "model": variant.variant_id,
                            "summary_model_name": variant.architecture,
                            "status": "dry_run",
                        }
                    )
            print(f"dry-run | {variant.variant_id} | {config_path}")
            continue

        started = time.time()
        try:
            suite_run_dir = Path(run_suite(str(config_path)))
            runtime_seconds = round(time.time() - started, 3)
            for experiment_id in experiment_ids:
                for cv_strategy in cv_strategies:
                    row: Dict[str, Any] = {
                        **base_row,
                        "experiment_id": experiment_id,
                        "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                        "cv_strategy": cv_strategy,
                        "model": variant.variant_id,
                        "summary_model_name": variant.architecture,
                        "suite_run_dir": str(suite_run_dir),
                        "runtime_seconds": runtime_seconds,
                        "status": "success",
                    }
                    try:
                        row.update(
                            _collect_metrics(
                                suite_run_dir=suite_run_dir,
                                cv_strategy=cv_strategy,
                                summary_model_name=variant.architecture,
                                experiment_id=experiment_id,
                            )
                        )
                    except Exception as metric_exc:
                        row["status"] = "failed"
                        row["error"] = f"{metric_exc}\n{traceback.format_exc()}"
                    rows.append(row)
        except Exception as exc:
            runtime_seconds = round(time.time() - started, 3)
            for experiment_id in experiment_ids:
                for cv_strategy in cv_strategies:
                    rows.append(
                        {
                            **base_row,
                            "experiment_id": experiment_id,
                            "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                            "cv_strategy": cv_strategy,
                            "model": variant.variant_id,
                            "summary_model_name": variant.architecture,
                            "runtime_seconds": runtime_seconds,
                            "status": "failed",
                            "error": f"{exc}\n{traceback.format_exc()}",
                        }
                    )

        for row in rows[-(len(cv_strategies) * len(experiment_ids)) :]:
            print(
                f"{row['status']} | {row['variant_id']} | {row['experiment_id']} | "
                f"{row['cv_strategy']} | balanced_accuracy={row.get('balanced_accuracy', np.nan)}"
            )

    summary = _rows_to_dataframe(rows)
    summary_path = output_dir / "conv_type_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    return output_dir


def main() -> None:
    """CLI entrypoint."""
    run_conv_type_comparison(parse_args())


if __name__ == "__main__":
    main()
