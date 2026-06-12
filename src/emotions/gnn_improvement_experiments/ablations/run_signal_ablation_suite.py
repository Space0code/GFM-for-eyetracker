"""Run signal-source ablations for the final weighted heterogeneous GCN.

This workflow keeps the `HeteroGCNMLPWeights` architecture fixed and removes
one information source at a time from graph construction: node features, learned
edge features, and graph relations derived from that source. It reuses the
existing HCI suite wrapper configs and trainers.

Examples:
  python src/emotions/gnn_improvement_experiments/ablations/run_signal_ablation_suite.py

  python src/emotions/gnn_improvement_experiments/ablations/run_signal_ablation_suite.py \
      --base-config src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_low_high.yaml \
      --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[3]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.suite.config_merge import merge_many
from emotions.suite.run_hci_experiment_suite import run_suite


DEFAULT_BASE_CONFIG = (
    "src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/"
    "run_hci_experiment_suite_table6_3class.yaml"
)
DEFAULT_VARIANTS = [
    "baseline_full",
    "without_temporal",
    "without_spatial_gaze",
    "without_fixation",
    "without_pupil",
    "without_screen_distance",
]
BASE_DROPNA_COLUMNS = ["time-rel-seconds", "emotion-id", "subject", "recording"]
GAZE_COLUMNS = ["x-avg", "y-avg"]
PUPIL_COLUMNS = ["pupil-size-left-avg", "pupil-size-right-avg"]


@dataclass(frozen=True)
class SignalAblationVariant:
    """One signal-ablation variant and its suite override block."""

    name: str
    description: str
    overrides: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run final weighted heterogeneous GCN signal ablations")
    parser.add_argument("--base-config", type=str, default=DEFAULT_BASE_CONFIG, help="Base suite wrapper YAML.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/signal_ablation_suite",
        help="Output root for generated configs and suite runs.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help=f"Comma-separated variants. Available: {', '.join(DEFAULT_VARIANTS)}.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Subject k-fold split count.")
    parser.add_argument("--val-size", type=int, default=1, help="Validation subject count per fold.")
    parser.add_argument("--seed", type=int, default=42, help="Suite/CV random seed.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Optional GNN epoch override.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs without running suites.")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load one YAML dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML dictionary at {path}.")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write one YAML dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _dropna_columns(*, use_gaze: bool = True, use_pupil: bool = True) -> List[str]:
    """Return dropna columns for active raw node signals."""
    columns = list(BASE_DROPNA_COLUMNS)
    if use_gaze:
        columns.extend(GAZE_COLUMNS)
    if use_pupil:
        columns.extend(PUPIL_COLUMNS)
    return columns


def _outlier_columns(*, use_gaze: bool = True, use_pupil: bool = True) -> List[str]:
    """Return signal-outlier columns for active raw gaze/pupil sources."""
    columns: List[str] = []
    if use_gaze:
        columns.extend(GAZE_COLUMNS)
    if use_pupil:
        columns.extend(PUPIL_COLUMNS)
    return columns


def _dataset_overrides(
    *,
    use_temporal: bool = True,
    use_gaze: bool = True,
    use_fixation: bool = True,
    use_pupil: bool = True,
    use_screen_distance: bool = True,
) -> Dict[str, Any]:
    """Build dataset overrides for one signal-source combination."""
    return {
        "graph_version": "v2",
        "edge_weight_mode": "learned_signed",
        "use_edge_weights": True,
        "use_relative_time": use_temporal,
        "use_temporal_node_feature": use_temporal,
        "use_temporal_edge_features": use_temporal,
        "use_temporal_edges": use_temporal,
        "use_gaze_node_features": use_gaze,
        "use_gaze_edge_features": use_gaze,
        "use_spatial_edges": use_gaze,
        "use_pupil_node_features": use_pupil,
        "use_distance_avg": use_screen_distance,
        "use_screen_distance_node_feature": use_screen_distance,
        "use_delta_distance_edge_feature": use_screen_distance,
        "use_screen_distance_edge_feature": use_screen_distance,
        "use_fixation_duration": use_fixation,
        "use_fixation_node_feature": use_fixation,
        "use_fixation_edges": use_fixation,
        "dropna_columns": _dropna_columns(use_gaze=use_gaze, use_pupil=use_pupil),
        "signal_outlier_filter": {
            "enabled": True,
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "columns": _outlier_columns(use_gaze=use_gaze, use_pupil=use_pupil),
        },
    }


def _fixed_gnn_overrides() -> Dict[str, Any]:
    """Return final weighted heterogeneous GCN overrides shared by all ablations."""
    return {
        "benchmarking": {"enabled": False},
        "run_experiments": {"baselines": False, "gnn": True},
        "gnn": {
            "model": {
                "model_version": "HeteroGCNMLPWeights",
                "edge_weight_mode": "learned_signed",
                # Keep the final architecture's fixation relation module even
                # when the dataset variant removes fixation edges.
                "use_fixation_edges": True,
            }
        },
    }


def build_signal_ablation_variant(name: str) -> SignalAblationVariant:
    """Build one named signal-ablation variant."""
    if name == "baseline_full":
        description = "Full GNN v2 graph with temporal, spatial/gaze, fixation, pupil, and screen-distance sources."
        dataset = _dataset_overrides()
    elif name == "without_temporal":
        description = "Remove temporal node features, temporal edge features, and temporal relations."
        dataset = _dataset_overrides(use_temporal=False)
    elif name == "without_spatial_gaze":
        description = "Remove gaze-position node features, gaze-derived edge features, and spatial edges."
        dataset = _dataset_overrides(use_gaze=False)
    elif name == "without_fixation":
        description = "Remove fixation-duration node features and intra-fixation edges."
        dataset = _dataset_overrides(use_fixation=False)
    elif name == "without_pupil":
        description = "Remove pupil-size node features."
        dataset = _dataset_overrides(use_pupil=False)
    elif name == "without_screen_distance":
        description = "Remove screen-distance node features and delta-distance edge features."
        dataset = _dataset_overrides(use_screen_distance=False)
    else:
        raise ValueError(f"Unknown ablation variant '{name}'. Available: {DEFAULT_VARIANTS}")

    return SignalAblationVariant(
        name=name,
        description=description,
        overrides={
            "global_overrides": merge_many(
                _fixed_gnn_overrides(),
                {"dataset": dataset},
            )
        },
    )


def build_command_overrides(args: argparse.Namespace, run_output_dir: Path) -> Dict[str, Any]:
    """Build command-level overrides shared by all variants."""
    overrides: Dict[str, Any] = {
        "suite": {
            "seed": int(args.seed),
            "results_dir": str(run_output_dir / "suite_runs"),
        },
        "global_overrides": {
            "cross_validation": {
                "strategies": ["subject_kfold"],
                "n_splits": int(args.n_splits),
                "val_size": int(args.val_size),
                "random_state": int(args.seed),
            }
        },
    }
    if args.num_epochs is not None:
        overrides["global_overrides"]["gnn"] = {"training": {"num_epochs": int(args.num_epochs)}}
    return overrides


def build_variant_payload(
    base_cfg: Dict[str, Any],
    command_overrides: Dict[str, Any],
    variant: SignalAblationVariant,
    variant_output_dir: Path,
) -> Dict[str, Any]:
    """Merge base, command, and variant overrides into one suite config."""
    payload = merge_many(
        base_cfg,
        command_overrides,
        variant.overrides,
        {"suite": {"results_dir": str(variant_output_dir / "suite")}},
    )
    return payload


def _parse_variants(raw_variants: str) -> List[str]:
    """Parse and validate requested variant names."""
    variants = [token.strip() for token in raw_variants.split(",") if token.strip()]
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}. Available: {DEFAULT_VARIANTS}")
    if not variants:
        raise ValueError("At least one ablation variant must be requested.")
    return list(dict.fromkeys(variants))


def _suite_registry_status(suite_run_dir: Path) -> tuple[str, str]:
    """Return aggregate status and message from a suite registry."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    if not registry_path.exists():
        return "failed", f"Missing suite registry: {registry_path}"

    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "failed", f"Empty suite registry: {registry_path}"

    failed = [row for row in rows if str(row.get("status", "")).lower() != "success"]
    if failed:
        failed_ids = [str(row.get("experiment_id", "<unknown>")) for row in failed]
        return "failed", "Failed suite experiment(s): " + ", ".join(failed_ids)
    return "success", ""


def run_signal_ablation_suite(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Generate and optionally execute suite configs for signal ablations."""
    base_config_path = Path(args.base_config)
    base_cfg = _load_yaml(base_config_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = Path(args.output_root) / timestamp
    command_overrides = build_command_overrides(args, run_output_dir=run_output_dir)

    rows: List[Dict[str, str]] = []
    for variant_name in _parse_variants(args.variants):
        variant = build_signal_ablation_variant(variant_name)
        variant_output_dir = run_output_dir / variant.name
        payload = build_variant_payload(
            base_cfg=base_cfg,
            command_overrides=command_overrides,
            variant=variant,
            variant_output_dir=variant_output_dir,
        )
        config_path = run_output_dir / "configs" / f"{variant.name}.yaml"
        _write_yaml(config_path, payload)

        suite_run_dir = ""
        status = "dry_run"
        error = ""
        if not args.dry_run:
            status = "success"
            try:
                suite_run_dir = run_suite(str(config_path))
                status, error = _suite_registry_status(Path(suite_run_dir))
            except Exception as exc:
                status = "failed"
                error = str(exc)

        rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "config_path": str(config_path),
                "status": status,
                "suite_run_dir": suite_run_dir,
                "error": error,
            }
        )

    summary_path = run_output_dir / "signal_ablation_runs.yaml"
    _write_yaml(summary_path, {"runs": rows})
    return rows


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    print("Signal ablation arguments:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    rows = run_signal_ablation_suite(args)
    print("\nSignal ablation run summary:")
    for row in rows:
        print(f"  {row['variant']}: {row['status']} ({row['config_path']})")
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
