"""Run focused GNN ablations for HCI binary valence/arousal tasks.

This script generates one-factor-at-a-time wrapper configs and runs the suite
sequentially to compare GNN variants quickly on a fixed data subset.

Usage:
  python src/emotions/gnn_improvement_experiments/run_gnn_ablation_suite.py \
      --base-config src/emotions/gnn_improvement_experiments/configs/run_hci_experiment_suite_small.yaml

Optional flags:
  --output-root results/gnn_improvement_experiments
  --seed 42
  --top-k 3
  --dry-run
  --skip-pass2
  --only-variant baseline_default
"""

from __future__ import annotations

import argparse
import sys
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


VALENCE_EXPERIMENT_ID = "binary_emotion_valence"
AROUSAL_EXPERIMENT_ID = "binary_emotion_arousal"
TARGET_EXPERIMENT_IDS = {VALENCE_EXPERIMENT_ID, AROUSAL_EXPERIMENT_ID}

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


@dataclass(frozen=True)
class VariantSpec:
    """One ablation variant with deterministic config overrides."""

    variant_id: str
    family: str
    description: str
    overrides: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for ablation orchestration."""
    parser = argparse.ArgumentParser(description="Run GNN ablation suite")
    parser.add_argument(
        "--base-config",
        type=str,
        default="src/emotions/gnn_improvement_experiments/configs/run_hci_experiment_suite_small.yaml",
        help="Path to base wrapper config YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/gnn_improvement_experiments",
        help="Directory for generated configs and ablation summaries.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of pass-1 variants promoted to pass-2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs/manifest only, without training runs.",
    )
    parser.add_argument(
        "--skip-pass2",
        action="store_true",
        help="Run pass-1 only.",
    )
    parser.add_argument(
        "--only-variant",
        type=str,
        default=None,
        help="Run only one pass-1 variant by id.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML from disk and return dictionary payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict at YAML root in {path}.")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write dictionary payload as YAML with stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _set_only_binary_valence_arousal(wrapper_cfg: Dict[str, Any]) -> None:
    """Enable only binary valence/arousal experiments inside wrapper config."""
    experiments = wrapper_cfg.get("experiments")
    if isinstance(experiments, dict):
        for experiment_id, experiment_cfg in experiments.items():
            if not isinstance(experiment_cfg, dict):
                continue
            experiment_cfg["enabled"] = experiment_id in TARGET_EXPERIMENT_IDS
        return

    if isinstance(experiments, list):
        for experiment_cfg in experiments:
            if not isinstance(experiment_cfg, dict):
                continue
            experiment_id = str(experiment_cfg.get("id", ""))
            experiment_cfg["enabled"] = experiment_id in TARGET_EXPERIMENT_IDS
        return

    raise ValueError("Wrapper config has unsupported 'experiments' format; expected dict or list.")


def build_fixed_wrapper_overrides(seed: int) -> Dict[str, Any]:
    """Build fixed scope/training overrides applied to all variants."""
    return {
        "suite": {
            "seed": int(seed),
        },
        "global_overrides": {
            "cross_validation": {
                "strategies": ["recording_kfold"],
                "n_splits": 3,
                "val_size": 1,
                "random_state": int(seed),
            },
            "run_experiments": {
                "baselines": False,
                "gnn": True,
            },
            "dataset": {
                "use_cache": True,
                "kt": 2,
                "ks": 2,
                "use_edge_weights": False,
                "filter_subjects": DEFAULT_SUBJECTS,
                "filter_recordings": DEFAULT_RECORDINGS,
                "target_aggregation": "mean",
            },
            "gnn": {
                "model": {
                    "conv_type": "GCNConv",
                    "pooling": "mean_max",
                    "num_layers": 2,
                },
                "training": {
                    "num_epochs": 20,
                    "num_workers": 0,
                    "persistent_workers": False,
                    "use_torch_compile": False,
                    "early_stopping_enabled": False,
                    "early_stopping_patience": 7,
                    "early_stopping_min_delta": 0.0,
                    "early_stopping_restore_best": True,
                },
            },
        },
    }


def build_pass1_variants() -> List[VariantSpec]:
    """Return all pass-1 one-factor ablation variants."""
    variants: List[VariantSpec] = [
        VariantSpec(
            variant_id="baseline_default",
            family="baseline",
            description="Default config with fixed subset/splits.",
            overrides={},
        ),
    ]

    for kt in [1, 3, 5]:
        for ks in [1, 3, 5]:
            variants.append(
                VariantSpec(
                    variant_id=f"kt{kt}_ks{ks}",
                    family="kt_ks_grid",
                    description=f"Set kt={kt}, ks={ks}.",
                    overrides={"global_overrides": {"dataset": {"kt": kt, "ks": ks}}},
                )
            )

    variants.extend(
        [
            VariantSpec(
                variant_id="pooling_mean",
                family="pooling",
                description="Use mean graph pooling head input.",
                overrides={"global_overrides": {"gnn": {"model": {"pooling": "mean"}}}},
            ),
            VariantSpec(
                variant_id="edge_weights_on",
                family="edge_weights",
                description="Enable temporal/spatial edge weights.",
                overrides={"global_overrides": {"dataset": {"use_edge_weights": True}}},
            ),
            VariantSpec(
                variant_id="conv_gat",
                family="conv_type",
                description="Use GATConv (unweighted).",
                overrides={"global_overrides": {"gnn": {"model": {"conv_type": "GATConv"}}}},
            ),
            VariantSpec(
                variant_id="target_aggregation_last",
                family="target_aggregation",
                description="Use last target value per window.",
                overrides={"global_overrides": {"dataset": {"target_aggregation": "last"}}},
            ),
            VariantSpec(
                variant_id="early_stopping_on",
                family="early_stopping",
                description="Enable early stopping (patience=7, min_delta=0).",
                overrides={
                    "global_overrides": {
                        "gnn": {
                            "training": {
                                "early_stopping_enabled": True,
                                "early_stopping_patience": 7,
                                "early_stopping_min_delta": 0.0,
                                "early_stopping_restore_best": True,
                            }
                        }
                    }
                },
            ),
        ]
    )

    for depth in [1, 3, 5, 10, 50]:
        variants.append(
            VariantSpec(
                variant_id=f"num_layers_{depth}",
                family="num_layers",
                description=f"Use {depth} GNN layer(s).",
                overrides={"global_overrides": {"gnn": {"model": {"num_layers": depth}}}},
            )
        )

    return variants


def _get_cv_strategy_name(wrapper_cfg: Dict[str, Any]) -> str:
    """Resolve primary CV strategy from wrapper global overrides."""
    global_overrides = wrapper_cfg.get("global_overrides", {})
    cv_cfg = global_overrides.get("cross_validation", {})
    strategies = cv_cfg.get("strategies", ["recording_kfold"])
    if isinstance(strategies, str):
        return strategies
    if not strategies:
        return "recording_kfold"
    return str(strategies[0])


def _build_wrapper_payload(
    base_wrapper_cfg: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    variant: VariantSpec,
    *,
    num_epochs: int,
    force_early_stopping: bool,
) -> Dict[str, Any]:
    """Assemble one wrapper payload for one variant and pass."""
    payload = merge_many(base_wrapper_cfg, fixed_overrides, variant.overrides)
    payload = merge_many(
        payload,
        {"global_overrides": {"gnn": {"training": {"num_epochs": int(num_epochs)}}}},
    )
    if force_early_stopping:
        payload = merge_many(
            payload,
            {
                "global_overrides": {
                    "gnn": {
                        "training": {
                            "early_stopping_enabled": True,
                            "early_stopping_patience": 7,
                            "early_stopping_min_delta": 0.0,
                            "early_stopping_restore_best": True,
                        }
                    }
                }
            },
        )
    _set_only_binary_valence_arousal(payload)
    return payload


def _read_gnn_balanced_accuracy(summary_csv_path: Path) -> float:
    """Read GNN aggregated balanced_accuracy from one trainer summary CSV."""
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv_path}")

    df = pd.read_csv(summary_csv_path)
    row = df[(df["model"] == "GNN") & (df["metric_type"] == "aggregated")]
    if row.empty:
        raise ValueError(f"No GNN aggregated row in {summary_csv_path}")
    if "balanced_accuracy" not in row.columns:
        raise ValueError(f"Column 'balanced_accuracy' not found in {summary_csv_path}")
    return float(row.iloc[0]["balanced_accuracy"])


def _collect_variant_metrics(suite_run_dir: Path, cv_strategy: str) -> Dict[str, float]:
    """Collect balanced accuracy for valence/arousal and mean score."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Suite registry not found: {registry_path}")

    registry = pd.read_csv(registry_path)
    result: Dict[str, float] = {}
    experiment_to_key = {
        VALENCE_EXPERIMENT_ID: "valence_balanced_accuracy",
        AROUSAL_EXPERIMENT_ID: "arousal_balanced_accuracy",
    }

    for experiment_id, key in experiment_to_key.items():
        subset = registry[
            (registry["experiment_id"] == experiment_id)
            & (registry["status"] == "success")
        ]
        if subset.empty:
            raise ValueError(f"Missing successful run for experiment_id='{experiment_id}' in {registry_path}")

        trainer_run_dir = Path(str(subset.iloc[-1]["trainer_run_dir"]))
        summary_csv_path = trainer_run_dir / cv_strategy / "summary.csv"
        result[key] = _read_gnn_balanced_accuracy(summary_csv_path)

    result["variant_score"] = float(
        np.mean([result["valence_balanced_accuracy"], result["arousal_balanced_accuracy"]])
    )
    return result


def _run_one_variant(
    variant: VariantSpec,
    wrapper_payload: Dict[str, Any],
    wrapper_config_path: Path,
    cv_strategy: str,
    *,
    dry_run: bool,
    run_phase: str,
) -> Dict[str, Any]:
    """Run one variant and return one result row."""
    _write_yaml(wrapper_config_path, wrapper_payload)

    row: Dict[str, Any] = {
        "phase": run_phase,
        "variant_id": variant.variant_id,
        "family": variant.family,
        "description": variant.description,
        "wrapper_config_path": str(wrapper_config_path),
        "suite_run_dir": "",
        "status": "pending",
        "valence_balanced_accuracy": np.nan,
        "arousal_balanced_accuracy": np.nan,
        "variant_score": np.nan,
        "error": "",
    }

    if dry_run:
        row["status"] = "dry_run"
        return row

    try:
        suite_run_dir = Path(run_suite(str(wrapper_config_path)))
        metrics = _collect_variant_metrics(suite_run_dir=suite_run_dir, cv_strategy=cv_strategy)
        row.update(metrics)
        row["suite_run_dir"] = str(suite_run_dir)
        row["status"] = "success"
    except Exception as exc:  # Keep runs resilient and continue remaining variants.
        row["status"] = "failed"
        row["error"] = f"{exc}\n{traceback.format_exc()}"

    return row


def _rows_to_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convert result rows to stable DataFrame schema."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    return df.sort_values(["status", "variant_score", "variant_id"], ascending=[True, False, True]).reset_index(drop=True)


def _select_top_variants(pass1_df: pd.DataFrame, top_k: int) -> List[str]:
    """Select top-k successful pass-1 variants by score."""
    if pass1_df.empty:
        return []
    successful = pass1_df[pass1_df["status"] == "success"].copy()
    if successful.empty:
        return []
    successful = successful.sort_values("variant_score", ascending=False)
    return successful["variant_id"].head(top_k).tolist()


def _build_final_ranking(pass1_df: pd.DataFrame, pass2_df: pd.DataFrame) -> pd.DataFrame:
    """Build final ranking table combining pass-1 and pass-2 scores."""
    if pass1_df.empty:
        return pd.DataFrame()

    pass1_scores = pass1_df[["variant_id", "variant_score"]].rename(
        columns={"variant_score": "pass1_variant_score"}
    )
    if pass2_df.empty:
        merged = pass1_scores.copy()
        merged["pass2_variant_score"] = np.nan
    else:
        pass2_scores = pass2_df[["variant_id", "variant_score"]].rename(
            columns={"variant_score": "pass2_variant_score"}
        )
        merged = pass1_scores.merge(pass2_scores, on="variant_id", how="left")

    merged["final_variant_score"] = merged["pass2_variant_score"].where(
        merged["pass2_variant_score"].notna(), merged["pass1_variant_score"]
    )
    return merged.sort_values("final_variant_score", ascending=False).reset_index(drop=True)


def run_ablation(args: argparse.Namespace) -> Path:
    """Run pass-1 and pass-2 ablations and save summary artifacts."""
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_root) / timestamp
    generated_dir = output_dir / "generated_wrapper_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    base_wrapper_cfg = _load_yaml(base_config_path)
    fixed_overrides = build_fixed_wrapper_overrides(seed=int(args.seed))
    pass1_variants = build_pass1_variants()
    if args.only_variant:
        pass1_variants = [v for v in pass1_variants if v.variant_id == args.only_variant]
        if not pass1_variants:
            raise ValueError(f"Unknown variant id: {args.only_variant}")

    cv_strategy = _get_cv_strategy_name(merge_many(base_wrapper_cfg, fixed_overrides))
    print(f"Running pass-1 variants: {len(pass1_variants)} | cv_strategy={cv_strategy}")

    pass1_rows: List[Dict[str, Any]] = []
    for variant in pass1_variants:
        wrapper_payload = _build_wrapper_payload(
            base_wrapper_cfg=base_wrapper_cfg,
            fixed_overrides=fixed_overrides,
            variant=variant,
            num_epochs=20,
            force_early_stopping=False,
        )
        cfg_path = generated_dir / f"pass1__{variant.variant_id}.yaml"
        row = _run_one_variant(
            variant=variant,
            wrapper_payload=wrapper_payload,
            wrapper_config_path=cfg_path,
            cv_strategy=cv_strategy,
            dry_run=bool(args.dry_run),
            run_phase="pass1",
        )
        print(f"pass1 | {variant.variant_id} | {row['status']} | score={row['variant_score']}")
        pass1_rows.append(row)

    pass1_df = _rows_to_dataframe(pass1_rows)
    pass1_path = output_dir / "pass1_scores.csv"
    pass1_df.to_csv(pass1_path, index=False)

    pass2_rows: List[Dict[str, Any]] = []
    top_variant_ids: List[str] = []
    if not args.skip_pass2 and args.only_variant is None:
        top_variant_ids = _select_top_variants(pass1_df=pass1_df, top_k=int(args.top_k))
        id_to_variant = {v.variant_id: v for v in pass1_variants}
        print(f"Running pass-2 variants (top {args.top_k}): {top_variant_ids}")
        for variant_id in top_variant_ids:
            variant = id_to_variant[variant_id]
            wrapper_payload = _build_wrapper_payload(
                base_wrapper_cfg=base_wrapper_cfg,
                fixed_overrides=fixed_overrides,
                variant=variant,
                num_epochs=50,
                force_early_stopping=True,
            )
            cfg_path = generated_dir / f"pass2__{variant.variant_id}.yaml"
            row = _run_one_variant(
                variant=variant,
                wrapper_payload=wrapper_payload,
                wrapper_config_path=cfg_path,
                cv_strategy=cv_strategy,
                dry_run=bool(args.dry_run),
                run_phase="pass2",
            )
            print(f"pass2 | {variant.variant_id} | {row['status']} | score={row['variant_score']}")
            pass2_rows.append(row)

    pass2_df = _rows_to_dataframe(pass2_rows)
    pass2_path = output_dir / "pass2_scores.csv"
    pass2_df.to_csv(pass2_path, index=False)

    final_df = _build_final_ranking(pass1_df=pass1_df, pass2_df=pass2_df)
    final_path = output_dir / "final_ranked_variants.csv"
    final_df.to_csv(final_path, index=False)

    manifest = {
        "created_at": timestamp,
        "base_config": str(base_config_path),
        "output_dir": str(output_dir),
        "fixed_subjects": DEFAULT_SUBJECTS,
        "fixed_recordings": DEFAULT_RECORDINGS,
        "cv_strategy": cv_strategy,
        "seed": int(args.seed),
        "top_k": int(args.top_k),
        "dry_run": bool(args.dry_run),
        "skip_pass2": bool(args.skip_pass2),
        "only_variant": args.only_variant,
        "pass1_variant_count": int(len(pass1_variants)),
        "pass2_selected_variants": top_variant_ids,
        "pass1_scores_csv": str(pass1_path),
        "pass2_scores_csv": str(pass2_path),
        "final_scores_csv": str(final_path),
    }
    with (output_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    print(f"Saved ablation artifacts to: {output_dir}")
    return output_dir


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
