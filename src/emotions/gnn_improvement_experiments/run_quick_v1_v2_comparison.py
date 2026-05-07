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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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


def build_fixed_overrides(args: argparse.Namespace, run_output_dir: Path | None = None) -> Dict[str, Any]:
    """Build fixed quick-run overrides shared by all variants."""
    results_dir = (
        run_output_dir / "model_runs"
        if run_output_dir is not None
        else Path(args.output_root) / "suite_runs"
    )
    return {
        "suite": {
            "seed": int(args.seed),
            "results_dir": str(results_dir),
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
    metric_aliases = {
        "accuracy": ["accuracy"],
        "balanced_accuracy": ["balanced_accuracy"],
        "macro_f1": ["macro_f1"],
        "weighted_f1": ["weighted_f1"],
        "auc": ["auc", "macro_auc_ovr", "weighted_auc_ovr"],
        "loss": ["loss"],
    }
    for output_metric, candidate_columns in metric_aliases.items():
        for column in candidate_columns:
            if column in model_row.columns and pd.notna(model_row.iloc[0][column]):
                metrics[output_metric] = float(model_row.iloc[0][column])
                break
    return metrics


def _resolve_trainer_run_dir(suite_run_dir: Path) -> Path:
    """Return the trainer run directory for the quick arousal experiment."""
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
    return Path(str(row.iloc[-1]["trainer_run_dir"]))


def _rows_to_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Build a stable summary dataframe from result rows."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    sort_cols = [col for col in ["status", "balanced_accuracy", "macro_f1", "model"] if col in df.columns]
    ascending = [True] + [False] * (len(sort_cols) - 1)
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _save_group_model_ranking(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save a command-level model ranking plot from quick summary metrics."""
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "auc"]
    available_metrics = [metric for metric in metric_columns if metric in summary.columns]
    if summary.empty or not available_metrics:
        return None
    plot_df = summary[summary["status"] == "success"].copy()
    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=available_metrics, how="all")
    if plot_df.empty:
        return None

    sort_metric = "balanced_accuracy" if "balanced_accuracy" in available_metrics else available_metrics[0]
    plot_df = plot_df.sort_values(sort_metric, ascending=False)
    long_df = plot_df.melt(
        id_vars=["model"],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long_df.empty:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "classification_group_model_ranking.png"

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.6 * len(plot_df))))
    sns.barplot(data=long_df, x="value", y="model", hue="metric", ax=ax, orient="h")
    ax.set_title("Quick Table-6 Arousal Model Ranking")
    ax.set_xlabel("metric value")
    ax.set_ylabel("model")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right", title="metric")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _load_class_display_names(trainer_run_dir: Path) -> Dict[int, str]:
    """Load encoded class display names from one trainer run."""
    metadata_path = trainer_run_dir / "class_metadata.yaml"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    index_to_name = payload.get("index_to_name") if isinstance(payload, dict) else None
    if not isinstance(index_to_name, dict):
        return {}
    result: Dict[int, str] = {}
    for raw_idx, raw_name in index_to_name.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        result[idx] = str(raw_name)
    return result


def _collect_predictions_for_variant(
    trainer_run_dir: Path,
    cv_strategy: str,
    summary_model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect concatenated test targets and predicted labels for one variant."""
    strategy_dir = trainer_run_dir / cv_strategy
    if not strategy_dir.exists():
        raise FileNotFoundError(f"Strategy directory not found: {strategy_dir}")

    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    for fold_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
        if summary_model_name == "GNN":
            pred_path = fold_dir / "test_predictions.npy"
            target_path = fold_dir / "test_targets.npy"
        else:
            pred_path = fold_dir / "baselines" / summary_model_name / "test_predictions.npy"
            target_path = fold_dir / "baselines" / summary_model_name / "test_targets.npy"
        if not pred_path.exists() or not target_path.exists():
            continue

        pred = np.asarray(np.load(pred_path))
        target = np.asarray(np.load(target_path)).reshape(-1)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        all_targets.append(target.astype(int))
        all_preds.append(np.argmax(pred, axis=1).astype(int))

    if not all_targets:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    return np.concatenate(all_targets), np.concatenate(all_preds)


def _save_combined_confusion_matrices(
    rows: Sequence[Dict[str, Any]],
    variants: Sequence[QuickVariant],
    output_dir: Path,
    cv_strategy: str,
) -> Path | None:
    """Save confusion matrices comparing all successful quick-run model types."""
    row_by_model = {str(row["model"]): row for row in rows if row.get("status") == "success"}
    variant_by_model = {variant.model_name: variant for variant in variants}
    model_names = [variant.model_name for variant in variants if variant.model_name in row_by_model]
    if not model_names:
        return None

    class_display_names: Dict[int, str] = {}
    collected: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_classes: List[np.ndarray] = []
    for model_name in model_names:
        suite_run_dir = Path(str(row_by_model[model_name]["suite_run_dir"]))
        trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir)
        if not class_display_names:
            class_display_names = _load_class_display_names(trainer_run_dir)
        y_true, y_pred = _collect_predictions_for_variant(
            trainer_run_dir=trainer_run_dir,
            cv_strategy=cv_strategy,
            summary_model_name=variant_by_model[model_name].summary_model_name,
        )
        if y_true.size == 0:
            continue
        collected[model_name] = (y_true, y_pred)
        all_classes.extend([y_true, y_pred])

    if not collected:
        return None

    classes = np.unique(np.concatenate(all_classes))
    tick_labels = [
        class_display_names.get(int(class_idx), str(int(class_idx)))
        for class_idx in classes
    ]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "confusion_matrices.png"

    n_models = len(collected)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models))
    if n_models == 1:
        axes = np.asarray([axes])

    for row_idx, (model_name, (y_true, y_pred)) in enumerate(collected.items()):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        abs_ax = axes[row_idx, 0]
        norm_ax = axes[row_idx, 1]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=abs_ax,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
        abs_ax.set_xlabel("predicted")
        abs_ax.set_ylabel("true")
        abs_ax.set_title(f"{model_name} - {cv_strategy} (absolute)")

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            ax=norm_ax,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
        norm_ax.set_xlabel("predicted")
        norm_ax.set_ylabel("true")
        norm_ax.set_title(f"{model_name} - {cv_strategy} (row-normalized)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    fixed_overrides = build_fixed_overrides(args, run_output_dir=output_dir)
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
    ranking_path = _save_group_model_ranking(summary=summary, output_dir=output_dir)
    confusion_path = None
    if not args.dry_run:
        confusion_path = _save_combined_confusion_matrices(
            rows=rows,
            variants=variants,
            output_dir=output_dir,
            cv_strategy=args.cv_strategy,
        )
    print(f"Saved quick comparison summary: {summary_path}")
    if ranking_path is not None:
        print(f"Saved ranking plot: {ranking_path}")
    if confusion_path is not None:
        print(f"Saved confusion matrices: {confusion_path}")
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
