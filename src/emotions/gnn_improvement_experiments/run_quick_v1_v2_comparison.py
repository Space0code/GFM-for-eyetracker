"""Run a quick Table-6 arousal comparison for GNN v1, GNN v2, and baselines.

This script generates focused suite-wrapper configs and optionally runs them
sequentially with the dataset, cross-validation, and training parameters from
the selected YAML suite config. By default it compares frozen
`Random`, `Majority`, frozen `GNN_v1`, current `GNN_v2`, and `LightGBM` on the
Table-6 three-class arousal task with proper k-fold splitting. Requested
baseline models are grouped into one suite invocation so they share the same
loaded dataset and CV splits.

Example:
  python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py

Useful options:
  python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py \
      --models Random,Majority,GNN_v1,GNN_v2,MLP,LightGBM,SVM \
      --num-epochs 20 \
      --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
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
DEFAULT_MODELS = ["Random", "Majority", "GNN_v1", "GNN_v2", "LightGBM"]
BASELINE_MODELS = {"Random", "Majority", "Mean", "SVM", "LightGBM", "MLP"}
PREFERRED_MODEL_ORDER = ["Random", "Majority", "GNN_v1", "GNN_v2", "MLP"]


@dataclass(frozen=True)
class QuickVariant:
    """One quick-comparison model variant."""

    model_name: str
    description: str
    overrides: Dict[str, Any]
    summary_model_name: str


@dataclass(frozen=True)
class QuickRun:
    """One suite invocation that can produce one or more model results."""

    run_name: str
    description: str
    model_names: List[str]
    summary_model_names: Dict[str, str]
    overrides: Dict[str, Any]


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
        help="Comma-separated models: Random,Majority,GNN_v1,GNN_v2,Mean,SVM,LightGBM,MLP.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Optional k-fold split override. By default, use the YAML config.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Optional validation-group override. By default, use the YAML config.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional GNN epoch override. By default, use the YAML config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional suite/CV seed override. By default, use the YAML config.",
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default=None,
        choices=["recording_kfold", "subject_kfold"],
        help="Optional CV strategy override. By default, use the YAML config.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help=(
            "Enable torch.compile for GNN quick runs. Disabled by default because "
            "dynamic PyG graph batches can trigger PyTorch Inductor failures."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only.")
    return parser.parse_args()


class _TeeStream:
    """Write console output to both the original stream and a log file."""

    def __init__(self, stream: TextIO, log_handle: TextIO) -> None:
        self.stream = stream
        self.log_handle = log_handle

    def write(self, message: str) -> int:
        self.stream.write(message)
        self.log_handle.write(message)
        self.log_handle.flush()
        return len(message)

    def flush(self) -> None:
        self.stream.flush()
        self.log_handle.flush()


@contextlib.contextmanager
def _tee_output(log_path: Path) -> Iterator[None]:
    """Mirror stdout and stderr to a command-level log file."""
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


def _ordered_models(model_names: Iterable[str]) -> List[str]:
    """Return model names in the preferred quick-comparison display order."""
    unique_models = list(dict.fromkeys(model_names))
    preferred_idx = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    return sorted(
        unique_models,
        key=lambda name: (preferred_idx.get(name, len(preferred_idx)), str(name).lower()),
    )


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
    """Build optional command-level overrides shared by all variants."""
    results_dir = (
        run_output_dir / "model_runs"
        if run_output_dir is not None
        else Path(args.output_root) / "suite_runs"
    )
    overrides: Dict[str, Any] = {
        "suite": {
            "results_dir": str(results_dir),
        },
        "global_overrides": {
            "gnn": {
                "training": {
                    "use_torch_compile": bool(getattr(args, "use_torch_compile", False)),
                }
            }
        },
    }
    global_overrides = overrides["global_overrides"]

    if args.seed is not None:
        overrides["suite"]["seed"] = int(args.seed)
        global_overrides.setdefault("cross_validation", {})["random_state"] = int(args.seed)
    if args.cv_strategy is not None:
        global_overrides.setdefault("cross_validation", {})["strategies"] = [str(args.cv_strategy)]
    if args.n_splits is not None:
        global_overrides.setdefault("cross_validation", {})["n_splits"] = int(args.n_splits)
    if args.val_size is not None:
        global_overrides.setdefault("cross_validation", {})["val_size"] = int(args.val_size)
    if args.num_epochs is not None:
        global_overrides.setdefault("gnn", {}).setdefault("training", {})["num_epochs"] = int(args.num_epochs)

    return overrides


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


def build_quick_runs(model_names: Sequence[str]) -> List[QuickRun]:
    """Group requested models into the minimum safe set of suite invocations."""
    baseline_names = [name for name in model_names if name in BASELINE_MODELS]
    emitted_baseline_run = False
    runs: List[QuickRun] = []

    for model_name in model_names:
        if model_name in BASELINE_MODELS:
            if emitted_baseline_run:
                continue
            emitted_baseline_run = True
            runs.append(
                QuickRun(
                    run_name="Baselines",
                    description=f"Baselines on the same quick subset and folds: {', '.join(baseline_names)}.",
                    model_names=baseline_names,
                    summary_model_names={name: name for name in baseline_names},
                    overrides={
                        "global_overrides": {
                            "run_experiments": {"baselines": True, "gnn": False},
                            "baselines": {"models": baseline_names},
                        }
                    },
                )
            )
            continue

        variant = build_variant(model_name)
        runs.append(
            QuickRun(
                run_name=variant.model_name,
                description=variant.description,
                model_names=[variant.model_name],
                summary_model_names={variant.model_name: variant.summary_model_name},
                overrides=variant.overrides,
            )
        )

    return runs


def _build_payload(base_cfg: Dict[str, Any], fixed_overrides: Dict[str, Any], variant: QuickVariant) -> Dict[str, Any]:
    """Merge base, fixed, and variant-specific overrides."""
    payload = merge_many(base_cfg, fixed_overrides, variant.overrides)
    _enable_only_arousal(payload)
    return payload


def _build_run_payload(base_cfg: Dict[str, Any], fixed_overrides: Dict[str, Any], quick_run: QuickRun) -> Dict[str, Any]:
    """Merge base, fixed, and run-specific overrides."""
    payload = merge_many(base_cfg, fixed_overrides, quick_run.overrides)
    _enable_only_arousal(payload)
    return payload


def _get_primary_cv_strategy(payload: Dict[str, Any]) -> str:
    """Return the first configured CV strategy from the merged suite payload."""
    cv_cfg = payload.get("global_overrides", {}).get("cross_validation", {})
    strategies = cv_cfg.get("strategies")
    if isinstance(strategies, list) and strategies:
        return str(strategies[0])
    if isinstance(strategies, str) and strategies:
        return strategies
    raise ValueError("No cross-validation strategy configured in the merged YAML payload.")


def _resolve_repo_path(path_value: str, anchor_path: Path | None = None) -> Path:
    """Resolve a path from the current working directory or a config-file parent."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if anchor_path is not None:
        return (anchor_path.parent / path).resolve()
    return path.resolve()


def _load_multiclass_base_config(base_cfg: Dict[str, Any], base_config_path: Path) -> Dict[str, Any]:
    """Load the multiclass trainer base config referenced by the suite YAML."""
    base_configs = base_cfg.get("suite", {}).get("base_configs", {})
    multiclass_config_raw = base_configs.get("multiclass")
    if not isinstance(multiclass_config_raw, str):
        return {}

    multiclass_config_path = _resolve_repo_path(multiclass_config_raw, base_config_path)
    if not multiclass_config_path.exists():
        return {}

    return _load_yaml(multiclass_config_path)


def _load_all_subject_recording_counts(trainer_cfg: Dict[str, Any], base_config_path: Path) -> tuple[int | None, int | None]:
    """Load total unique subject/recording counts from the configured trainer CSV."""
    data_filepath_raw = trainer_cfg.get("dataset", {}).get("data_filepath")
    if not isinstance(data_filepath_raw, str):
        return None, None

    data_path = _resolve_repo_path(data_filepath_raw, base_config_path)
    if not data_path.exists():
        return None, None

    columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    usecols = [column for column in ["subject", "recording"] if column in columns]
    if not usecols:
        return None, None

    data = pd.read_csv(data_path, usecols=usecols)
    subject_count = int(data["subject"].dropna().astype(str).nunique()) if "subject" in data else None
    recording_count = int(data["recording"].dropna().astype(str).nunique()) if "recording" in data else None
    return subject_count, recording_count


def _format_used_total(filter_values: Any, total_count: int | None) -> str:
    """Format configured filter usage as used/total."""
    if total_count is None:
        total_text = "unknown"
    else:
        total_text = str(total_count)

    if filter_values is None:
        used_text = total_text
    elif isinstance(filter_values, (list, tuple, set)):
        used_text = str(len(filter_values))
    else:
        used_text = "1"
    return f"{used_text}/{total_text}"


def _build_resolved_run_context(args: argparse.Namespace) -> Dict[str, str]:
    """Build human-readable resolved YAML context for startup logging."""
    base_config_path = Path(args.base_config)
    base_cfg = _load_yaml(base_config_path)
    fixed_overrides = build_fixed_overrides(args, run_output_dir=Path(args.output_root) / "<timestamp>")
    payload = merge_many(base_cfg, fixed_overrides)
    _enable_only_arousal(payload)

    trainer_base_cfg = _load_multiclass_base_config(base_cfg, base_config_path)
    trainer_cfg = merge_many(trainer_base_cfg, payload.get("global_overrides", {}))
    cv_cfg = trainer_cfg.get("cross_validation", {})
    dataset_cfg = trainer_cfg.get("dataset", {})
    gnn_training_cfg = trainer_cfg.get("gnn", {}).get("training", {})
    total_subjects, total_recordings = _load_all_subject_recording_counts(trainer_cfg, base_config_path)

    return {
        "cv_strategy": _get_primary_cv_strategy(payload),
        "n_splits": str(cv_cfg.get("n_splits", "not configured")),
        "val_size": str(cv_cfg.get("val_size", "not configured")),
        "num_epochs": str(gnn_training_cfg.get("num_epochs", "not configured")),
        "use_torch_compile": str(gnn_training_cfg.get("use_torch_compile", "not configured")),
        "subjects": _format_used_total(dataset_cfg.get("filter_subjects"), total_subjects),
        "recordings": _format_used_total(dataset_cfg.get("filter_recordings"), total_recordings),
    }


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

    model_order = _ordered_models(plot_df["model"].astype(str).tolist())
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
    sns.barplot(
        data=long_df,
        x="value",
        y="model",
        hue="metric",
        order=model_order,
        ax=ax,
        orient="h",
    )
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
    output_dir: Path,
    cv_strategy: str,
) -> Path | None:
    """Save confusion matrices comparing all successful quick-run model types."""
    row_by_model = {str(row["model"]): row for row in rows if row.get("status") == "success"}
    model_names = _ordered_models(row_by_model.keys())
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
            summary_model_name=str(row_by_model[model_name]["summary_model_name"]),
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


def run_quick_comparison(args: argparse.Namespace, output_dir: Path | None = None) -> Path:
    """Run or generate the quick comparison configs and summary."""
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(args.output_root) / timestamp
    generated_dir = output_dir / "generated_wrapper_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(base_config_path)
    fixed_overrides = build_fixed_overrides(args, run_output_dir=output_dir)
    model_names = _parse_models(args.models)
    quick_runs = build_quick_runs(model_names)

    rows: List[Dict[str, Any]] = []
    for quick_run in quick_runs:
        payload = _build_run_payload(base_cfg=base_cfg, fixed_overrides=fixed_overrides, quick_run=quick_run)
        cv_strategy = _get_primary_cv_strategy(payload)
        config_path = generated_dir / f"{quick_run.run_name}.yaml"
        _write_yaml(config_path, payload)

        base_row: Dict[str, Any] = {
            "run_name": quick_run.run_name,
            "description": quick_run.description,
            "wrapper_config_path": str(config_path),
            "suite_run_dir": "",
            "cv_strategy": cv_strategy,
            "runtime_seconds": np.nan,
            "error": "",
        }
        if args.dry_run:
            for model_name in quick_run.model_names:
                rows.append(
                    {
                        **base_row,
                        "model": model_name,
                        "summary_model_name": quick_run.summary_model_names[model_name],
                        "status": "dry_run",
                    }
                )
            print(f"dry-run | {quick_run.run_name} | models={','.join(quick_run.model_names)} | {config_path}")
            continue

        started = time.time()
        try:
            suite_run_dir = Path(run_suite(str(config_path)))
            runtime_seconds = round(time.time() - started, 3)
            for model_name in quick_run.model_names:
                row: Dict[str, Any] = {
                    **base_row,
                    "model": model_name,
                    "summary_model_name": quick_run.summary_model_names[model_name],
                    "suite_run_dir": str(suite_run_dir),
                    "runtime_seconds": runtime_seconds,
                    "status": "success",
                }
                try:
                    row.update(
                        _collect_metrics(
                            suite_run_dir=suite_run_dir,
                            cv_strategy=cv_strategy,
                            summary_model_name=quick_run.summary_model_names[model_name],
                        )
                    )
                except Exception as metric_exc:
                    row["status"] = "failed"
                    row["error"] = f"{metric_exc}\n{traceback.format_exc()}"
                rows.append(row)
        except Exception as exc:
            runtime_seconds = round(time.time() - started, 3)
            for model_name in quick_run.model_names:
                rows.append(
                    {
                        **base_row,
                        "model": model_name,
                        "summary_model_name": quick_run.summary_model_names[model_name],
                        "runtime_seconds": runtime_seconds,
                        "status": "failed",
                        "error": f"{exc}\n{traceback.format_exc()}",
                    }
                )

        for row in rows[-len(quick_run.model_names) :]:
            print(f"{row['status']} | {row['model']} | balanced_accuracy={row.get('balanced_accuracy', np.nan)}")

    summary = _rows_to_dataframe(rows)
    summary_path = output_dir / "quick_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    ranking_path = _save_group_model_ranking(summary=summary, output_dir=output_dir)
    confusion_path = None
    if not args.dry_run:
        confusion_cv_strategy = str(rows[0]["cv_strategy"]) if rows else _get_primary_cv_strategy(base_cfg)
        confusion_path = _save_combined_confusion_matrices(
            rows=rows,
            output_dir=output_dir,
            cv_strategy=confusion_cv_strategy,
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "quick_v1_v2_comparison.log"

    with _tee_output(log_path):
        print(f"Logging quick comparison output to: {log_path}")
        print("Quick v1/v2 comparison arguments:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        resolved_context = _build_resolved_run_context(args)
        print("Resolved YAML run context:")
        for key, value in resolved_context.items():
            print(f"  {key}: {value}")
        run_quick_comparison(args, output_dir=output_dir)


if __name__ == "__main__":
    main()
