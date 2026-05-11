"""Run a quick Table-6 arousal/valence comparison for GNN v1, GNN v2, and baselines.

This script generates focused suite-wrapper configs and optionally runs them
sequentially with the dataset, cross-validation, and training parameters from
the selected YAML suite config. By default it compares frozen
`Random`, `Majority`, frozen `GNN_v1`, current `GNN_v2`, and `LightGBM` on the
Table-6 three-class arousal and valence tasks with proper k-fold splitting. Requested
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
VALENCE_EXPERIMENT_ID = "multiclass_table6_valence_3class"
QUICK_EXPERIMENT_IDS = [AROUSAL_EXPERIMENT_ID, VALENCE_EXPERIMENT_ID]
EXPERIMENT_DISPLAY_NAMES = {
    AROUSAL_EXPERIMENT_ID: "Table-6 Arousal",
    VALENCE_EXPERIMENT_ID: "Table-6 Valence",
}
DEFAULT_MODELS = ["Random", "Majority", "GNN_v1", "GNN_v2", "LightGBM"]
BASELINE_MODELS = {"Random", "Majority", "Mean", "SVM", "LightGBM", "MLP"}
PREFERRED_MODEL_ORDER = ["Random", "Majority", "GNN_v1", "GNN_v2", "MLP"]
VALID_CV_STRATEGIES = {"subject_loo", "recording_loo", "recording_kfold", "subject_kfold"}
MODEL_ALIASES = {
    "random": "Random",
    "rand": "Random",
    "majority": "Majority",
    "maj": "Majority",
    "mean": "Mean",
    "svm": "SVM",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "gnn1": "GNN_v1",
    "gnn_v1": "GNN_v1",
    "gnn-v1": "GNN_v1",
    "v1": "GNN_v1",
    "gnn2": "GNN_v2",
    "gnn_v2": "GNN_v2",
    "gnn-v2": "GNN_v2",
    "v2": "GNN_v2",
}


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
        help=(
            "Comma-separated models: Random,Majority,GNN_v1,GNN_v2,Mean,SVM,LightGBM,MLP. "
            "Common lowercase aliases like random, majority, gnn1, gnn2, and lgbm are accepted."
        ),
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
        help=(
            "Optional CV strategy override. Use one strategy or a comma-separated list, "
            "for example subject_loo,recording_loo. By default, use the YAML config."
        ),
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
    raw_tokens = [token.strip() for token in raw_models.split(",") if token.strip()]
    models = [MODEL_ALIASES.get(token.lower(), token) for token in raw_tokens]
    allowed = {"GNN_v1", "GNN_v2"} | BASELINE_MODELS
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Allowed: {sorted(allowed)}")
    if not models:
        raise ValueError("At least one model must be requested.")
    return models


def _parse_cv_strategies(raw_strategies: str) -> List[str]:
    """Parse and validate one or more comma-separated CV strategies."""
    strategies = [token.strip() for token in raw_strategies.split(",") if token.strip()]
    unknown = sorted(set(strategies) - VALID_CV_STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown CV strategy/strategies: {unknown}. Allowed: {sorted(VALID_CV_STRATEGIES)}")
    if not strategies:
        raise ValueError("At least one CV strategy must be requested.")
    return strategies


def _ordered_models(model_names: Iterable[str]) -> List[str]:
    """Return model names in the preferred quick-comparison display order."""
    unique_models = list(dict.fromkeys(model_names))
    preferred_idx = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    return sorted(
        unique_models,
        key=lambda name: (preferred_idx.get(name, len(preferred_idx)), str(name).lower()),
    )


def _enable_quick_table6_tasks(wrapper_cfg: Dict[str, Any]) -> None:
    """Enable only the quick-comparison Table-6 tasks in a suite wrapper config."""
    experiments = wrapper_cfg.get("experiments")
    if isinstance(experiments, dict):
        for experiment_id, experiment_cfg in experiments.items():
            if isinstance(experiment_cfg, dict):
                experiment_cfg["enabled"] = experiment_id in QUICK_EXPERIMENT_IDS
        return
    if isinstance(experiments, list):
        for experiment_cfg in experiments:
            if isinstance(experiment_cfg, dict):
                experiment_cfg["enabled"] = str(experiment_cfg.get("id", "")) in QUICK_EXPERIMENT_IDS
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
                    # Quick runs favor robustness over throughput because some PyG/CUDA
                    # combinations intermittently fail in worker pin-memory threads.
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                }
            }
        },
    }
    global_overrides = overrides["global_overrides"]

    if args.seed is not None:
        overrides["suite"]["seed"] = int(args.seed)
        global_overrides.setdefault("cross_validation", {})["random_state"] = int(args.seed)
    if args.cv_strategy is not None:
        global_overrides.setdefault("cross_validation", {})["strategies"] = _parse_cv_strategies(str(args.cv_strategy))
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
    _enable_quick_table6_tasks(payload)
    return payload


def _build_run_payload(base_cfg: Dict[str, Any], fixed_overrides: Dict[str, Any], quick_run: QuickRun) -> Dict[str, Any]:
    """Merge base, fixed, and run-specific overrides."""
    payload = merge_many(base_cfg, fixed_overrides, quick_run.overrides)
    _enable_quick_table6_tasks(payload)
    return payload


def _get_cv_strategies(payload: Dict[str, Any]) -> List[str]:
    """Return configured CV strategies from the merged suite payload."""
    cv_cfg = payload.get("global_overrides", {}).get("cross_validation", {})
    strategies = cv_cfg.get("strategies")
    if isinstance(strategies, list) and strategies:
        return [str(strategy) for strategy in strategies]
    if isinstance(strategies, str) and strategies:
        return [strategies]
    raise ValueError("No cross-validation strategy configured in the merged YAML payload.")


def _get_enabled_experiment_ids(payload: Dict[str, Any]) -> List[str]:
    """Return enabled quick-comparison experiment IDs from a merged wrapper payload."""
    experiments = payload.get("experiments")
    enabled_ids: List[str] = []
    if isinstance(experiments, dict):
        for experiment_id, experiment_cfg in experiments.items():
            if (
                experiment_id in QUICK_EXPERIMENT_IDS
                and isinstance(experiment_cfg, dict)
                and bool(experiment_cfg.get("enabled", False))
            ):
                enabled_ids.append(experiment_id)
    elif isinstance(experiments, list):
        for experiment_cfg in experiments:
            if not isinstance(experiment_cfg, dict):
                continue
            experiment_id = str(experiment_cfg.get("id", ""))
            if experiment_id in QUICK_EXPERIMENT_IDS and bool(experiment_cfg.get("enabled", False)):
                enabled_ids.append(experiment_id)
    else:
        raise ValueError("Unsupported experiments format; expected dict or list.")
    if not enabled_ids:
        raise ValueError("No enabled quick-comparison Table-6 experiments found in merged wrapper payload.")
    return enabled_ids


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


def _load_subject_recording_usage(
    trainer_cfg: Dict[str, Any],
    base_config_path: Path,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Load total and configured-used subject/recording counts from the trainer CSV."""
    data_filepath_raw = trainer_cfg.get("dataset", {}).get("data_filepath")
    if not isinstance(data_filepath_raw, str):
        return None, None, None, None

    data_path = _resolve_repo_path(data_filepath_raw, base_config_path)
    if not data_path.exists():
        return None, None, None, None

    columns = pd.read_csv(data_path, nrows=0).columns.tolist()
    usecols = [column for column in ["subject", "recording"] if column in columns]
    if not usecols:
        return None, None, None, None

    data = pd.read_csv(data_path, usecols=usecols)
    subject_values = sorted(data["subject"].dropna().astype(str).unique().tolist()) if "subject" in data else []
    recording_values = sorted(data["recording"].dropna().astype(str).unique().tolist()) if "recording" in data else []

    dataset_cfg = trainer_cfg.get("dataset", {})
    filter_subjects = dataset_cfg.get("filter_subjects")
    exclude_subjects = dataset_cfg.get("exclude_subjects")
    filter_recordings = dataset_cfg.get("filter_recordings")

    used_subjects = set(subject_values)
    if filter_subjects is not None:
        used_subjects &= {str(value) for value in filter_subjects}
    if exclude_subjects is not None:
        used_subjects -= {str(value) for value in exclude_subjects}

    used_recordings = set(recording_values)
    if filter_recordings is not None:
        used_recordings &= {str(value) for value in filter_recordings}

    return (
        len(subject_values),
        len(recording_values),
        len(used_subjects),
        len(used_recordings),
    )


def _build_resolved_run_context(args: argparse.Namespace) -> Dict[str, str]:
    """Build human-readable resolved YAML context for startup logging."""
    base_config_path = Path(args.base_config)
    base_cfg = _load_yaml(base_config_path)
    fixed_overrides = build_fixed_overrides(args, run_output_dir=Path(args.output_root) / "<timestamp>")
    payload = merge_many(base_cfg, fixed_overrides)
    _enable_quick_table6_tasks(payload)

    trainer_base_cfg = _load_multiclass_base_config(base_cfg, base_config_path)
    trainer_cfg = merge_many(trainer_base_cfg, payload.get("global_overrides", {}))
    cv_cfg = trainer_cfg.get("cross_validation", {})
    dataset_cfg = trainer_cfg.get("dataset", {})
    gnn_training_cfg = trainer_cfg.get("gnn", {}).get("training", {})
    total_subjects, total_recordings, used_subjects, used_recordings = _load_subject_recording_usage(
        trainer_cfg,
        base_config_path,
    )

    return {
        "cv_strategies": ",".join(_get_cv_strategies(payload)),
        "n_splits": str(cv_cfg.get("n_splits", "not configured")),
        "val_size": str(cv_cfg.get("val_size", "not configured")),
        "num_epochs": str(gnn_training_cfg.get("num_epochs", "not configured")),
        "use_torch_compile": str(gnn_training_cfg.get("use_torch_compile", "not configured")),
        "num_workers": str(gnn_training_cfg.get("num_workers", "not configured")),
        "pin_memory": str(gnn_training_cfg.get("pin_memory", "not configured")),
        "persistent_workers": str(gnn_training_cfg.get("persistent_workers", "not configured")),
        "subjects": (
            f"{used_subjects}/{total_subjects}"
            if used_subjects is not None and total_subjects is not None
            else "unknown/unknown"
        ),
        "recordings": (
            f"{used_recordings}/{total_recordings}"
            if used_recordings is not None and total_recordings is not None
            else "unknown/unknown"
        ),
        "experiments": ",".join(_get_enabled_experiment_ids(payload)),
    }


def _collect_metrics(
    suite_run_dir: Path,
    cv_strategy: str,
    summary_model_name: str,
    experiment_id: str,
) -> Dict[str, float]:
    """Collect aggregate metrics from one suite run and one experiment."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Suite registry not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    row = registry[
        (registry["experiment_id"] == experiment_id)
        & (registry["status"] == "success")
    ]
    if row.empty:
        raise ValueError(f"No successful {experiment_id} run found in {registry_path}.")

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


def _resolve_trainer_run_dir(suite_run_dir: Path, experiment_id: str) -> Path:
    """Return the trainer run directory for one quick-comparison experiment."""
    registry_path = suite_run_dir / "suite_experiment_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Suite registry not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    row = registry[
        (registry["experiment_id"] == experiment_id)
        & (registry["status"] == "success")
    ]
    if row.empty:
        raise ValueError(f"No successful {experiment_id} run found in {registry_path}.")
    return Path(str(row.iloc[-1]["trainer_run_dir"]))


def _rows_to_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Build a stable summary dataframe from result rows."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    sort_cols = [col for col in ["status", "cv_strategy", "balanced_accuracy", "macro_f1", "model"] if col in df.columns]
    descending_metric_cols = {"balanced_accuracy", "macro_f1"}
    ascending = [col not in descending_metric_cols for col in sort_cols]
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
    strategy_order = (
        list(dict.fromkeys(plot_df["cv_strategy"].astype(str).tolist()))
        if "cv_strategy" in plot_df.columns
        else ["all"]
    )
    experiment_order = (
        list(dict.fromkeys(plot_df["experiment_id"].astype(str).tolist()))
        if "experiment_id" in plot_df.columns
        else ["all"]
    )
    long_df = plot_df.melt(
        id_vars=[col for col in ["experiment_id", "experiment_display_name", "cv_strategy", "model"] if col in plot_df.columns],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long_df.empty:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "classification_group_model_ranking.png"

    fig, axes = plt.subplots(
        len(experiment_order) * len(strategy_order),
        1,
        figsize=(12, max(4.5, 0.65 * len(model_order)) * len(experiment_order) * len(strategy_order)),
        squeeze=False,
    )
    row_idx = 0
    for experiment_id in experiment_order:
        experiment_df = (
            long_df[long_df["experiment_id"].astype(str) == experiment_id]
            if "experiment_id" in long_df.columns
            else long_df
        )
        experiment_name = EXPERIMENT_DISPLAY_NAMES.get(str(experiment_id), str(experiment_id))
        for strategy in strategy_order:
            ax = axes[row_idx, 0]
            strategy_df = (
                experiment_df[experiment_df["cv_strategy"].astype(str) == strategy]
                if "cv_strategy" in experiment_df.columns
                else experiment_df
            )
            sns.barplot(
                data=strategy_df,
                x="value",
                y="model",
                hue="metric",
                order=model_order,
                ax=ax,
                orient="h",
            )
            title_suffix = f" - {strategy}" if "cv_strategy" in experiment_df.columns else ""
            ax.set_title(f"Quick {experiment_name} Model Ranking{title_suffix}")
            ax.set_xlabel("metric value")
            ax.set_ylabel("model")
            ax.set_xlim(0.0, 1.0)
            ax.legend(loc="lower right", title="metric")
            row_idx += 1
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


def _collect_fold_targets_for_variant(
    trainer_run_dir: Path,
    cv_strategy: str,
    summary_model_name: str,
) -> Dict[str, np.ndarray]:
    """Collect test targets per fold for one model output directory."""
    strategy_dir = trainer_run_dir / cv_strategy
    if not strategy_dir.exists():
        raise FileNotFoundError(f"Strategy directory not found: {strategy_dir}")

    fold_targets: Dict[str, np.ndarray] = {}
    for fold_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
        if summary_model_name == "GNN":
            target_path = fold_dir / "test_targets.npy"
        else:
            target_path = fold_dir / "baselines" / summary_model_name / "test_targets.npy"
        if not target_path.exists():
            continue
        fold_targets[fold_dir.name] = np.asarray(np.load(target_path)).reshape(-1).astype(int)
    return fold_targets


def _build_label_distribution_tables(rows: Sequence[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-fold and aggregate label-distribution tables from saved test targets."""
    successful_rows = [row for row in rows if row.get("status") == "success"]
    pairs = list(
        dict.fromkeys(
            (str(row.get("experiment_id")), str(row.get("cv_strategy")))
            for row in successful_rows
        )
    )

    fold_records: List[Dict[str, Any]] = []
    aggregate_records: List[Dict[str, Any]] = []
    for experiment_id, cv_strategy in pairs:
        pair_rows = [
            row
            for row in successful_rows
            if str(row.get("experiment_id")) == experiment_id and str(row.get("cv_strategy")) == cv_strategy
        ]
        row_by_model = {str(row["model"]): row for row in pair_rows}
        source_row: Dict[str, Any] | None = None
        fold_targets: Dict[str, np.ndarray] = {}
        class_display_names: Dict[int, str] = {}

        for model_name in _ordered_models(row_by_model.keys()):
            candidate = row_by_model[model_name]
            try:
                suite_run_dir = Path(str(candidate["suite_run_dir"]))
                trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir, experiment_id=experiment_id)
                fold_targets = _collect_fold_targets_for_variant(
                    trainer_run_dir=trainer_run_dir,
                    cv_strategy=cv_strategy,
                    summary_model_name=str(candidate["summary_model_name"]),
                )
                if fold_targets:
                    source_row = candidate
                    class_display_names = _load_class_display_names(trainer_run_dir)
                    break
            except Exception:
                continue

        if source_row is None or not fold_targets:
            continue

        observed_classes = sorted(
            {
                int(class_idx)
                for targets in fold_targets.values()
                for class_idx in np.unique(targets).tolist()
            }
        )
        metadata_classes = sorted(class_display_names.keys())
        classes = sorted(set(observed_classes) | set(metadata_classes))
        if not classes:
            continue

        aggregate_counts = {class_idx: 0 for class_idx in classes}
        for fold_name, targets in fold_targets.items():
            total = int(targets.size)
            counts = {class_idx: int(np.sum(targets == class_idx)) for class_idx in classes}
            for class_idx, count in counts.items():
                aggregate_counts[class_idx] += count
                fold_records.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                        "cv_strategy": cv_strategy,
                        "source_model": str(source_row["model"]),
                        "split": "test",
                        "fold": fold_name,
                        "class_index": class_idx,
                        "class_name": class_display_names.get(class_idx, str(class_idx)),
                        "count": count,
                        "total": total,
                        "proportion": float(count / total) if total else np.nan,
                    }
                )

        aggregate_total = int(sum(aggregate_counts.values()))
        for class_idx, count in aggregate_counts.items():
            aggregate_records.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                    "cv_strategy": cv_strategy,
                    "source_model": str(source_row["model"]),
                    "split": "test",
                    "class_index": class_idx,
                    "class_name": class_display_names.get(class_idx, str(class_idx)),
                    "count": count,
                    "total": aggregate_total,
                    "proportion": float(count / aggregate_total) if aggregate_total else np.nan,
                }
            )

    return pd.DataFrame(fold_records), pd.DataFrame(aggregate_records)


def _plot_label_distribution(
    aggregate_distribution: pd.DataFrame,
    output_dir: Path,
    y_column: str,
    output_name: str,
    ylabel: str,
) -> Path | None:
    """Save one aggregate class-balance plot."""
    if aggregate_distribution.empty or y_column not in aggregate_distribution.columns:
        return None

    plot_df = aggregate_distribution.copy()
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_column])
    if plot_df.empty:
        return None

    group_keys = ["experiment_id", "cv_strategy"]
    groups = list(plot_df[group_keys].drop_duplicates().itertuples(index=False, name=None))
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / output_name

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(10, max(3.5, 2.8 * len(groups))),
        squeeze=False,
    )
    for idx, (experiment_id, cv_strategy) in enumerate(groups):
        ax = axes[idx, 0]
        group_df = plot_df[
            (plot_df["experiment_id"] == experiment_id)
            & (plot_df["cv_strategy"] == cv_strategy)
        ].copy()
        sns.barplot(
            data=group_df,
            x="class_name",
            y=y_column,
            color="#4C78A8",
            ax=ax,
        )
        experiment_name = EXPERIMENT_DISPLAY_NAMES.get(str(experiment_id), str(experiment_id))
        ax.set_title(f"{experiment_name} Label Distribution - {cv_strategy}")
        ax.set_xlabel("class")
        ax.set_ylabel(ylabel)
        if y_column == "proportion":
            ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_label_distribution_outputs(rows: Sequence[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Save class-balance CSV tables and aggregate distribution figures."""
    fold_distribution, aggregate_distribution = _build_label_distribution_tables(rows)
    if fold_distribution.empty and aggregate_distribution.empty:
        return []

    saved_paths: List[Path] = []
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not fold_distribution.empty:
        fold_path = tables_dir / "label_distribution_by_fold.csv"
        fold_distribution.to_csv(fold_path, index=False)
        saved_paths.append(fold_path)
    if not aggregate_distribution.empty:
        aggregate_path = tables_dir / "label_distribution_aggregate.csv"
        aggregate_distribution.to_csv(aggregate_path, index=False)
        saved_paths.append(aggregate_path)

    for plot_path in [
        _plot_label_distribution(
            aggregate_distribution=aggregate_distribution,
            output_dir=output_dir,
            y_column="count",
            output_name="label_distribution_counts.png",
            ylabel="count",
        ),
        _plot_label_distribution(
            aggregate_distribution=aggregate_distribution,
            output_dir=output_dir,
            y_column="proportion",
            output_name="label_distribution_proportions.png",
            ylabel="proportion",
        ),
    ]:
        if plot_path is not None:
            saved_paths.append(plot_path)

    return saved_paths


def _save_combined_confusion_matrices(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    experiment_id: str,
    cv_strategy: str,
    use_strategy_suffix: bool = True,
) -> Path | None:
    """Save confusion matrices comparing all successful quick-run model types."""
    row_by_model = {
        str(row["model"]): row
        for row in rows
        if row.get("status") == "success"
        and str(row.get("experiment_id")) == experiment_id
        and str(row.get("cv_strategy")) == cv_strategy
    }
    model_names = _ordered_models(row_by_model.keys())
    if not model_names:
        return None

    class_display_names: Dict[int, str] = {}
    collected: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_classes: List[np.ndarray] = []
    for model_name in model_names:
        suite_run_dir = Path(str(row_by_model[model_name]["suite_run_dir"]))
        trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir, experiment_id=experiment_id)
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
    experiment_slug = str(experiment_id).replace("multiclass_", "").replace("_3class", "")
    if use_strategy_suffix:
        filename = f"confusion_matrices_{experiment_slug}_{cv_strategy}.png"
    else:
        filename = f"confusion_matrices_{experiment_slug}.png"
    output_path = figures_dir / filename

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
        cv_strategies = _get_cv_strategies(payload)
        experiment_ids = _get_enabled_experiment_ids(payload)
        config_path = generated_dir / f"{quick_run.run_name}.yaml"
        _write_yaml(config_path, payload)

        base_row: Dict[str, Any] = {
            "run_name": quick_run.run_name,
            "description": quick_run.description,
            "wrapper_config_path": str(config_path),
            "suite_run_dir": "",
            "runtime_seconds": np.nan,
            "error": "",
        }
        if args.dry_run:
            for experiment_id in experiment_ids:
                for cv_strategy in cv_strategies:
                    for model_name in quick_run.model_names:
                        rows.append(
                            {
                                **base_row,
                                "experiment_id": experiment_id,
                                "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                                "cv_strategy": cv_strategy,
                                "model": model_name,
                                "summary_model_name": quick_run.summary_model_names[model_name],
                                "status": "dry_run",
                            }
                        )
            print(
                f"dry-run | {quick_run.run_name} | models={','.join(quick_run.model_names)} "
                f"| tasks={','.join(experiment_ids)} | cv={','.join(cv_strategies)} | {config_path}"
            )
            continue

        started = time.time()
        try:
            suite_run_dir = Path(run_suite(str(config_path)))
            runtime_seconds = round(time.time() - started, 3)
            for experiment_id in experiment_ids:
                for cv_strategy in cv_strategies:
                    for model_name in quick_run.model_names:
                        row: Dict[str, Any] = {
                            **base_row,
                            "experiment_id": experiment_id,
                            "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                            "cv_strategy": cv_strategy,
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
                    for model_name in quick_run.model_names:
                        rows.append(
                            {
                                **base_row,
                                "experiment_id": experiment_id,
                                "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                                "cv_strategy": cv_strategy,
                                "model": model_name,
                                "summary_model_name": quick_run.summary_model_names[model_name],
                                "runtime_seconds": runtime_seconds,
                                "status": "failed",
                                "error": f"{exc}\n{traceback.format_exc()}",
                            }
                        )

        for row in rows[-(len(quick_run.model_names) * len(cv_strategies) * len(experiment_ids)) :]:
            print(
                f"{row['status']} | {row.get('experiment_id')} | {row['cv_strategy']} | {row['model']} "
                f"| balanced_accuracy={row.get('balanced_accuracy', np.nan)}"
            )

    summary = _rows_to_dataframe(rows)
    summary_path = output_dir / "quick_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    ranking_path = _save_group_model_ranking(summary=summary, output_dir=output_dir)
    confusion_paths: List[Path] = []
    label_distribution_paths: List[Path] = []
    if not args.dry_run:
        label_distribution_paths = _save_label_distribution_outputs(rows=rows, output_dir=output_dir)
        successful_pairs = list(
            dict.fromkeys(
                (
                    str(row["experiment_id"]),
                    str(row["cv_strategy"]),
                )
                for row in rows
                if str(row.get("status")) == "success"
            )
        )
        successful_strategies = [cv_strategy for _, cv_strategy in successful_pairs]
        use_strategy_suffix = len(set(successful_strategies)) > 1 or len(successful_pairs) > 1
        for experiment_id, cv_strategy in successful_pairs:
            confusion_path = _save_combined_confusion_matrices(
                rows=rows,
                output_dir=output_dir,
                experiment_id=experiment_id,
                cv_strategy=cv_strategy,
                use_strategy_suffix=use_strategy_suffix,
            )
            if confusion_path is not None:
                confusion_paths.append(confusion_path)
    print(f"Saved quick comparison summary: {summary_path}")
    if ranking_path is not None:
        print(f"Saved ranking plot: {ranking_path}")
    for label_distribution_path in label_distribution_paths:
        print(f"Saved label distribution output: {label_distribution_path}")
    for confusion_path in confusion_paths:
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
