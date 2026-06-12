"""Run a quick Table-6 low/high valence comparison for graph models and baselines.

This script generates focused suite-wrapper configs and optionally runs them
sequentially with the dataset, cross-validation, models, and training parameters
from the selected YAML suite config. The default wrapper config uses the
low/high binary-style Table-6 valence target and all four thesis signal sets:
gaze-only, pupil-only, gaze+pupil, and all signals. Select models and signal
sets in the wrapper config with `quick_comparison.models` and
`quick_comparison.signal_sets`, or override them from the command line with
`--models` and `--signal-sets`. Requested baseline models are grouped into one
suite invocation so they share the same loaded dataset and CV splits.

Example:
  python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py

Useful options:
  python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py \
      --models Random,Majority,BasicGCN,HeteroGCNMean,HeteroGCNMLP,HeteroGCNMLPWeights,GazeMAE_MLP,MLP,LightGBM,SVM \
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
from sklearn.metrics import confusion_matrix, log_loss

# Add src directory only for direct script execution.
if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from emotions.common.model_benchmarking import load_benchmark_records, summarize_benchmark_records
from emotions.moment_baseline import MOMENT_MODEL_NAMES
from emotions.suite.config_merge import merge_many
from emotions.suite.run_hci_experiment_suite import run_suite
from emotions.utils import TimestampedLineWriter


AROUSAL_EXPERIMENT_ID = "multiclass_table6_arousal_3class"
VALENCE_EXPERIMENT_ID = "multiclass_table6_valence_3class"
QUICK_EXPERIMENT_IDS = [AROUSAL_EXPERIMENT_ID, VALENCE_EXPERIMENT_ID]
DEFAULT_QUICK_EXPERIMENT_IDS = [VALENCE_EXPERIMENT_ID]
TABLE6_TASK_TO_EXPERIMENT_ID = {
    "arousal": AROUSAL_EXPERIMENT_ID,
    "table6-arousal-3class": AROUSAL_EXPERIMENT_ID,
    AROUSAL_EXPERIMENT_ID: AROUSAL_EXPERIMENT_ID,
    AROUSAL_EXPERIMENT_ID.replace("_", "-"): AROUSAL_EXPERIMENT_ID,
    "valence": VALENCE_EXPERIMENT_ID,
    "table6-valence-3class": VALENCE_EXPERIMENT_ID,
    VALENCE_EXPERIMENT_ID: VALENCE_EXPERIMENT_ID,
    VALENCE_EXPERIMENT_ID.replace("_", "-"): VALENCE_EXPERIMENT_ID,
}
EXPERIMENT_DISPLAY_NAMES = {
    AROUSAL_EXPERIMENT_ID: "Table-6 Arousal",
    VALENCE_EXPERIMENT_ID: "Table-6 Valence",
}
DEFAULT_MODELS = ["Random", "Majority", "BasicGCN", "HeteroGCNMean", "HeteroGCNMLP", "HeteroGCNMLPWeights", "LightGBM"]
BASELINE_MODELS = {"Random", "Majority", "Mean", "SVM", "LightGBM", "MLP", "GazeMAE_MLP"} | MOMENT_MODEL_NAMES
GNN_MODELS = {"BasicGCN", "HeteroGCNMean", "HeteroGCNMLP", "HeteroGCNMLPWeights"}
PREFERRED_MODEL_ORDER = [
    "Random",
    "Majority",
    "BasicGCN",
    "HeteroGCNMean",
    "HeteroGCNMLP",
    "HeteroGCNMLPWeights",
    "GazeMAE_MLP",
    "MOMENT_gaze",
    "MOMENT_pupil",
    "MOMENT_gaze_pupil",
    "MOMENT_all_signals",
    "MOMENT_GazeMAE_gaze_pupil",
    "MOMENT_GazeMAE_all_signals",
    "MLP",
    "LightGBM",
    "SVM",
]
VALID_CV_STRATEGIES = {"subject_loo", "recording_loo", "recording_kfold", "subject_kfold"}
REPORT_PROFILES = {"thesis", "full", "minimal"}
SIGNAL_SET_ORDER = ["gaze_only", "pupil_only", "gaze_pupil", "all_signals"]
SIGNAL_SET_DESCRIPTIONS = {
    "gaze_only": "Gaze x/y nodes with temporal and spatial graph relations.",
    "pupil_only": "Pupil left/right nodes with temporal graph relations only.",
    "gaze_pupil": "Gaze x/y plus pupil nodes with temporal and spatial graph relations.",
    "all_signals": "Gaze, pupil, screen distance, and fixation-duration nodes with all graph relations.",
}
SIGNAL_SET_ALIASES = {
    "gaze": "gaze_only",
    "gaze_only": "gaze_only",
    "gaze-only": "gaze_only",
    "pupil": "pupil_only",
    "pupil_only": "pupil_only",
    "pupil-only": "pupil_only",
    "gaze+pupil": "gaze_pupil",
    "gaze_pupil": "gaze_pupil",
    "gaze-pupil": "gaze_pupil",
    "all": "all_signals",
    "full": "all_signals",
    "all_signals": "all_signals",
    "all-signals": "all_signals",
}
FROZEN_EMBEDDING_MODELS = {"GazeMAE_MLP"} | MOMENT_MODEL_NAMES
SIGNAL_SET_EMBEDDING_MODEL = {
    "gaze_only": "GazeMAE_MLP",
    "pupil_only": "MOMENT_pupil",
    "gaze_pupil": "MOMENT_GazeMAE_gaze_pupil",
    "all_signals": "MOMENT_GazeMAE_all_signals",
}
METADATA_DROPNA_COLUMNS = ["time-rel-seconds", "subject", "recording"]
GAZE_RAW_COLUMNS = ["x-avg", "y-avg"]
PUPIL_RAW_COLUMNS = ["pupil-size-left-avg", "pupil-size-right-avg"]
SIGNAL_OUTLIER_FILTER_DEFAULTS = {
    "enabled": True,
    "lower_quantile": 0.01,
    "upper_quantile": 0.99,
}
MODEL_COLOR_PALETTE = {
    "Random": "#4C72B0",
    "Majority": "#DD8452",
    "BasicGCN": "#64B5CD",
    "HeteroGCNMean": "#55A868",
    "HeteroGCNMLP": "#4C9BB0",
    "HeteroGCNMLPWeights": "#C44E52",
    "GazeMAE_MLP": "#8172B3",
    "MOMENT_gaze": "#44AA99",
    "MOMENT_pupil": "#88CCEE",
    "MOMENT_gaze_pupil": "#117733",
    "MOMENT_all_signals": "#999933",
    "MOMENT_GazeMAE_gaze_pupil": "#AA4499",
    "MOMENT_GazeMAE_all_signals": "#CC6677",
    "MLP": "#937860",
    "LightGBM": "#DA8BC3",
    "SVM": "#8C8C8C",
    "Mean": "#CCB974",
}
THESIS_MODEL_DISPLAY_NAMES = {
    "Random": "Random",
    "Majority": "Majority",
    "Mean": "Mean",
    "SVM": "SVM",
    "LightGBM": "LightGBM",
    "MLP": "MLP",
    "GazeMAE_MLP": "GazeMAE_MLP",
    "MOMENT_gaze": "MOMENT_Gaze",
    "MOMENT_pupil": "MOMENT_Pupil",
    "MOMENT_gaze_pupil": "MOMENT_Gaze_Pupil",
    "MOMENT_all_signals": "MOMENT_All_Signals",
    "MOMENT_GazeMAE_gaze_pupil": "MOMENT_GazeMAE_Gaze_Pupil",
    "MOMENT_GazeMAE_all_signals": "MOMENT_GazeMAE_All_Signals",
    "BasicGCN": "Basic_GCN",
    "HeteroGCNMean": "Hetero_GCN_Mean",
    "HeteroGCNMLP": "Hetero_GCN_MLP",
    "HeteroGCNMLPWeights": "Hetero_GCN_MLP_Weights",
}
LOSS_METRIC_COLUMNS = ["train_loss", "val_loss", "test_loss"]
LOSS_AXIS_UPPER_LIMIT = 2.0
LOSS_METRIC_STYLES = {
    "train_loss": {"label": "Train loss", "color": "#0072B2", "linestyle": "-"},
    "val_loss": {"label": "Validation loss", "color": "#E69F00", "linestyle": "-"},
    "test_loss": {"label": "Held-out test loss", "color": "#666666", "linestyle": "--"},
}
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
    "gazemae": "GazeMAE_MLP",
    "gazemae_mlp": "GazeMAE_MLP",
    "gazemae-mlp": "GazeMAE_MLP",
    "gaze_mae": "GazeMAE_MLP",
    "gaze_mae_mlp": "GazeMAE_MLP",
    "moment": "MOMENT_all_signals",
    "moment_mlp": "MOMENT_all_signals",
    "moment_gaze": "MOMENT_gaze",
    "moment-gaze": "MOMENT_gaze",
    "moment_pupil": "MOMENT_pupil",
    "moment-pupil": "MOMENT_pupil",
    "moment_gaze_pupil": "MOMENT_gaze_pupil",
    "moment-gaze-pupil": "MOMENT_gaze_pupil",
    "moment_gaze+pupil": "MOMENT_gaze_pupil",
    "moment_all": "MOMENT_all_signals",
    "moment_all_signals": "MOMENT_all_signals",
    "moment-all-signals": "MOMENT_all_signals",
    "moment_gazemae_gaze_pupil": "MOMENT_GazeMAE_gaze_pupil",
    "moment-gazemae-gaze-pupil": "MOMENT_GazeMAE_gaze_pupil",
    "moment_gazemae_all": "MOMENT_GazeMAE_all_signals",
    "moment_gazemae_all_signals": "MOMENT_GazeMAE_all_signals",
    "moment-gazemae-all-signals": "MOMENT_GazeMAE_all_signals",
    "basicgcn": "BasicGCN",
    "basic_gcn": "BasicGCN",
    "basic-gcn": "BasicGCN",
    "gcn": "BasicGCN",
    "heterogcnmean": "HeteroGCNMean",
    "hetero_gcn_mean": "HeteroGCNMean",
    "hetero-gcn-mean": "HeteroGCNMean",
    "heterogcn_mean": "HeteroGCNMean",
    "heterogcnmlp": "HeteroGCNMLP",
    "hetero_gcn_mlp": "HeteroGCNMLP",
    "hetero-gcn-mlp": "HeteroGCNMLP",
    "heterogcn_mlp": "HeteroGCNMLP",
    "heterogcnmlpweights": "HeteroGCNMLPWeights",
    "hetero_gcn_mlp_weights": "HeteroGCNMLPWeights",
    "hetero-gcn-mlp-weights": "HeteroGCNMLPWeights",
    "heterogcn_mlp_weights": "HeteroGCNMLPWeights",
    "weighted_heterogcn": "HeteroGCNMLPWeights",
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


@dataclass(frozen=True)
class SignalSetVariant:
    """One signal-subset experiment axis value."""

    name: str
    description: str
    overrides: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run quick GNN architecture comparison")
    parser.add_argument(
        "--base-config",
        type=str,
        default=(
            "src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/"
            "run_hci_experiment_suite_table6_low_high.yaml"
        ),
        help="Base suite wrapper config. Defaults to the low/high valence config.",
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
        default=None,
        help=(
            "Comma-separated models: Random,Majority,BasicGCN,HeteroGCNMean,HeteroGCNMLP,"
            "HeteroGCNMLPWeights,Mean,SVM,LightGBM,MLP,GazeMAE_MLP,"
            "MOMENT_gaze,MOMENT_pupil,MOMENT_gaze_pupil,MOMENT_all_signals,"
            "MOMENT_GazeMAE_gaze_pupil,MOMENT_GazeMAE_all_signals. "
            "Common lowercase aliases like random, majority, basicgcn, heterogcnmlp, and lgbm are accepted. "
            "By default, use quick_comparison.models from the YAML config."
        ),
    )
    parser.add_argument(
        "--signal-sets",
        type=str,
        default=None,
        help=(
            "Comma-separated signal sets: gaze_only,pupil_only,gaze_pupil,all_signals. "
            "Aliases accepted: gaze, pupil, gaze+pupil, all, full."
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
    parser.add_argument(
        "--report-profile",
        choices=sorted(REPORT_PROFILES),
        default="thesis",
        help=(
            "Reporting profile. thesis saves thesis-ready tables/plots without per-fold loss PNGs; "
            "full keeps debug-heavy plots; minimal saves CSV summaries only."
        ),
    )
    parser.add_argument(
        "--save-fold-loss-plots",
        action="store_true",
        help="Save one loss plot per fold/model even outside the full report profile.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only.")
    return parser.parse_args()


class _TeeStream:
    """Write console output to both the original stream and a timestamped log file."""

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


def _parse_models(raw_models: str | Sequence[Any]) -> List[str]:
    """Parse and validate requested model names."""
    if isinstance(raw_models, str):
        raw_tokens = [token.strip() for token in raw_models.split(",") if token.strip()]
    elif isinstance(raw_models, Sequence):
        raw_tokens = [str(token).strip() for token in raw_models if str(token).strip()]
    else:
        raise ValueError(
            "Model specification must be a comma-separated string or a list, "
            f"got {type(raw_models).__name__}."
        )
    models = [MODEL_ALIASES.get(token.lower(), token) for token in raw_tokens]
    allowed = GNN_MODELS | BASELINE_MODELS
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Allowed: {sorted(allowed)}")
    if not models:
        raise ValueError("At least one model must be requested.")
    return models


def _resolve_requested_models(wrapper_cfg: Dict[str, Any], cli_models: str | None = None) -> List[str]:
    """Resolve requested quick-comparison models from CLI or YAML config."""
    if cli_models is not None:
        return _parse_models(cli_models)

    quick_cfg = wrapper_cfg.get("quick_comparison", {})
    if quick_cfg is None:
        quick_cfg = {}
    if not isinstance(quick_cfg, dict):
        raise ValueError("quick_comparison must be a dictionary when configured.")

    return _parse_models(quick_cfg.get("models", DEFAULT_MODELS))


def _parse_signal_sets(raw_signal_sets: str | Sequence[Any]) -> List[str]:
    """Parse and validate requested signal-set names."""
    if isinstance(raw_signal_sets, str):
        raw_tokens = [token.strip() for token in raw_signal_sets.split(",") if token.strip()]
    elif isinstance(raw_signal_sets, Sequence):
        raw_tokens = [str(token).strip() for token in raw_signal_sets if str(token).strip()]
    else:
        raise ValueError(
            "Signal-set specification must be a comma-separated string or a list, "
            f"got {type(raw_signal_sets).__name__}."
        )

    signal_sets: List[str] = []
    unknown: List[str] = []
    for token in raw_tokens:
        normalized = token.lower().replace("_", "-")
        canonical = SIGNAL_SET_ALIASES.get(normalized)
        if canonical is None:
            unknown.append(token)
        else:
            signal_sets.append(canonical)
    if unknown:
        raise ValueError(f"Unknown signal set(s): {unknown}. Allowed: {SIGNAL_SET_ORDER}")
    unique_signal_sets = list(dict.fromkeys(signal_sets))
    if not unique_signal_sets:
        raise ValueError("At least one signal set must be requested.")
    return unique_signal_sets


def _resolve_requested_signal_sets(wrapper_cfg: Dict[str, Any], cli_signal_sets: str | None = None) -> List[str]:
    """Resolve requested signal sets from CLI, YAML config, or fallback."""
    if cli_signal_sets is not None:
        return _parse_signal_sets(cli_signal_sets)

    quick_cfg = wrapper_cfg.get("quick_comparison", {})
    if quick_cfg is None:
        quick_cfg = {}
    if not isinstance(quick_cfg, dict):
        raise ValueError("quick_comparison must be a dictionary when configured.")

    return _parse_signal_sets(quick_cfg.get("signal_sets", SIGNAL_SET_ORDER))


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


def _model_color_map(model_names: Sequence[str]) -> Dict[str, str]:
    """Return a stable color map for model names, with fallback colors for unknown models."""
    color_map = {
        model_name: MODEL_COLOR_PALETTE[model_name]
        for model_name in model_names
        if model_name in MODEL_COLOR_PALETTE
    }
    missing_models = [model_name for model_name in model_names if model_name not in color_map]
    if missing_models:
        fallback_colors = sns.color_palette("deep", n_colors=len(missing_models)).as_hex()
        color_map.update(dict(zip(missing_models, fallback_colors)))
    return color_map


def _thesis_model_display_name(model_name: str) -> str:
    """Return a thesis-facing model name for plots."""
    return THESIS_MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name).replace("-", "_"))


def _resolve_report_profile(args: argparse.Namespace) -> str:
    """Resolve and validate the command reporting profile."""
    profile = str(getattr(args, "report_profile", "thesis") or "thesis").strip().lower()
    if profile not in REPORT_PROFILES:
        raise ValueError(f"Unknown report profile '{profile}'. Allowed: {sorted(REPORT_PROFILES)}")
    return profile


def _should_save_fold_loss_plots(args: argparse.Namespace, report_profile: str) -> bool:
    """Return whether per-fold loss PNGs should be generated."""
    return bool(getattr(args, "save_fold_loss_plots", False)) or report_profile == "full"


def _resolve_quick_table6_experiment_ids(wrapper_cfg: Dict[str, Any]) -> List[str]:
    """Resolve requested quick Table-6 experiment IDs from wrapper config."""
    quick_cfg = wrapper_cfg.get("quick_comparison", {})
    if quick_cfg is None:
        quick_cfg = {}
    if not isinstance(quick_cfg, dict):
        raise ValueError("quick_comparison must be a dictionary when configured.")

    task_spec = quick_cfg.get("table6_tasks", quick_cfg.get("tasks", DEFAULT_QUICK_EXPERIMENT_IDS))
    if isinstance(task_spec, str):
        normalized = task_spec.strip().lower().replace("_", "-")
        if normalized in {"both", "all", "arousal-valence", "valence-arousal"}:
            requested_tasks = ["arousal", "valence"]
        else:
            requested_tasks = [task_spec]
    elif isinstance(task_spec, list):
        requested_tasks = task_spec
    else:
        raise ValueError(
            "quick_comparison.table6_tasks must be a string or list, e.g. "
            "[arousal], [valence], or [arousal, valence]."
        )

    experiment_ids: List[str] = []
    for raw_task in requested_tasks:
        normalized = str(raw_task).strip().lower().replace("_", "-")
        experiment_id = TABLE6_TASK_TO_EXPERIMENT_ID.get(normalized)
        if experiment_id is None:
            raise ValueError(
                f"Unknown quick Table-6 task '{raw_task}'. "
                "Allowed values: arousal, valence, both."
            )
        experiment_ids.append(experiment_id)

    unique_experiment_ids = list(dict.fromkeys(experiment_ids))
    if not unique_experiment_ids:
        raise ValueError("quick_comparison.table6_tasks must select at least one task.")
    return unique_experiment_ids


def _enable_quick_table6_tasks(wrapper_cfg: Dict[str, Any]) -> None:
    """Enable only the selected quick-comparison Table-6 tasks in a suite wrapper config."""
    selected_experiment_ids = set(_resolve_quick_table6_experiment_ids(wrapper_cfg))
    experiments = wrapper_cfg.get("experiments")
    if isinstance(experiments, dict):
        for experiment_id, experiment_cfg in experiments.items():
            if isinstance(experiment_cfg, dict):
                experiment_cfg["enabled"] = experiment_id in selected_experiment_ids
        return
    if isinstance(experiments, list):
        for experiment_cfg in experiments:
            if isinstance(experiment_cfg, dict):
                experiment_cfg["enabled"] = str(experiment_cfg.get("id", "")) in selected_experiment_ids
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
                    # "num_workers": 0,
                    # "pin_memory": False,
                    # "persistent_workers": False,
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


def _signal_set_dropna_columns(*, use_gaze: bool, use_pupil: bool) -> List[str]:
    """Return raw columns required by one signal-set run."""
    columns = list(METADATA_DROPNA_COLUMNS)
    if use_gaze:
        columns.extend(GAZE_RAW_COLUMNS)
    if use_pupil:
        columns.extend(PUPIL_RAW_COLUMNS)
    return list(dict.fromkeys(columns))


def build_signal_set_variant(signal_set: str) -> SignalSetVariant:
    """Build dataset/model overrides for one signal-set axis value."""
    if signal_set not in SIGNAL_SET_ORDER:
        raise ValueError(f"Unknown signal set: {signal_set}")

    use_gaze = signal_set in {"gaze_only", "gaze_pupil", "all_signals"}
    use_pupil = signal_set in {"pupil_only", "gaze_pupil", "all_signals"}
    use_spatial = signal_set in {"gaze_only", "gaze_pupil", "all_signals"}
    use_distance = signal_set == "all_signals"
    use_fixation = signal_set == "all_signals"
    outlier_columns = []
    if use_gaze:
        outlier_columns.extend(GAZE_RAW_COLUMNS)
    if use_pupil:
        outlier_columns.extend(PUPIL_RAW_COLUMNS)

    dataset_overrides = {
        "use_relative_time": True,
        "use_temporal_node_feature": True,
        "use_temporal_edge_features": True,
        "use_temporal_edges": True,
        "use_spatial_edges": use_spatial,
        "use_fixation_edges": use_fixation,
        "use_gaze_node_features": use_gaze,
        "use_gaze_edge_features": use_gaze,
        "use_pupil_node_features": use_pupil,
        "use_pupil_edge_features": use_pupil,
        "use_distance_avg": use_distance,
        "use_screen_distance_node_feature": use_distance,
        "use_delta_distance_edge_feature": use_distance,
        "use_screen_distance_edge_feature": use_distance,
        "use_fixation_duration": use_fixation,
        "use_fixation_node_feature": use_fixation,
        "dropna_columns": _signal_set_dropna_columns(use_gaze=use_gaze, use_pupil=use_pupil),
        "signal_outlier_filter": {
            **SIGNAL_OUTLIER_FILTER_DEFAULTS,
            "columns": outlier_columns,
        },
    }

    return SignalSetVariant(
        name=signal_set,
        description=SIGNAL_SET_DESCRIPTIONS[signal_set],
        overrides={
            "global_overrides": {
                "dataset": dataset_overrides,
                "gnn": {
                    "model": {
                        "use_spatial_edges": use_spatial,
                        "use_fixation_edges": use_fixation,
                        "use_delta_distance_edge_feature": use_distance,
                    }
                },
            },
        },
    )


def _model_names_for_signal_set(model_names: Sequence[str], signal_set: str) -> List[str]:
    """Map any requested frozen embedding baseline to the one signal-aware baseline."""
    mapped_embedding = SIGNAL_SET_EMBEDDING_MODEL[signal_set]
    resolved: List[str] = []
    emitted_embedding = False
    for model_name in model_names:
        if model_name in FROZEN_EMBEDDING_MODELS:
            if not emitted_embedding:
                resolved.append(mapped_embedding)
                emitted_embedding = True
            continue
        resolved.append(model_name)
    return list(dict.fromkeys(resolved))


def build_variant(model_name: str) -> QuickVariant:
    """Build one variant override block."""
    if model_name in GNN_MODELS:
        descriptions = {
            "BasicGCN": (
                "Homogeneous GCN baseline over the v2 graph schema with collapsed, "
                "deduplicated relations, no edge attributes, and attention readout."
            ),
            "HeteroGCNMean": (
                "Heterogeneous GCN with separate temporal, spatial, and fixation relation paths, "
                "mean relation fusion, no edge weights, and attention readout."
            ),
            "HeteroGCNMLP": (
                "Heterogeneous GCN with separate relation paths, MLP relation fusion, "
                "no edge weights, and attention readout."
            ),
            "HeteroGCNMLPWeights": (
                "Heterogeneous GCN with MLP relation fusion, learned signed edge weights, "
                "and attention readout."
            ),
        }
        return QuickVariant(
            model_name=model_name,
            description=descriptions[model_name],
            summary_model_name=model_name,
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
                            "model_version": model_name,
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
    gnn_names = [name for name in model_names if name in GNN_MODELS]
    ordered_run_models = list(dict.fromkeys([*baseline_names, *gnn_names]))
    if not ordered_run_models:
        return []

    overrides: Dict[str, Any] = {
        "global_overrides": {
            "run_experiments": {"baselines": bool(baseline_names), "gnn": bool(gnn_names)},
        }
    }
    if baseline_names:
        overrides["global_overrides"]["baselines"] = {"models": baseline_names}
    if gnn_names:
        overrides["global_overrides"]["dataset"] = {
            "graph_version": "v2",
            "edge_weight_mode": "learned_signed",
            "use_edge_weights": True,
        }
        overrides["global_overrides"]["gnn"] = {
            "models": gnn_names,
            "model": {
                "model_version": gnn_names[0],
            },
        }

    if baseline_names and gnn_names:
        run_name = "Models"
        description = (
            "Baselines and GNN variants on one shared suite run and aligned CV folds: "
            f"{', '.join(ordered_run_models)}."
        )
    elif baseline_names:
        run_name = "Baselines"
        description = f"Baselines on the same quick subset and folds: {', '.join(baseline_names)}."
    else:
        run_name = "GNNs"
        description = f"GNN variants on one shared graph dataset and folds: {', '.join(gnn_names)}."

    return [
        QuickRun(
            run_name=run_name,
            description=description,
            model_names=ordered_run_models,
            summary_model_names={name: name for name in ordered_run_models},
            overrides=overrides,
        )
    ]


def _build_payload(base_cfg: Dict[str, Any], fixed_overrides: Dict[str, Any], variant: QuickVariant) -> Dict[str, Any]:
    """Merge base, fixed, and variant-specific overrides."""
    payload = merge_many(base_cfg, fixed_overrides, variant.overrides)
    _enable_quick_table6_tasks(payload)
    return payload


def _build_run_payload(
    base_cfg: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    signal_set: SignalSetVariant,
    quick_run: QuickRun,
) -> Dict[str, Any]:
    """Merge base, fixed, signal-set, and run-specific overrides."""
    payload = merge_many(base_cfg, fixed_overrides, signal_set.overrides, quick_run.overrides)
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
    signal_sets = _resolve_requested_signal_sets(base_cfg, args.signal_sets)
    requested_models = _resolve_requested_models(base_cfg, args.models)
    report_profile = _resolve_report_profile(args)

    trainer_base_cfg = _load_multiclass_base_config(base_cfg, base_config_path)
    trainer_cfg = merge_many(trainer_base_cfg, payload.get("global_overrides", {}))
    cv_cfg = trainer_cfg.get("cross_validation", {})
    gnn_training_cfg = trainer_cfg.get("gnn", {}).get("training", {})
    gnn_model_cfg = trainer_cfg.get("gnn", {}).get("model", {})
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
        "signal_sets": ",".join(signal_sets),
        "models": ",".join(requested_models),
        "signal_set_models": "; ".join(
            f"{signal_set}:{','.join(_model_names_for_signal_set(requested_models, signal_set))}"
            for signal_set in signal_sets
        ),
        "report_profile": report_profile,
        "save_fold_loss_plots": str(_should_save_fold_loss_plots(args, report_profile)),
        "GNN conv_type": str(gnn_model_cfg.get("conv_type", "not configured")),
        "relation_pooling": str(gnn_model_cfg.get("relation_pooling", "not configured")),
        "graph_pooling": str(gnn_model_cfg.get("graph_pooling", "not configured")),
        "head_pooling": str(gnn_model_cfg.get("head_pooling", "not configured")),
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


def _artifact_dir_for_summary_model(
    fold_dir: Path,
    summary_model_name: str,
    model_name: str | None = None,
) -> Path:
    """Return the fold-level artifact directory for a summary model row."""
    if summary_model_name in GNN_MODELS:
        return fold_dir / "gnn" / summary_model_name
    if summary_model_name == "GNN":
        return fold_dir
    return fold_dir / "baselines" / (model_name or summary_model_name)


def _rows_to_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Build a stable summary dataframe from result rows."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    sort_cols = [
        col
        for col in ["status", "signal_set", "experiment_id", "cv_strategy", "balanced_accuracy", "macro_f1", "model"]
        if col in df.columns
    ]
    descending_metric_cols = {"balanced_accuracy", "macro_f1"}
    ascending = [col not in descending_metric_cols for col in sort_cols]
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _collect_fold_metrics(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Collect per-fold metrics from successful quick-run trainer outputs."""
    records: List[pd.DataFrame] = []
    successful_rows = [row for row in rows if row.get("status") == "success"]

    for row in successful_rows:
        experiment_id = str(row.get("experiment_id", ""))
        cv_strategy = str(row.get("cv_strategy", ""))
        model_name = str(row.get("model", ""))
        summary_model_name = str(row.get("summary_model_name", ""))
        suite_run_dir = Path(str(row.get("suite_run_dir", "")))
        try:
            trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir, experiment_id=experiment_id)
        except Exception:
            continue

        fold_metrics_path = trainer_run_dir / cv_strategy / "fold_metrics.csv"
        if not fold_metrics_path.exists():
            continue

        fold_metrics = pd.read_csv(fold_metrics_path)
        if fold_metrics.empty or "model" not in fold_metrics.columns:
            continue

        model_metrics = fold_metrics[fold_metrics["model"].astype(str) == summary_model_name].copy()
        if model_metrics.empty:
            continue

        model_metrics = model_metrics.rename(columns={"model": "metric_source_model"})
        model_metrics.insert(0, "fold_metrics_path", str(fold_metrics_path))
        model_metrics.insert(0, "trainer_run_dir", str(trainer_run_dir))
        model_metrics.insert(0, "suite_run_dir", str(suite_run_dir))
        model_metrics.insert(0, "summary_model_name", summary_model_name)
        model_metrics.insert(0, "model", model_name)
        model_metrics.insert(0, "cv_strategy", cv_strategy)
        model_metrics.insert(0, "experiment_display_name", EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id))
        model_metrics.insert(0, "experiment_id", experiment_id)
        if "signal_set_description" in row:
            model_metrics.insert(0, "signal_set_description", str(row.get("signal_set_description", "")))
        if "signal_set" in row:
            model_metrics.insert(0, "signal_set", str(row.get("signal_set", "")))
        records.append(model_metrics)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _build_metric_summary_with_std(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level metrics into long-form mean/std rows."""
    group_columns = [
        *([column for column in ["signal_set", "signal_set_description"] if column in fold_metrics.columns]),
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
        "summary_model_name",
        "metric_type",
    ]
    if fold_metrics.empty or not set(group_columns).issubset(fold_metrics.columns):
        return pd.DataFrame()

    metadata_columns = set(group_columns) | {
        "fold_id",
        "metric_source_model",
        "suite_run_dir",
        "trainer_run_dir",
        "fold_metrics_path",
    }
    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in metadata_columns and pd.api.types.is_numeric_dtype(fold_metrics[column])
    ]
    if not metric_columns:
        return pd.DataFrame()

    long_metrics = fold_metrics.melt(
        id_vars=group_columns,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long_metrics.empty:
        return pd.DataFrame()

    summary = (
        long_metrics.groupby([*group_columns, "metric"], dropna=False)["value"]
        .agg(n_folds="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary.sort_values([*group_columns, "metric"]).reset_index(drop=True)


def _save_fold_metric_outputs(rows: Sequence[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Save top-level fold metrics and fold-derived metric summaries."""
    fold_metrics = _collect_fold_metrics(rows)
    if fold_metrics.empty:
        return []

    saved_paths: List[Path] = []
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_path = tables_dir / "fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False)
    saved_paths.append(fold_metrics_path)

    metric_summary = _build_metric_summary_with_std(fold_metrics)
    if not metric_summary.empty:
        metric_summary_path = tables_dir / "metric_summary_with_std.csv"
        metric_summary.to_csv(metric_summary_path, index=False)
        saved_paths.append(metric_summary_path)

    return saved_paths


def _collect_model_benchmark_records(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Collect per-fold benchmark artifacts from successful quick-comparison runs."""
    records: List[pd.DataFrame] = []
    for row in rows:
        if str(row.get("status")) != "success":
            continue

        model_name = str(row.get("model", ""))
        summary_model_name = str(row.get("summary_model_name", ""))
        experiment_id = str(row.get("experiment_id", ""))
        cv_strategy = str(row.get("cv_strategy", ""))
        suite_run_dir = Path(str(row.get("suite_run_dir", "")))
        try:
            trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir, experiment_id=experiment_id)
        except Exception:
            continue

        strategy_dir = trainer_run_dir / cv_strategy
        if not strategy_dir.exists():
            continue

        benchmark_paths: List[Path] = []
        for fold_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
            benchmark_paths.append(
                _artifact_dir_for_summary_model(
                    fold_dir=fold_dir,
                    summary_model_name=summary_model_name,
                    model_name=model_name,
                )
                / "model_benchmark.json"
            )

        benchmark_df = load_benchmark_records(benchmark_paths)
        if benchmark_df.empty:
            continue

        benchmark_df.insert(0, "trainer_model", benchmark_df.get("model", ""))
        benchmark_df["model"] = model_name
        benchmark_df["summary_model_name"] = summary_model_name
        benchmark_df["experiment_id"] = experiment_id
        benchmark_df["experiment_display_name"] = EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id)
        benchmark_df["cv_strategy"] = cv_strategy
        if "signal_set" in row:
            benchmark_df["signal_set"] = str(row.get("signal_set", ""))
        if "signal_set_description" in row:
            benchmark_df["signal_set_description"] = str(row.get("signal_set_description", ""))
        benchmark_df["suite_run_dir"] = str(suite_run_dir)
        benchmark_df["trainer_run_dir"] = str(trainer_run_dir)
        records.append(benchmark_df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _format_accuracy_macro_f1(accuracy: Any, macro_f1: Any) -> str:
    """Format the compact main-text metric column."""
    accuracy_value = pd.to_numeric(pd.Series([accuracy]), errors="coerce").iloc[0]
    macro_f1_value = pd.to_numeric(pd.Series([macro_f1]), errors="coerce").iloc[0]
    if pd.isna(accuracy_value) and pd.isna(macro_f1_value):
        return ""
    return f"{accuracy_value:.4f} / {macro_f1_value:.4f}"


def _write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    """Write a small GitHub-flavored markdown table without extra dependencies."""
    if df.empty:
        output_path.write_text("", encoding="utf-8")
        return

    text_df = df.fillna("").astype(str)
    headers = text_df.columns.tolist()
    rows = text_df.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def fmt_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_model_benchmark_outputs(
    *,
    rows: Sequence[Dict[str, Any]],
    quick_summary: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Save raw, summary, and main-text benchmark tables."""
    records = _collect_model_benchmark_records(rows)
    if records.empty:
        return []

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    raw_path = tables_dir / "model_benchmark_raw.csv"
    records.to_csv(raw_path, index=False)
    saved_paths.append(raw_path)

    group_columns = [
        *([column for column in ["signal_set", "signal_set_description"] if column in records.columns]),
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
    ]
    benchmark_summary = summarize_benchmark_records(records, group_columns=group_columns)
    if benchmark_summary.empty:
        return saved_paths

    metric_columns = [
        *([column for column in ["signal_set"] if column in quick_summary.columns]),
        "experiment_id",
        "cv_strategy",
        "model",
        "accuracy",
        "macro_f1",
    ]
    if not quick_summary.empty and set(metric_columns).issubset(quick_summary.columns):
        metrics = quick_summary[quick_summary["status"] == "success"].loc[:, metric_columns].copy()
        for metric in ["accuracy", "macro_f1"]:
            metrics[metric] = pd.to_numeric(metrics[metric], errors="coerce")
        benchmark_summary = benchmark_summary.drop(
            columns=[column for column in ["accuracy", "macro_f1"] if column in benchmark_summary.columns],
            errors="ignore",
        ).merge(
            metrics,
            on=[column for column in ["signal_set", "experiment_id", "cv_strategy", "model"] if column in metric_columns],
            how="left",
        )

    summary_path = tables_dir / "model_benchmark_summary.csv"
    benchmark_summary.to_csv(summary_path, index=False)
    saved_paths.append(summary_path)

    main_models = [
        "LightGBM",
        "SVM",
        "MLP",
        "GazeMAE_MLP",
        "MOMENT_gaze",
        "MOMENT_pupil",
        "MOMENT_gaze_pupil",
        "MOMENT_all_signals",
        "MOMENT_GazeMAE_gaze_pupil",
        "MOMENT_GazeMAE_all_signals",
        "BasicGCN",
        "HeteroGCNMean",
        "HeteroGCNMLP",
        "HeteroGCNMLPWeights",
    ]
    main_report = benchmark_summary[benchmark_summary["model"].isin(main_models)].copy()
    if not main_report.empty:
        if "encoder_head_inference_ms_per_window" in main_report.columns:
            encoder_head_time = pd.to_numeric(
                main_report["encoder_head_inference_ms_per_window"],
                errors="coerce",
            )
            main_report["inference_ms_per_window"] = np.where(
                (main_report["model"] == "GazeMAE_MLP") & encoder_head_time.notna(),
                encoder_head_time,
                main_report["inference_ms_per_window"],
            )
        model_order = {model: idx for idx, model in enumerate(main_models)}
        main_report["model_order"] = main_report["model"].map(model_order)
        main_report["accuracy / macro-F1"] = [
            _format_accuracy_macro_f1(row.accuracy, row.macro_f1)
            for row in main_report[["accuracy", "macro_f1"]].itertuples(index=False)
        ]
        main_report = main_report.sort_values(
            [column for column in ["signal_set", "experiment_id", "cv_strategy", "model_order"] if column in main_report.columns]
        ).rename(
            columns={
                "experiment_display_name": "task",
                "signal_set": "signal_set",
                "model": "model",
                "trainable_parameters": "trainable_parameters",
                "total_parameters": "total_parameters",
                "train_ms_per_window": "train_ms_per_window",
                "inference_ms_per_window": "inference_ms_per_window",
            }
        )
        main_report = main_report[
            [
                *([column for column in ["signal_set"] if column in main_report.columns]),
                "task",
                "cv_strategy",
                "model",
                "trainable_parameters",
                "total_parameters",
                "train_ms_per_window",
                "inference_ms_per_window",
                "accuracy / macro-F1",
            ]
        ]
        main_csv_path = tables_dir / "main_model_complexity_report.csv"
        main_md_path = tables_dir / "main_model_complexity_report.md"
        main_report.to_csv(main_csv_path, index=False)
        _write_markdown_table(main_report, main_md_path)
        saved_paths.extend([main_csv_path, main_md_path])

    return saved_paths


def _save_group_model_ranking(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save a command-level metric comparison plot from quick summary metrics."""
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
    signal_order = (
        [signal_set for signal_set in SIGNAL_SET_ORDER if signal_set in set(plot_df["signal_set"].astype(str))]
        if "signal_set" in plot_df.columns
        else ["all"]
    )
    long_df = plot_df.melt(
        id_vars=[
            col
            for col in ["signal_set", "signal_set_description", "experiment_id", "experiment_display_name", "cv_strategy", "model"]
            if col in plot_df.columns
        ],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long_df.empty:
        return None

    metric_order = [metric for metric in metric_columns if metric in available_metrics]
    palette = _model_color_map(model_order)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "classification_group_model_ranking.png"

    n_panels = len(signal_order) * len(experiment_order) * len(strategy_order)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(14.5, 4.35 * n_panels),
        squeeze=False,
    )
    row_idx = 0
    for signal_set in signal_order:
        signal_df = (
            long_df[long_df["signal_set"].astype(str) == signal_set]
            if "signal_set" in long_df.columns
            else long_df
        )
        signal_title = SIGNAL_SET_DESCRIPTIONS.get(str(signal_set), str(signal_set))
        for experiment_id in experiment_order:
            experiment_df = (
                signal_df[signal_df["experiment_id"].astype(str) == experiment_id]
                if "experiment_id" in signal_df.columns
                else signal_df
            )
            experiment_name = EXPERIMENT_DISPLAY_NAMES.get(str(experiment_id), str(experiment_id))
            for strategy in strategy_order:
                ax = axes[row_idx, 0]
                strategy_df = (
                    experiment_df[experiment_df["cv_strategy"].astype(str) == strategy]
                    if "cv_strategy" in experiment_df.columns
                    else experiment_df
                )
                if strategy_df.empty:
                    row_idx += 1
                    continue
                sns.barplot(
                    data=strategy_df,
                    x="metric",
                    y="value",
                    hue="model",
                    order=metric_order,
                    hue_order=model_order,
                    palette=palette,
                    ax=ax,
                    width=0.80,
                )
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

                for container, model_name in zip(ax.containers, model_order):
                    for patch in container:
                        height = patch.get_height()
                        if not pd.notna(height):
                            continue
                        x_pos = patch.get_x() + patch.get_width() / 2.0
                        if height >= 0.19:
                            ax.text(
                                x_pos,
                                height / 2.0,
                                model_name,
                                ha="center",
                                va="center",
                                rotation=90,
                                color="white",
                                fontsize=7.2,
                                fontweight="bold",
                                clip_on=True,
                            )
                        else:
                            ax.text(
                                x_pos,
                                height + 0.012,
                                model_name,
                                ha="center",
                                va="bottom",
                                rotation=90,
                                color="#222222",
                                fontsize=6.5,
                                fontweight="bold",
                                clip_on=True,
                            )
                title_suffix = f" - {strategy}" if "cv_strategy" in experiment_df.columns else ""
                signal_prefix = f"{signal_set}: " if "signal_set" in long_df.columns else ""
                ax.set_title(
                    f"{signal_prefix}Quick {experiment_name} Metrics by Model{title_suffix}",
                    fontsize=14,
                    pad=10,
                )
                if "signal_set" in long_df.columns:
                    ax.text(
                        0.0,
                        1.01,
                        signal_title,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        color="#555555",
                    )
                ax.set_xlabel("")
                ax.set_ylabel("metric value", fontsize=11)
                ax.set_ylim(0.0, 1.0)
                ax.tick_params(axis="x", rotation=20, labelsize=11)
                ax.tick_params(axis="y", labelsize=10)
                ax.grid(axis="y", alpha=0.30)
                ax.grid(axis="x", visible=False)
                row_idx += 1
    fig.subplots_adjust(top=0.965, bottom=0.035, left=0.06, right=0.995, hspace=0.50)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _save_group_model_ranking_interactive(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save an interactive HTML metric comparison plot with legend-toggleable models."""
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "auc"]
    available_metrics = [metric for metric in metric_columns if metric in summary.columns]
    if summary.empty or not available_metrics:
        return None

    try:
        import plotly.express as px
    except ImportError:
        return None

    plot_df = summary[summary["status"] == "success"].copy()
    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=available_metrics, how="all")
    if plot_df.empty:
        return None

    id_vars = [
        col
        for col in ["signal_set", "signal_set_description", "experiment_id", "experiment_display_name", "cv_strategy", "model"]
        if col in plot_df.columns
    ]
    long_df = plot_df.melt(
        id_vars=id_vars,
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long_df.empty:
        return None

    long_df["group"] = long_df.apply(
        lambda row: (
            (f"{row.get('signal_set')} - " if pd.notna(row.get("signal_set", np.nan)) else "")
            + f"{row.get('experiment_display_name', row.get('experiment_id', 'Experiment'))}"
            + f" - {row.get('cv_strategy', 'all')}"
        ),
        axis=1,
    )
    model_order = _ordered_models(long_df["model"].astype(str).unique().tolist())
    metric_order = [metric for metric in metric_columns if metric in available_metrics]
    group_order = list(dict.fromkeys(long_df["group"].astype(str).tolist()))
    palette = _model_color_map(model_order)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "classification_group_model_ranking.html"
    hover_data = {"metric": True, "value": ":.4f"}
    if "experiment_id" in long_df.columns:
        hover_data["experiment_id"] = True
    if "cv_strategy" in long_df.columns:
        hover_data["cv_strategy"] = True

    fig = px.bar(
        long_df,
        x="metric",
        y="value",
        color="model",
        color_discrete_map=palette,
        facet_row="group",
        barmode="group",
        text="model",
        category_orders={
            "model": model_order,
            "metric": metric_order,
            "group": group_order,
        },
        hover_data=hover_data,
        title="Quick Metrics by Model",
        height=max(420, 360 * len(group_order)),
    )
    fig.update_xaxes(title_text="metric")
    fig.update_yaxes(range=[0.0, 1.0], title_text="metric value")
    fig.update_traces(textangle=-90, textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        legend_title_text="model",
        showlegend=False,
        bargap=0.18,
        margin={"l": 70, "r": 40, "t": 70, "b": 80},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def _save_thesis_metric_table(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save compact thesis-oriented signal-set/model metric table with ranks."""
    metric_columns = ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1", "loss"]
    if summary.empty or "status" not in summary.columns:
        return None
    available_metrics = [column for column in metric_columns if column in summary.columns]
    group_columns = [column for column in ["signal_set", "experiment_id", "experiment_display_name", "cv_strategy"] if column in summary.columns]
    required_columns = {"model", *group_columns}
    if not available_metrics or not required_columns.issubset(summary.columns):
        return None

    table = summary[summary["status"] == "success"].copy()
    if table.empty:
        return None
    for metric in available_metrics:
        table[metric] = pd.to_numeric(table[metric], errors="coerce")
    table = table.dropna(subset=["accuracy"], how="any") if "accuracy" in table.columns else table
    if table.empty:
        return None

    output_columns = [
        *group_columns,
        *([column for column in ["signal_set_description"] if column in table.columns]),
        "model",
        *available_metrics,
    ]
    table = table.loc[:, output_columns].copy()
    sort_columns = [*group_columns, "accuracy"]
    ascending = [True] * len(group_columns) + [False]
    if "macro_f1" in table.columns:
        sort_columns.append("macro_f1")
        ascending.append(False)
    sort_columns.append("model")
    ascending.append(True)
    table = table.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    if group_columns:
        table["rank"] = table.groupby(group_columns, dropna=False).cumcount() + 1
    else:
        table["rank"] = np.arange(1, len(table) + 1)

    table = table.loc[
        :,
        [
            *group_columns,
            *([column for column in ["signal_set_description"] if column in table.columns]),
            "rank",
            "model",
            *available_metrics,
        ],
    ]

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "thesis_signal_set_model_metrics.csv"
    table.to_csv(output_path, index=False)
    return output_path


def _slug_for_plot(value: Any) -> str:
    """Return a stable filename token for a plot grouping value."""
    return _slugify_filename(str(value)).strip("_") or "all"


def _save_one_thesis_metric_heatmap(
    plot_df: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str,
    experiment_id: str,
    experiment_name: str,
    cv_strategy: str,
) -> Path | None:
    """Save one signal-set by model thesis heatmap."""
    if plot_df.empty:
        return None

    model_order = _ordered_models(plot_df["model"].astype(str).tolist())
    signal_order = (
        [signal_set for signal_set in SIGNAL_SET_ORDER if signal_set in set(plot_df["signal_set"].astype(str))]
        if "signal_set" in plot_df.columns
        else ["all"]
    )
    if not model_order or not signal_order:
        return None

    if "signal_set" not in plot_df.columns:
        heatmap_df = pd.DataFrame([plot_df.set_index("model")[metric]], index=["all"])
    else:
        heatmap_df = plot_df.pivot_table(
            index="signal_set",
            columns="model",
            values=metric,
            aggfunc="first",
        )
    heatmap_df = heatmap_df.reindex(index=signal_order, columns=model_order)
    if heatmap_df.dropna(how="all").empty:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        plots_dir
        / f"thesis_{metric}_heatmap_{_slug_for_plot(experiment_id)}_{_slug_for_plot(cv_strategy)}.png"
    )

    width = max(7.5, 1.0 * len(model_order) + 2.0)
    height = max(3.2, 0.55 * len(signal_order) + 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric},
        ax=ax,
    )
    ax.set_title(f"{experiment_name} - {cv_strategy} - {metric}")
    ax.set_xlabel("model")
    ax.set_ylabel("signal set")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_thesis_metric_heatmaps(thesis_metrics: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Save thesis heatmaps for accuracy and macro-F1 by signal set and model."""
    if thesis_metrics.empty:
        return []
    required = {"model", "experiment_id", "cv_strategy"}
    if not required.issubset(thesis_metrics.columns):
        return []

    saved_paths: List[Path] = []
    group_columns = ["experiment_id", "cv_strategy"]
    if "experiment_display_name" in thesis_metrics.columns:
        group_columns.insert(1, "experiment_display_name")

    for group_values, group_df in thesis_metrics.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(group_columns, group_values))
        experiment_id = str(group.get("experiment_id", "experiment"))
        experiment_name = str(group.get("experiment_display_name", experiment_id))
        cv_strategy = str(group.get("cv_strategy", "all"))
        for metric in ["accuracy", "macro_f1"]:
            if metric not in group_df.columns:
                continue
            metric_df = group_df.copy()
            metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
            metric_df = metric_df.dropna(subset=[metric])
            path = _save_one_thesis_metric_heatmap(
                metric_df,
                output_dir,
                metric=metric,
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                cv_strategy=cv_strategy,
            )
            if path is not None:
                saved_paths.append(path)
    return saved_paths


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
        artifact_dir = _artifact_dir_for_summary_model(
            fold_dir=fold_dir,
            summary_model_name=summary_model_name,
        )
        pred_path = artifact_dir / "test_predictions.npy"
        target_path = artifact_dir / "test_targets.npy"
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
        target_path = (
            _artifact_dir_for_summary_model(
                fold_dir=fold_dir,
                summary_model_name=summary_model_name,
            )
            / "test_targets.npy"
        )
        if not target_path.exists():
            continue
        fold_targets[fold_dir.name] = np.asarray(np.load(target_path)).reshape(-1).astype(int)
    return fold_targets


def _build_label_distribution_tables(rows: Sequence[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-fold and aggregate label-distribution tables from saved test targets."""
    successful_rows = [row for row in rows if row.get("status") == "success"]
    pairs = list(
        dict.fromkeys(
            (str(row.get("signal_set", "")), str(row.get("experiment_id")), str(row.get("cv_strategy")))
            for row in successful_rows
        )
    )

    fold_records: List[Dict[str, Any]] = []
    aggregate_records: List[Dict[str, Any]] = []
    for signal_set, experiment_id, cv_strategy in pairs:
        pair_rows = [
            row
            for row in successful_rows
            if str(row.get("signal_set", "")) == signal_set
            and str(row.get("experiment_id")) == experiment_id
            and str(row.get("cv_strategy")) == cv_strategy
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
                        "signal_set": signal_set,
                        "signal_set_description": str(source_row.get("signal_set_description", "")),
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
                    "signal_set": signal_set,
                    "signal_set_description": str(source_row.get("signal_set_description", "")),
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

    group_keys = [column for column in ["signal_set", "experiment_id", "cv_strategy"] if column in plot_df.columns]
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
    for idx, group_values in enumerate(groups):
        group = dict(zip(group_keys, group_values))
        ax = axes[idx, 0]
        group_df = plot_df.copy()
        for key, value in group.items():
            group_df = group_df[group_df[key] == value]
        sns.barplot(
            data=group_df,
            x="class_name",
            y=y_column,
            color="#4C78A8",
            ax=ax,
        )
        experiment_id = str(group.get("experiment_id", ""))
        cv_strategy = str(group.get("cv_strategy", ""))
        experiment_name = EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id)
        signal_prefix = f"{group.get('signal_set')}: " if "signal_set" in group else ""
        ax.set_title(f"{signal_prefix}{experiment_name} Label Distribution - {cv_strategy}")
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


def _collect_training_history(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Collect available fold-level GNN and MLP-head training histories."""
    records: List[pd.DataFrame] = []
    successful_rows = [row for row in rows if row.get("status") == "success"]

    for row in successful_rows:
        model_name = str(row.get("model", ""))
        summary_model_name = str(row.get("summary_model_name", ""))
        if (
            summary_model_name not in GNN_MODELS
            and summary_model_name != "GNN"
            and model_name not in {"MLP", "GazeMAE_MLP", *MOMENT_MODEL_NAMES}
        ):
            continue

        experiment_id = str(row.get("experiment_id", ""))
        cv_strategy = str(row.get("cv_strategy", ""))
        suite_run_dir = Path(str(row.get("suite_run_dir", "")))
        try:
            trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir, experiment_id=experiment_id)
        except Exception:
            continue

        strategy_dir = trainer_run_dir / cv_strategy
        if not strategy_dir.exists():
            continue
        test_loss_by_fold = _load_fold_test_losses(
            strategy_dir=strategy_dir,
            model_name=model_name,
            summary_model_name=summary_model_name,
        )

        for fold_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
            if summary_model_name in GNN_MODELS or summary_model_name == "GNN":
                history_path = (
                    _artifact_dir_for_summary_model(
                        fold_dir=fold_dir,
                        summary_model_name=summary_model_name,
                        model_name=model_name,
                    )
                    / "gnn_training_history.csv"
                )
                history_kind = "gnn"
            elif model_name in {"MLP", "GazeMAE_MLP", *MOMENT_MODEL_NAMES}:
                history_path = fold_dir / "baselines" / model_name / "mlp_training_history.csv"
                history_kind = str(model_name).lower()
            else:
                continue

            if not history_path.exists():
                continue

            history = pd.read_csv(history_path)
            if history.empty or "epoch" not in history.columns:
                continue
            test_loss = test_loss_by_fold.get(fold_dir.name)
            if test_loss is not None and np.isfinite(test_loss):
                history["test_loss"] = float(test_loss)

            history.insert(0, "history_kind", history_kind)
            history.insert(0, "history_path", str(history_path))
            history.insert(0, "trainer_run_dir", str(trainer_run_dir))
            history.insert(0, "suite_run_dir", str(suite_run_dir))
            history.insert(0, "fold_id", fold_dir.name)
            history.insert(0, "model", model_name)
            history.insert(0, "summary_model_name", summary_model_name)
            history.insert(0, "cv_strategy", cv_strategy)
            history.insert(0, "experiment_display_name", EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id))
            history.insert(0, "experiment_id", experiment_id)
            if "signal_set_description" in row:
                history.insert(0, "signal_set_description", str(row.get("signal_set_description", "")))
            if "signal_set" in row:
                history.insert(0, "signal_set", str(row.get("signal_set", "")))
            records.append(history)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _collect_training_diagnostics(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Collect available fold-level GNN diagnostics from trainer outputs."""
    records: List[pd.DataFrame] = []
    successful_rows = [row for row in rows if row.get("status") == "success"]

    for row in successful_rows:
        model_name = str(row.get("model", ""))
        summary_model_name = str(row.get("summary_model_name", ""))
        if summary_model_name not in GNN_MODELS and summary_model_name != "GNN":
            continue

        experiment_id = str(row.get("experiment_id", ""))
        cv_strategy = str(row.get("cv_strategy", ""))
        suite_run_dir = Path(str(row.get("suite_run_dir", "")))
        try:
            trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir, experiment_id=experiment_id)
        except Exception:
            continue

        strategy_dir = trainer_run_dir / cv_strategy
        if not strategy_dir.exists():
            continue

        for fold_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
            diagnostics_path = (
                _artifact_dir_for_summary_model(
                    fold_dir=fold_dir,
                    summary_model_name=summary_model_name,
                    model_name=model_name,
                )
                / "gnn_fold_diagnostics.csv"
            )
            if not diagnostics_path.exists():
                continue

            diagnostics = pd.read_csv(diagnostics_path)
            if diagnostics.empty or "split" not in diagnostics.columns:
                continue

            diagnostics.insert(0, "diagnostics_path", str(diagnostics_path))
            diagnostics.insert(0, "trainer_run_dir", str(trainer_run_dir))
            diagnostics.insert(0, "suite_run_dir", str(suite_run_dir))
            diagnostics.insert(0, "fold_id", fold_dir.name)
            diagnostics.insert(0, "model", model_name)
            diagnostics.insert(0, "summary_model_name", summary_model_name)
            diagnostics.insert(0, "cv_strategy", cv_strategy)
            diagnostics.insert(0, "experiment_display_name", EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id))
            diagnostics.insert(0, "experiment_id", experiment_id)
            if "signal_set_description" in row:
                diagnostics.insert(0, "signal_set_description", str(row.get("signal_set_description", "")))
            if "signal_set" in row:
                diagnostics.insert(0, "signal_set", str(row.get("signal_set", "")))
            records.append(diagnostics)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _compute_saved_prediction_log_loss(prediction_path: Path, target_path: Path) -> float | None:
    """Compute multiclass log loss from saved probabilities and encoded targets."""
    if not prediction_path.exists() or not target_path.exists():
        return None

    y_pred = np.load(prediction_path)
    y_true = np.load(target_path)
    if y_pred.ndim != 2 or y_pred.shape[0] == 0:
        return None

    labels = np.arange(y_pred.shape[1], dtype=int)
    try:
        return float(log_loss(y_true, y_pred, labels=labels))
    except ValueError:
        return None


def _load_fold_test_losses(
    strategy_dir: Path,
    model_name: str,
    summary_model_name: str,
) -> Dict[str, float]:
    """Load one held-out test loss per fold for a plotted training-history model."""
    expected_metric_model = "GNN" if summary_model_name == "GNN" else model_name
    fold_metrics_path = strategy_dir / "fold_metrics.csv"
    if fold_metrics_path.exists():
        fold_metrics = pd.read_csv(fold_metrics_path)
        required_columns = {"model", "fold_id", "metric_type", "loss"}
        if required_columns.issubset(fold_metrics.columns):
            metric_df = fold_metrics[
                (fold_metrics["model"].astype(str) == expected_metric_model)
                & (fold_metrics["metric_type"].astype(str) == "aggregated")
            ].copy()
            metric_df["loss"] = pd.to_numeric(metric_df["loss"], errors="coerce")
            metric_df = metric_df.dropna(subset=["loss"])
            if not metric_df.empty:
                return {
                    str(row.fold_id): float(row.loss)
                    for row in metric_df[["fold_id", "loss"]].itertuples(index=False)
                }

    losses: Dict[str, float] = {}
    for fold_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
        if summary_model_name in GNN_MODELS or summary_model_name == "GNN":
            artifact_dir = _artifact_dir_for_summary_model(
                fold_dir=fold_dir,
                summary_model_name=summary_model_name,
                model_name=model_name,
            )
            prediction_path = artifact_dir / "test_predictions.npy"
            target_path = artifact_dir / "test_targets.npy"
        elif model_name in {"MLP", "GazeMAE_MLP", *MOMENT_MODEL_NAMES}:
            prediction_path = fold_dir / "baselines" / model_name / "test_predictions.npy"
            target_path = fold_dir / "baselines" / model_name / "test_targets.npy"
        else:
            continue

        loss_value = _compute_saved_prediction_log_loss(prediction_path, target_path)
        if loss_value is not None and np.isfinite(loss_value):
            losses[fold_dir.name] = loss_value

    return losses


def _slugify_filename(value: str) -> str:
    """Return a compact filename-safe representation."""
    safe_chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)]
    return "_".join("".join(safe_chars).split("_"))


def _metric_display_name(metric: str) -> str:
    """Return a readable metric label for plot titles and legends."""
    if metric in LOSS_METRIC_STYLES:
        return str(LOSS_METRIC_STYLES[metric]["label"])
    return {
        "val_balanced_accuracy": "Validation balanced accuracy",
        "val_macro_f1": "Validation macro F1",
    }.get(metric, metric)


def _compute_aggregated_metric_curve(
    group_df: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute mean/std/min/max curve, keeping test loss constant over epochs."""
    if group_df.empty:
        return None

    if metric == "test_loss":
        metric_values = (
            group_df[group_df["metric"] == metric]
            .drop_duplicates(subset=["fold_id"])["value"]
            .dropna()
            .to_numpy(dtype=float)
        )
        x = np.sort(group_df["epoch"].dropna().unique().astype(float))
        if metric_values.size == 0 or x.size == 0:
            return None
        finite_values = metric_values[np.isfinite(metric_values)]
        if finite_values.size == 0:
            return None
        mean_value = float(np.mean(finite_values))
        std_value = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        return (
            x,
            np.full_like(x, mean_value, dtype=float),
            np.full_like(x, std_value, dtype=float),
            np.full_like(x, min_value, dtype=float),
            np.full_like(x, max_value, dtype=float),
        )

    metric_df = group_df[group_df["metric"] == metric]
    if metric_df.empty:
        return None
    metric_stats = (
        metric_df.groupby("epoch", as_index=False)["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("epoch")
    )
    if metric_stats.empty:
        return None

    return (
        metric_stats["epoch"].to_numpy(dtype=float),
        metric_stats["mean"].to_numpy(dtype=float),
        metric_stats["std"].fillna(0.0).to_numpy(dtype=float),
        metric_stats["min"].to_numpy(dtype=float),
        metric_stats["max"].to_numpy(dtype=float),
    )


def _plot_band_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    std: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    """Draw a mean curve with min-max and standard-deviation bands."""
    std_lower = np.maximum(y - std, min_values)
    std_upper = np.minimum(y + std, max_values)
    ax.fill_between(x, min_values, max_values, color=color, alpha=0.055, linewidth=0)
    ax.fill_between(x, std_lower, std_upper, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)


def _set_loss_ylim_from_values(ax: plt.Axes, values: Sequence[np.ndarray]) -> None:
    """Set a local y-axis range from finite loss values, clipping extreme losses."""
    finite_parts = [
        np.asarray(value, dtype=float).reshape(-1)
        for value in values
        if np.asarray(value, dtype=float).size > 0
    ]
    if not finite_parts:
        return
    finite = np.concatenate(finite_parts)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    finite = finite[finite <= LOSS_AXIS_UPPER_LIMIT]
    if finite.size == 0:
        ax.set_ylim(0.0, LOSS_AXIS_UPPER_LIMIT)
        return

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if np.isclose(y_min, y_max):
        pad = max(0.05, abs(y_min) * 0.05)
    else:
        pad = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - pad, LOSS_AXIS_UPPER_LIMIT)


def _set_loss_xlim_from_values(ax: plt.Axes, values: Sequence[np.ndarray]) -> None:
    """Set a local x-axis range from the finite epoch values drawn in one subplot."""
    finite_parts = [
        np.asarray(value, dtype=float).reshape(-1)
        for value in values
        if np.asarray(value, dtype=float).size > 0
    ]
    if not finite_parts:
        return
    finite = np.concatenate(finite_parts)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return

    x_min = float(np.min(finite))
    x_max = float(np.max(finite))
    if np.isclose(x_min, x_max):
        pad = 1.0
    else:
        pad = 0.03 * (x_max - x_min)
    ax.set_xlim(x_min - pad, x_max + pad)


def _plot_loss_lines(
    ax: plt.Axes,
    fold_history: pd.DataFrame,
    *,
    show_legend: bool,
) -> None:
    """Draw train/validation/test losses for one fold without uncertainty bands."""
    x_values = pd.to_numeric(fold_history["epoch"], errors="coerce")
    for metric in LOSS_METRIC_COLUMNS:
        if metric not in fold_history.columns:
            continue
        y_values = pd.to_numeric(fold_history[metric], errors="coerce")
        if y_values.notna().sum() == 0:
            continue
        style = LOSS_METRIC_STYLES[metric]
        ax.plot(
            x_values,
            y_values,
            label=str(style["label"]),
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=2.0,
        )
    if show_legend:
        ax.legend(loc="best", frameon=False, fontsize="small")


def _style_loss_axis(ax: plt.Axes, *, ylabel: str | None = None) -> None:
    """Apply shared styling for loss-curve axes."""
    ax.set_xlabel("epoch")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_training_curve_grid(
    history: pd.DataFrame,
    output_dir: Path,
    metric_columns: Sequence[str],
    output_name: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> Path | None:
    """Save mean training curves with standard-deviation and min-max bands."""
    available_metrics = [metric for metric in metric_columns if metric in history.columns]
    if history.empty or not available_metrics:
        return None

    plot_df = history.copy()
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    id_vars = [
        "signal_set",
        "signal_set_description",
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
        "fold_id",
        "epoch",
    ]
    long_df = plot_df.melt(
        id_vars=[column for column in id_vars if column in plot_df.columns],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["epoch", "value"])
    if long_df.empty:
        return None

    group_keys = [column for column in ["signal_set", "experiment_id", "experiment_display_name", "cv_strategy"] if column in long_df.columns]
    groups = list(long_df[group_keys].drop_duplicates().itertuples(index=False, name=None))
    if not groups:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / output_name
    figure_title = "Training Progress"
    if len(groups) == 1:
        group = dict(zip(group_keys, groups[0]))
        experiment_name = group.get("experiment_display_name", group.get("experiment_id", "Experiment"))
        cv_strategy = group.get("cv_strategy", "all")
        signal_prefix = f"{group.get('signal_set')} " if "signal_set" in group else ""
        figure_title = f"{signal_prefix}{experiment_name} Training Progress - {cv_strategy}"

    n_rows = len(groups)
    n_cols = len(available_metrics)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(10.0, 4.8 * n_cols), max(4.0, 3.7 * n_rows)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    palette = sns.color_palette("colorblind", n_colors=max(1, long_df["model"].nunique()))
    model_order = _ordered_models(long_df["model"].astype(str).unique().tolist())
    color_by_model = {model: palette[idx % len(palette)] for idx, model in enumerate(model_order)}
    line_style_by_metric = {"test_loss": "--"}
    legend_handles = {}

    for row_idx, group_values in enumerate(groups):
        group = dict(zip(group_keys, group_values))
        experiment_name = group.get("experiment_display_name", group.get("experiment_id", "Experiment"))
        cv_strategy = group.get("cv_strategy", "all")
        group_df = long_df.copy()
        for key, value in group.items():
            if key == "experiment_display_name":
                continue
            group_df = group_df[group_df[key].astype(str) == str(value)]
        for col_idx, metric in enumerate(available_metrics):
            ax = axes[row_idx, col_idx]
            x_limit_values: List[np.ndarray] = []
            y_limit_values: List[np.ndarray] = []
            for model_name in model_order:
                model_long_df = group_df[group_df["model"].astype(str) == model_name]
                if model_long_df.empty:
                    continue
                curve = _compute_aggregated_metric_curve(model_long_df, metric)
                if curve is None:
                    continue
                x, y, std, min_values, max_values = curve
                x_limit_values.append(x)
                y_limit_values.extend([min_values, max_values])
                color = color_by_model.get(model_name)
                _plot_band_curve(
                    ax,
                    x,
                    y,
                    std,
                    min_values,
                    max_values,
                    color=color,
                    label=model_name,
                    linestyle=line_style_by_metric.get(metric, "-"),
                )
                line = ax.lines[-1]
                legend_handles.setdefault(model_name, line)

            if row_idx == 0:
                ax.set_title(_metric_display_name(metric), fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("epoch")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
                if n_rows > 1:
                    ax.text(
                        -0.08,
                        1.04,
                        f"{group.get('signal_set', '') + ' - ' if 'signal_set' in group else ''}{experiment_name} - {cv_strategy}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        fontweight="semibold",
                    )
            if ylim is not None:
                ax.set_ylim(*ylim)
            elif y_limit_values:
                _set_loss_ylim_from_values(ax, y_limit_values)
            if x_limit_values:
                _set_loss_xlim_from_values(ax, x_limit_values)
            _style_loss_axis(ax, ylabel=None)

    if len(legend_handles) > 1:
        fig.legend(
            handles=[legend_handles[model] for model in model_order if model in legend_handles],
            labels=[model for model in model_order if model in legend_handles],
            loc="upper center",
            ncol=min(len(legend_handles), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 1.005),
        )
        title_y = 1.06
    else:
        title_y = 1.03
    fig.suptitle(figure_title, y=title_y, fontsize=14)
    fig.text(
        0.995,
        0.005,
        "bands: darker = mean +/- std, lighter = fold min-max",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_training_loss_combined(history: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save aggregated train/validation/test loss curves in shared panels."""
    available_metrics = [metric for metric in LOSS_METRIC_COLUMNS if metric in history.columns]
    if history.empty or not available_metrics:
        return None

    plot_df = history.copy()
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    id_vars = [
        "signal_set",
        "signal_set_description",
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
        "fold_id",
        "epoch",
    ]
    long_df = plot_df.melt(
        id_vars=[column for column in id_vars if column in plot_df.columns],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["epoch", "value"])
    if long_df.empty:
        return None

    group_keys = [column for column in ["signal_set", "experiment_id", "experiment_display_name", "cv_strategy"] if column in long_df.columns]
    groups = list(long_df[group_keys].drop_duplicates().itertuples(index=False, name=None))
    model_order = _ordered_models(long_df["model"].astype(str).unique().tolist())
    if not groups or not model_order:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "training_progress_loss_combined.png"

    n_rows = len(groups)
    n_cols = len(model_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(7.0, 5.4 * n_cols), max(4.0, 3.8 * n_rows)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    for row_idx, group_values in enumerate(groups):
        group = dict(zip(group_keys, group_values))
        experiment_name = group.get("experiment_display_name", group.get("experiment_id", "Experiment"))
        cv_strategy = group.get("cv_strategy", "all")
        group_df = long_df.copy()
        for key, value in group.items():
            if key == "experiment_display_name":
                continue
            group_df = group_df[group_df[key].astype(str) == str(value)]
        for col_idx, model_name in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            model_df = group_df[group_df["model"].astype(str) == model_name]
            if model_df.empty:
                ax.set_visible(False)
                continue

            x_limit_values: List[np.ndarray] = []
            y_limit_values: List[np.ndarray] = []
            for metric in available_metrics:
                curve = _compute_aggregated_metric_curve(model_df, metric)
                if curve is None:
                    continue
                x, y, std, min_values, max_values = curve
                x_limit_values.append(x)
                y_limit_values.extend([min_values, max_values])
                style = LOSS_METRIC_STYLES[metric]
                _plot_band_curve(
                    ax,
                    x,
                    y,
                    std,
                    min_values,
                    max_values,
                    color=str(style["color"]),
                    label=str(style["label"]),
                    linestyle=str(style["linestyle"]),
                )

            if row_idx == 0:
                ax.set_title(model_name, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("loss")
                if n_rows > 1:
                    ax.text(
                        -0.08,
                        1.04,
                        f"{group.get('signal_set', '') + ' - ' if 'signal_set' in group else ''}{experiment_name} - {cv_strategy}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        fontweight="semibold",
                    )
            if y_limit_values:
                _set_loss_ylim_from_values(ax, y_limit_values)
            if x_limit_values:
                _set_loss_xlim_from_values(ax, x_limit_values)
            _style_loss_axis(ax)
            ax.legend(loc="best", frameon=False, fontsize="small")

    figure_title = "Training Progress Losses"
    if len(groups) == 1:
        group = dict(zip(group_keys, groups[0]))
        experiment_name = group.get("experiment_display_name", group.get("experiment_id", "Experiment"))
        cv_strategy = group.get("cv_strategy", "all")
        signal_prefix = f"{group.get('signal_set')} " if "signal_set" in group else ""
        figure_title = f"{signal_prefix}{experiment_name} Training Losses - {cv_strategy}"
    fig.suptitle(figure_title, y=1.03, fontsize=14)
    fig.text(
        0.995,
        0.005,
        "bands: darker = mean +/- std, lighter = fold min-max",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_fold_loss_plots(history: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Save per-fold train/validation/test loss plots in one combined layout."""
    available_metrics = [metric for metric in LOSS_METRIC_COLUMNS if metric in history.columns]
    required_columns = {
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
        "fold_id",
        "epoch",
    }
    if history.empty or not available_metrics or not required_columns.issubset(history.columns):
        return []

    plot_df = history.copy()
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    losses_dir = output_dir / "plots" / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    group_columns = [
        *([column for column in ["signal_set"] if column in plot_df.columns]),
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
        "fold_id",
    ]

    for group_values, fold_df in plot_df.groupby(group_columns, sort=True):
        group = dict(zip(group_columns, [str(value) for value in group_values]))
        experiment_id = group["experiment_id"]
        experiment_name = group["experiment_display_name"]
        cv_strategy = group["cv_strategy"]
        model_name = group["model"]
        fold_id = group["fold_id"]
        signal_prefix = f"{group['signal_set']}_" if "signal_set" in group else ""
        signal_title_prefix = f"{group['signal_set']} - " if "signal_set" in group else ""
        fold_df = fold_df.sort_values("epoch")
        if fold_df[available_metrics].notna().sum().sum() == 0:
            continue

        filename_base = _slugify_filename(f"{signal_prefix}{experiment_id}_{cv_strategy}_{model_name}_{fold_id}")
        title = f"{signal_title_prefix}{experiment_name} - {cv_strategy} - {model_name} - {fold_id}"

        combined_path = losses_dir / f"{filename_base}_combined.png"
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8))
        _plot_loss_lines(ax, fold_history=fold_df, show_legend=True)
        ax.set_title(title, fontsize=11)
        _style_loss_axis(ax, ylabel="loss")
        fig.tight_layout()
        fig.savefig(combined_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(combined_path)

    return saved_paths


def _plot_best_epoch_distribution(history: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save GNN best-epoch distributions by task and CV strategy."""
    required_columns = {"history_kind", "best_epoch", "experiment_id", "experiment_display_name", "cv_strategy", "model", "fold_id"}
    if history.empty or not required_columns.issubset(history.columns):
        return None

    best_df = history[history["history_kind"] == "gnn"].copy()
    best_df["best_epoch"] = pd.to_numeric(best_df["best_epoch"], errors="coerce")
    best_df = best_df.dropna(subset=["best_epoch"])
    if best_df.empty:
        return None

    best_df = best_df[
        [
            *([column for column in ["signal_set"] if column in best_df.columns]),
            "experiment_id",
            "experiment_display_name",
            "cv_strategy",
            "model",
            "fold_id",
            "best_epoch",
        ]
    ].drop_duplicates()
    if best_df.empty:
        return None

    group_keys = [column for column in ["signal_set", "experiment_id", "experiment_display_name", "cv_strategy"] if column in best_df.columns]
    groups = list(best_df[group_keys].drop_duplicates().itertuples(index=False, name=None))
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "best_epoch_distribution.png"

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(10, max(3.5, 3.0 * len(groups))),
        squeeze=False,
    )
    model_order = _ordered_models(best_df["model"].astype(str).unique().tolist())
    for row_idx, group_values in enumerate(groups):
        group = dict(zip(group_keys, group_values))
        experiment_name = group.get("experiment_display_name", group.get("experiment_id", "Experiment"))
        cv_strategy = group.get("cv_strategy", "all")
        ax = axes[row_idx, 0]
        group_df = best_df.copy()
        for key, value in group.items():
            if key == "experiment_display_name":
                continue
            group_df = group_df[group_df[key].astype(str) == str(value)]
        sns.boxplot(
            data=group_df,
            x="model",
            y="best_epoch",
            order=model_order,
            color="#D9EAF7",
            ax=ax,
        )
        sns.stripplot(
            data=group_df,
            x="model",
            y="best_epoch",
            order=model_order,
            color="#1F4E79",
            size=4,
            jitter=0.15,
            ax=ax,
        )
        signal_prefix = f"{group.get('signal_set')} - " if "signal_set" in group else ""
        ax.set_title(f"{signal_prefix}{experiment_name} Best Epoch - {cv_strategy}")
        ax.set_xlabel("model")
        ax.set_ylabel("best epoch")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_test_loss_summary(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save final held-out test-loss comparison from quick-run summaries."""
    required_columns = {"status", "loss", "experiment_id", "experiment_display_name", "cv_strategy", "model"}
    if summary.empty or not required_columns.issubset(summary.columns):
        return None

    plot_df = summary[summary["status"] == "success"].copy()
    plot_df = plot_df[~plot_df["model"].astype(str).isin({"Random", "Majority"})]
    plot_df["loss"] = pd.to_numeric(plot_df["loss"], errors="coerce")
    plot_df = plot_df.dropna(subset=["loss"])
    if plot_df.empty:
        return None

    group_keys = [column for column in ["signal_set", "experiment_id", "experiment_display_name", "cv_strategy"] if column in plot_df.columns]
    groups = list(plot_df[group_keys].drop_duplicates().itertuples(index=False, name=None))
    if not groups:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "test_loss_by_model.png"

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(10, max(3.5, 3.0 * len(groups))),
        squeeze=False,
    )
    model_order = _ordered_models(plot_df["model"].astype(str).unique().tolist())
    for row_idx, group_values in enumerate(groups):
        group = dict(zip(group_keys, group_values))
        experiment_name = group.get("experiment_display_name", group.get("experiment_id", "Experiment"))
        cv_strategy = group.get("cv_strategy", "all")
        ax = axes[row_idx, 0]
        group_df = plot_df.copy()
        for key, value in group.items():
            if key == "experiment_display_name":
                continue
            group_df = group_df[group_df[key].astype(str) == str(value)]
        sns.barplot(
            data=group_df,
            x="model",
            y="loss",
            order=model_order,
            color="#D9EAF7",
            ax=ax,
        )
        sns.stripplot(
            data=group_df,
            x="model",
            y="loss",
            order=model_order,
            color="#1F4E79",
            size=4,
            jitter=0.1,
            ax=ax,
        )
        signal_prefix = f"{group.get('signal_set')} - " if "signal_set" in group else ""
        ax.set_title(f"{signal_prefix}{experiment_name} Held-out Test Loss - {cv_strategy}")
        ax.set_xlabel("model")
        ax.set_ylabel("test loss")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_training_history_outputs(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    save_plots: bool = True,
    save_fold_loss_plots: bool = False,
) -> List[Path]:
    """Save command-level training-history table and plots."""
    history = _collect_training_history(rows)
    if history.empty:
        return _save_training_diagnostics_outputs(rows=rows, output_dir=output_dir)

    saved_paths: List[Path] = []
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    history_path = tables_dir / "training_history.csv"
    history.to_csv(history_path, index=False)
    saved_paths.append(history_path)

    if save_plots:
        for plot_path in [
            _plot_training_curve_grid(
                history=history,
                output_dir=output_dir,
                metric_columns=LOSS_METRIC_COLUMNS,
                output_name="training_progress_loss.png",
                ylabel="loss",
            ),
            _plot_training_loss_combined(history=history, output_dir=output_dir),
            _plot_training_curve_grid(
                history=history,
                output_dir=output_dir,
                metric_columns=["val_balanced_accuracy", "val_macro_f1"],
                output_name="training_progress_validation_metrics.png",
                ylabel="validation metric",
                ylim=(0.0, 1.0),
            ),
            _plot_best_epoch_distribution(history=history, output_dir=output_dir),
        ]:
            if plot_path is not None:
                saved_paths.append(plot_path)

    if save_fold_loss_plots:
        saved_paths.extend(_save_fold_loss_plots(history=history, output_dir=output_dir))
    saved_paths.extend(_save_training_diagnostics_outputs(rows=rows, output_dir=output_dir))

    return saved_paths


def _save_training_diagnostics_outputs(rows: Sequence[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Save command-level GNN diagnostic tables when fold diagnostics exist."""
    diagnostics = _collect_training_diagnostics(rows)
    if diagnostics.empty:
        return []

    saved_paths: List[Path] = []
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = tables_dir / "training_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    saved_paths.append(diagnostics_path)

    group_columns = [
        *([column for column in ["signal_set", "signal_set_description"] if column in diagnostics.columns]),
        "experiment_id",
        "experiment_display_name",
        "cv_strategy",
        "model",
        "split",
    ]
    numeric_columns = [
        column
        for column in diagnostics.columns
        if column not in set(group_columns)
        and pd.api.types.is_numeric_dtype(diagnostics[column])
    ]
    if numeric_columns:
        summary = (
            diagnostics[group_columns + numeric_columns]
            .groupby(group_columns, dropna=False)
            .agg(["mean", "std"])
        )
        summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
        summary = summary.reset_index()
        summary_path = tables_dir / "training_diagnostics_summary.csv"
        summary.to_csv(summary_path, index=False)
        saved_paths.append(summary_path)

    return saved_paths


def _save_combined_confusion_matrices(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    experiment_id: str,
    cv_strategy: str,
    signal_set: str | None = None,
    use_strategy_suffix: bool = True,
) -> Path | None:
    """Save confusion matrices comparing all successful quick-run model types."""
    row_by_model = {
        str(row["model"]): row
        for row in rows
        if row.get("status") == "success"
        and str(row.get("experiment_id")) == experiment_id
        and str(row.get("cv_strategy")) == cv_strategy
        and (signal_set is None or str(row.get("signal_set", "")) == signal_set)
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
    signal_slug = f"{signal_set}_" if signal_set else ""
    if use_strategy_suffix:
        filename = f"confusion_matrices_{signal_slug}{experiment_slug}_{cv_strategy}.png"
    else:
        filename = f"confusion_matrices_{signal_slug}{experiment_slug}.png"
    output_path = figures_dir / filename

    n_models = len(collected)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models))
    if n_models == 1:
        axes = np.asarray([axes])

    for row_idx, (model_name, (y_true, y_pred)) in enumerate(collected.items()):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        display_model_name = _thesis_model_display_name(model_name)

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
        signal_title = f"{signal_set} - " if signal_set else ""
        abs_ax.set_title(f"{signal_title}{display_model_name} - {cv_strategy} (absolute)")
        abs_ax.tick_params(axis="y", rotation=90)
        for label in abs_ax.get_yticklabels():
            label.set_horizontalalignment("center")
            label.set_verticalalignment("center")

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
        norm_ax.set_title(f"{signal_title}{display_model_name} - {cv_strategy} (row-normalized)")
        norm_ax.tick_params(axis="y", rotation=90)
        for label in norm_ax.get_yticklabels():
            label.set_horizontalalignment("center")
            label.set_verticalalignment("center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_confusion_matrix_table(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Build absolute and row-normalized confusion-matrix records from saved predictions."""
    records: List[Dict[str, Any]] = []
    successful_rows = [row for row in rows if row.get("status") == "success"]

    for row in successful_rows:
        experiment_id = str(row.get("experiment_id", ""))
        cv_strategy = str(row.get("cv_strategy", ""))
        model_name = str(row.get("model", ""))
        summary_model_name = str(row.get("summary_model_name", ""))
        signal_set = str(row.get("signal_set", ""))
        signal_set_description = str(row.get("signal_set_description", ""))
        suite_run_dir = Path(str(row.get("suite_run_dir", "")))

        try:
            trainer_run_dir = _resolve_trainer_run_dir(suite_run_dir, experiment_id=experiment_id)
            class_display_names = _load_class_display_names(trainer_run_dir)
            y_true, y_pred = _collect_predictions_for_variant(
                trainer_run_dir=trainer_run_dir,
                cv_strategy=cv_strategy,
                summary_model_name=summary_model_name,
            )
        except Exception:
            continue

        if y_true.size == 0:
            continue

        observed_classes = {int(class_idx) for class_idx in np.unique(np.concatenate([y_true, y_pred])).tolist()}
        classes = np.asarray(sorted(observed_classes | set(class_display_names.keys())), dtype=int)
        if classes.size == 0:
            continue

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        for true_position, true_class_idx in enumerate(classes):
            for pred_position, pred_class_idx in enumerate(classes):
                records.append(
                    {
                        "signal_set": signal_set,
                        "signal_set_description": signal_set_description,
                        "experiment_id": experiment_id,
                        "experiment_display_name": EXPERIMENT_DISPLAY_NAMES.get(experiment_id, experiment_id),
                        "cv_strategy": cv_strategy,
                        "model": model_name,
                        "summary_model_name": summary_model_name,
                        "true_class_index": int(true_class_idx),
                        "true_class_name": class_display_names.get(int(true_class_idx), str(int(true_class_idx))),
                        "pred_class_index": int(pred_class_idx),
                        "pred_class_name": class_display_names.get(int(pred_class_idx), str(int(pred_class_idx))),
                        "count": int(cm[true_position, pred_position]),
                        "row_normalized": float(cm_norm[true_position, pred_position]),
                    }
                )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        [
            "experiment_id",
            "signal_set",
            "cv_strategy",
            "model",
            "true_class_index",
            "pred_class_index",
        ]
    ).reset_index(drop=True)


def _save_confusion_matrix_table(rows: Sequence[Dict[str, Any]], output_dir: Path) -> Path | None:
    """Save a top-level numeric confusion-matrix table for thesis figures."""
    confusion_table = _build_confusion_matrix_table(rows)
    if confusion_table.empty:
        return None

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "confusion_matrices.csv"
    confusion_table.to_csv(output_path, index=False)
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
    report_profile = _resolve_report_profile(args)
    save_fold_loss_plots = _should_save_fold_loss_plots(args, report_profile)
    fixed_overrides = build_fixed_overrides(args, run_output_dir=output_dir)
    requested_model_names = _resolve_requested_models(base_cfg, args.models)
    signal_sets = [build_signal_set_variant(name) for name in _resolve_requested_signal_sets(base_cfg, args.signal_sets)]

    rows: List[Dict[str, Any]] = []
    for signal_set in signal_sets:
        signal_model_names = _model_names_for_signal_set(requested_model_names, signal_set.name)
        quick_runs = build_quick_runs(signal_model_names)
        signal_fixed_overrides = merge_many(
            fixed_overrides,
            {"suite": {"results_dir": str(output_dir / "model_runs" / signal_set.name)}},
        )
        signal_generated_dir = generated_dir / signal_set.name
        signal_generated_dir.mkdir(parents=True, exist_ok=True)

        for quick_run in quick_runs:
            payload = _build_run_payload(
                base_cfg=base_cfg,
                fixed_overrides=signal_fixed_overrides,
                signal_set=signal_set,
                quick_run=quick_run,
            )
            cv_strategies = _get_cv_strategies(payload)
            experiment_ids = _get_enabled_experiment_ids(payload)
            config_path = signal_generated_dir / f"{quick_run.run_name}.yaml"
            _write_yaml(config_path, payload)

            base_row: Dict[str, Any] = {
                "signal_set": signal_set.name,
                "signal_set_description": signal_set.description,
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
                    f"dry-run | signal_set={signal_set.name} | {quick_run.run_name} "
                    f"| models={','.join(quick_run.model_names)} "
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
                    f"{row['status']} | {row.get('signal_set')} | {row.get('experiment_id')} "
                    f"| {row['cv_strategy']} | {row['model']} "
                    f"| balanced_accuracy={row.get('balanced_accuracy', np.nan)}"
                )

    summary = _rows_to_dataframe(rows)
    summary_path = output_dir / "quick_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    thesis_metrics_path: Path | None = None
    thesis_heatmap_paths: List[Path] = []
    ranking_path: Path | None = None
    interactive_ranking_path: Path | None = None
    test_loss_path: Path | None = None
    if report_profile in {"thesis", "full"}:
        thesis_metrics_path = _save_thesis_metric_table(summary=summary, output_dir=output_dir)
        if thesis_metrics_path is not None:
            thesis_metrics = pd.read_csv(thesis_metrics_path)
            thesis_heatmap_paths = _save_thesis_metric_heatmaps(thesis_metrics=thesis_metrics, output_dir=output_dir)
    if report_profile == "full":
        ranking_path = _save_group_model_ranking(summary=summary, output_dir=output_dir)
        interactive_ranking_path = _save_group_model_ranking_interactive(summary=summary, output_dir=output_dir)
    if report_profile in {"thesis", "full"}:
        test_loss_path = _plot_test_loss_summary(summary=summary, output_dir=output_dir)
    confusion_paths: List[Path] = []
    confusion_table_path: Path | None = None
    fold_metric_paths: List[Path] = []
    benchmark_paths: List[Path] = []
    label_distribution_paths: List[Path] = []
    training_history_paths: List[Path] = []
    if not args.dry_run:
        fold_metric_paths = _save_fold_metric_outputs(rows=rows, output_dir=output_dir)
        benchmark_paths = _save_model_benchmark_outputs(
            rows=rows,
            quick_summary=summary,
            output_dir=output_dir,
        )
        confusion_table_path = _save_confusion_matrix_table(rows=rows, output_dir=output_dir)
        if report_profile in {"thesis", "full"}:
            label_distribution_paths = _save_label_distribution_outputs(rows=rows, output_dir=output_dir)
        training_history_paths = _save_training_history_outputs(
            rows=rows,
            output_dir=output_dir,
            save_plots=report_profile in {"thesis", "full"},
            save_fold_loss_plots=save_fold_loss_plots,
        )
        if report_profile in {"thesis", "full"}:
            successful_pairs = list(
                dict.fromkeys(
                    (
                        str(row["signal_set"]),
                        str(row["experiment_id"]),
                        str(row["cv_strategy"]),
                    )
                    for row in rows
                    if str(row.get("status")) == "success"
                )
            )
            successful_strategies = [cv_strategy for _, _, cv_strategy in successful_pairs]
            use_strategy_suffix = len(set(successful_strategies)) > 1 or len(successful_pairs) > 1
            for signal_set_name, experiment_id, cv_strategy in successful_pairs:
                confusion_path = _save_combined_confusion_matrices(
                    rows=rows,
                    output_dir=output_dir,
                    experiment_id=experiment_id,
                    cv_strategy=cv_strategy,
                    signal_set=signal_set_name,
                    use_strategy_suffix=use_strategy_suffix,
                )
                if confusion_path is not None:
                    confusion_paths.append(confusion_path)
    print(f"Saved quick comparison summary: {summary_path}")
    if thesis_metrics_path is not None:
        print(f"Saved thesis metric table: {thesis_metrics_path}")
    for thesis_heatmap_path in thesis_heatmap_paths:
        print(f"Saved thesis heatmap: {thesis_heatmap_path}")
    if ranking_path is not None:
        print(f"Saved ranking plot: {ranking_path}")
    if interactive_ranking_path is not None:
        print(f"Saved interactive ranking plot: {interactive_ranking_path}")
    if test_loss_path is not None:
        print(f"Saved test loss plot: {test_loss_path}")
    for fold_metric_path in fold_metric_paths:
        print(f"Saved fold metric output: {fold_metric_path}")
    for benchmark_path in benchmark_paths:
        print(f"Saved benchmark output: {benchmark_path}")
    if confusion_table_path is not None:
        print(f"Saved confusion matrix table: {confusion_table_path}")
    for label_distribution_path in label_distribution_paths:
        print(f"Saved label distribution output: {label_distribution_path}")
    for training_history_path in training_history_paths:
        print(f"Saved training history output: {training_history_path}")
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
