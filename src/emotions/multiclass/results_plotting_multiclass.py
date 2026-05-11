"""Plotting helpers for multiclass training results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix

from emotions.label_names import (
    build_encoded_class_name_mapping,
    resolve_multiclass_label_name_mapping,
)


def _discover_strategy_dirs(run_dir: Path, summary_file: str) -> List[Path]:
    return sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and (path / summary_file).exists()]
    )


def _load_results_df(strategy_dirs: Sequence[Path], summary_file: str, run_name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for strategy_dir in strategy_dirs:
        df = pd.read_csv(strategy_dir / summary_file)
        if "metric_type" in df.columns:
            df = df[df["metric_type"] != "aggregated"]
        df["strategy"] = strategy_dir.name
        df["run"] = run_name
        frames.append(df)
    if not frames:
        raise RuntimeError("No strategy summaries found for multiclass plotting.")
    return pd.concat(frames, ignore_index=True)


def _infer_models_for_cm(strategy_dirs: Sequence[Path], preferred_order: Sequence[str]) -> List[str]:
    discovered_models: set[str] = set()
    has_gnn = False

    for strategy_dir in strategy_dirs:
        fold_dirs = [path for path in strategy_dir.iterdir() if path.is_dir()]
        for fold_dir in fold_dirs:
            if (fold_dir / "test_predictions.npy").exists() and (fold_dir / "test_targets.npy").exists():
                has_gnn = True
            baselines_dir = fold_dir / "baselines"
            if baselines_dir.exists():
                for model_dir in baselines_dir.iterdir():
                    if model_dir.is_dir():
                        discovered_models.add(model_dir.name)

    models: List[str] = []
    if has_gnn:
        models.append("GNN")
    for model_name in preferred_order:
        if model_name in discovered_models:
            models.append(model_name)
    models.extend(sorted([m for m in discovered_models if m not in preferred_order]))
    return models


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_class_display_names(run_dir: Path) -> Dict[int, str]:
    """Load encoded-class display names from run metadata or infer from config."""
    metadata_path = run_dir / "class_metadata.yaml"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if isinstance(payload, dict):
            index_to_name = payload.get("index_to_name")
            if isinstance(index_to_name, dict):
                mapping: Dict[int, str] = {}
                for raw_idx, raw_name in index_to_name.items():
                    try:
                        idx = int(raw_idx)
                    except (TypeError, ValueError):
                        continue
                    name = str(raw_name).strip()
                    if not name:
                        continue
                    mapping[idx] = name
                if mapping:
                    return mapping

    # Fallback for older runs that do not have class_metadata.yaml.
    return _infer_class_display_names_from_config(run_dir)


def _resolve_data_filepath(run_dir: Path, configured_path: str) -> Path | None:
    """Resolve dataset CSV path from config in a robust way."""
    configured = Path(configured_path)
    suite_experiments_candidate: Path | None = None
    if "experiments" in configured.parts:
        exp_idx = configured.parts.index("experiments")
        suite_experiments_candidate = run_dir.parent / Path(*configured.parts[exp_idx:])

    candidates = [
        configured,
        run_dir / configured,
        Path.cwd() / configured,
    ]
    if suite_experiments_candidate is not None:
        candidates.append(suite_experiments_candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _infer_class_display_names_from_config(run_dir: Path) -> Dict[int, str]:
    """Infer encoded-class display names for old runs from config + snapshot."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        return {}

    task_cfg = config.get("multiclass_task")
    dataset_cfg = config.get("dataset")
    if not isinstance(task_cfg, dict):
        return {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}

    raw_label_names = resolve_multiclass_label_name_mapping(
        multiclass_task_cfg=task_cfg,
        dataset_cfg=dataset_cfg,
    )

    task_name = str(task_cfg.get("task_name", "emotion-id")).strip().lower().replace("_", "-")
    if task_name in {"va-quadrant", "va-quadrants", "va-quadrant-4", "va-quadrant4"}:
        return build_encoded_class_name_mapping(
            unique_raw_labels=[0, 1, 2, 3],
            raw_label_name_mapping=raw_label_names,
        )

    target_column = str(task_cfg.get("target_column", "emotion-id"))
    data_filepath_raw = dataset_cfg.get("data_filepath")
    if not isinstance(data_filepath_raw, str) or not data_filepath_raw.strip():
        return {}

    data_path = _resolve_data_filepath(run_dir=run_dir, configured_path=data_filepath_raw)
    if data_path is None:
        return {}

    try:
        labels_series = pd.read_csv(data_path, usecols=[target_column])[target_column]
    except Exception:
        return {}

    labels_numeric = pd.to_numeric(labels_series, errors="coerce").dropna()
    if labels_numeric.empty:
        return {}

    unique_raw_labels = sorted(np.unique(np.round(labels_numeric).astype(int)).tolist())
    return build_encoded_class_name_mapping(
        unique_raw_labels=unique_raw_labels,
        raw_label_name_mapping=raw_label_names,
    )


def _plot_metrics_barplots(
    results_df: pd.DataFrame,
    strategies: Sequence[str],
    run_name: str,
    output_path: Path,
    candidate_metrics: Sequence[str],
) -> Path:
    metrics = [metric for metric in candidate_metrics if metric in results_df.columns]
    if not metrics:
        raise RuntimeError("No multiclass metrics available for barplot.")

    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 4), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        subset = results_df[results_df["strategy"] == strategy]
        if subset.empty:
            ax.set_axis_off()
            ax.set_title(f"{strategy} (no data)")
            continue
        subset.plot(x="model", y=metrics, kind="bar", ax=ax)
        ax.set_title(f"{run_name} - {strategy}")
        ax.set_xlabel("model")
        ax.set_ylabel("metric")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    return _save_figure(fig, output_path)


def _plot_confusion_matrices(
    run_dir: Path,
    strategies: Sequence[str],
    models_for_cm: Sequence[str],
    class_display_names: Mapping[int, str] | None,
    output_path: Path,
) -> Path:
    n_models = len(models_for_cm)
    n_cols = 2 * len(strategies)

    fig, axes = plt.subplots(n_models, n_cols, figsize=(5 * n_cols, 4 * n_models))
    if n_models == 1:
        axes = np.array([axes])

    for row_idx, model_name in enumerate(models_for_cm):
        for strategy_idx, strategy in enumerate(strategies):
            abs_ax = axes[row_idx, 2 * strategy_idx]
            norm_ax = axes[row_idx, 2 * strategy_idx + 1]

            fold_dirs = sorted([path for path in (run_dir / strategy).iterdir() if path.is_dir()])
            all_targets: List[np.ndarray] = []
            all_preds: List[np.ndarray] = []

            for fold_dir in fold_dirs:
                if model_name == "GNN":
                    pred_path = fold_dir / "test_predictions.npy"
                    target_path = fold_dir / "test_targets.npy"
                else:
                    pred_path = fold_dir / "baselines" / model_name / "test_predictions.npy"
                    target_path = fold_dir / "baselines" / model_name / "test_targets.npy"

                if not pred_path.exists() or not target_path.exists():
                    continue

                pred = np.asarray(np.load(pred_path))
                target = np.asarray(np.load(target_path)).reshape(-1)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

                pred_labels = np.argmax(pred, axis=1)
                all_targets.append(target.astype(int))
                all_preds.append(pred_labels.astype(int))

            if not all_targets:
                abs_ax.set_axis_off()
                norm_ax.set_axis_off()
                abs_ax.set_title(f"{model_name} - {strategy} (no predictions)")
                norm_ax.set_title(f"{model_name} - {strategy} (no predictions)")
                continue

            y_true = np.concatenate(all_targets)
            y_pred = np.concatenate(all_preds)
            classes = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            tick_labels = [
                class_display_names.get(int(class_idx), str(int(class_idx)))
                if class_display_names
                else str(int(class_idx))
                for class_idx in classes
            ]

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

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
            abs_ax.set_title(f"{model_name} - {strategy} (absolute)")

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
            norm_ax.set_title(f"{model_name} - {strategy} (row-normalized)")

    fig.tight_layout()
    return _save_figure(fig, output_path)


def generate_and_save_multiclass_results_plots(
    run_dir: str | Path,
    models_for_cm: Sequence[str] | None = None,
    class_display_names: Mapping[int, str] | None = None,
    summary_file: str = "summary.csv",
    figures_dir_name: str = "plots",
    candidate_metrics: Sequence[str] = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auc_ovr",
        "weighted_auc_ovr",
    ),
    preferred_baseline_order: Sequence[str] = ("Mean", "SVM", "LightGBM", "MLP", "GNN"),
) -> List[Path]:
    """Generate and save multiclass summary and confusion-matrix figures."""
    run_dir_path = Path(run_dir)
    strategy_dirs = _discover_strategy_dirs(run_dir_path, summary_file=summary_file)
    if not strategy_dirs:
        raise RuntimeError(f"No strategy folders with {summary_file} found under {run_dir_path}.")

    strategies = [path.name for path in strategy_dirs]
    results_df = _load_results_df(strategy_dirs, summary_file=summary_file, run_name=run_dir_path.name)

    figures_dir = run_dir_path / figures_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    saved.append(
        _plot_metrics_barplots(
            results_df=results_df,
            strategies=strategies,
            run_name=run_dir_path.name,
            output_path=figures_dir / "metrics_barplots.png",
            candidate_metrics=candidate_metrics,
        )
    )

    models = list(models_for_cm) if models_for_cm is not None else _infer_models_for_cm(
        strategy_dirs=strategy_dirs,
        preferred_order=preferred_baseline_order,
    )
    resolved_class_display_names = (
        dict(class_display_names)
        if class_display_names is not None
        else _load_class_display_names(run_dir_path)
    )
    if models:
        saved.append(
            _plot_confusion_matrices(
                run_dir=run_dir_path,
                strategies=strategies,
                models_for_cm=models,
                class_display_names=resolved_class_display_names,
                output_path=figures_dir / "confusion_matrices.png",
            )
        )

    return saved
