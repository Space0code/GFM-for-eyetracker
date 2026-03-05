"""Utilities to generate and save binary-training result plots.

This module recreates the visualizations from `results_analysis.ipynb`:
1. Per-strategy metric bar plots from `summary.csv`.
2. Per-model confusion matrices (absolute and row-normalized) aggregated across folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _discover_strategy_dirs(run_dir: Path, summary_file: str) -> List[Path]:
    """Return strategy directories that contain a summary CSV."""
    return sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and (path / summary_file).exists()]
    )


def _load_results_df(strategy_dirs: Sequence[Path], summary_file: str, run_name: str) -> pd.DataFrame:
    """Load and concatenate strategy summaries into a single dataframe."""
    all_results: List[pd.DataFrame] = []
    for strategy_dir in strategy_dirs:
        df = pd.read_csv(strategy_dir / summary_file)
        if "metric_type" in df.columns:
            df = df[df["metric_type"] != "aggregated"]
        df["strategy"] = strategy_dir.name
        df["run"] = run_name
        all_results.append(df)

    if not all_results:
        raise RuntimeError("No summary data loaded.")
    return pd.concat(all_results, ignore_index=True)


def _infer_models_for_cm(strategy_dirs: Sequence[Path], preferred_order: Sequence[str]) -> List[str]:
    """Infer available models from fold outputs with a stable, readable order."""
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

    ordered_models: List[str] = []
    if has_gnn:
        ordered_models.append("GNN")

    for model_name in preferred_order:
        if model_name in discovered_models:
            ordered_models.append(model_name)

    remaining = sorted(model_name for model_name in discovered_models if model_name not in preferred_order)
    ordered_models.extend(remaining)
    return ordered_models


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save figure to disk and close it to release memory."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_metrics_barplots(
    results_df: pd.DataFrame,
    strategies: Sequence[str],
    run_name: str,
    output_path: Path,
    candidate_metrics: Sequence[str],
) -> Path:
    """Create and save per-strategy bar plots for selected summary metrics."""
    plot_metrics = [metric for metric in candidate_metrics if metric in results_df.columns]
    if not plot_metrics:
        raise RuntimeError("No metric columns found to plot.")

    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 4), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        subdir_results = results_df[results_df["strategy"] == strategy]
        if subdir_results.empty:
            ax.set_axis_off()
            ax.set_title(f"{strategy} (no data)")
            continue

        subdir_results.plot(x="model", y=plot_metrics, kind="bar", ax=ax)
        ax.set_title(f"{run_name} - {strategy.replace('_', ' ')}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Metric Value")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    return _save_figure(fig, output_path)


def _plot_confusion_matrices(
    run_dir: Path,
    strategies: Sequence[str],
    models_for_cm: Sequence[str],
    decision_threshold: float,
    output_path: Path,
) -> Path:
    """Create and save absolute + row-normalized confusion matrices."""
    n_models = len(models_for_cm)
    n_strategies = len(strategies)
    n_cols = 2 * n_strategies

    fig, axes = plt.subplots(n_models, n_cols, figsize=(5 * n_cols, 4 * n_models))
    if n_models == 1:
        axes = np.array([axes])

    for row_idx, model_name in enumerate(models_for_cm):
        for strategy_idx, strategy in enumerate(strategies):
            abs_ax = axes[row_idx, 2 * strategy_idx]
            rel_ax = axes[row_idx, 2 * strategy_idx + 1]
            strategy_path = run_dir / strategy
            fold_dirs = sorted([path for path in strategy_path.iterdir() if path.is_dir()])

            all_predictions: List[np.ndarray] = []
            all_targets: List[np.ndarray] = []

            for fold_dir in fold_dirs:
                if model_name == "GNN":
                    pred_path = fold_dir / "test_predictions.npy"
                    target_path = fold_dir / "test_targets.npy"
                else:
                    pred_path = fold_dir / "baselines" / model_name / "test_predictions.npy"
                    target_path = fold_dir / "baselines" / model_name / "test_targets.npy"

                if pred_path.exists() and target_path.exists():
                    preds = np.asarray(np.load(pred_path)).reshape(-1)
                    targets = np.asarray(np.load(target_path)).reshape(-1)
                    all_predictions.append(preds)
                    all_targets.append(targets)

            if all_predictions and all_targets:
                predictions = np.concatenate(all_predictions)
                targets = np.concatenate(all_targets).astype(int)
                pred_classes = (predictions >= decision_threshold).astype(int)

                cm = confusion_matrix(targets, pred_classes, labels=[0, 1])
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_row_norm = np.divide(
                    cm,
                    row_sums,
                    out=np.zeros_like(cm, dtype=float),
                    where=row_sums != 0,
                )

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    ax=abs_ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                )
                abs_ax.set_xlabel("Predicted Label")
                abs_ax.set_ylabel("True Label")
                abs_ax.set_title(f"{model_name} - {strategy.replace('_', ' ')} (Absolute)")

                sns.heatmap(
                    cm_row_norm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    vmin=0.0,
                    vmax=1.0,
                    cbar=False,
                    ax=rel_ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                )
                rel_ax.set_xlabel("Predicted Label")
                rel_ax.set_ylabel("True Label")
                rel_ax.set_title(f"{model_name} - {strategy.replace('_', ' ')} (Row-normalized)")
            else:
                abs_ax.set_axis_off()
                rel_ax.set_axis_off()
                abs_ax.set_title(f"{model_name} - {strategy} (no prediction files)")
                rel_ax.set_title(f"{model_name} - {strategy} (no prediction files)")

    plt.tight_layout()
    return _save_figure(fig, output_path)


def generate_and_save_binary_results_plots(
    run_dir: Path | str,
    decision_threshold: float,
    models_for_cm: Sequence[str] | None = None,
    summary_file: str = "summary.csv",
    figures_dir_name: str = "figures",
    candidate_metrics: Sequence[str] = ("accuracy", "balanced_accuracy", "f1", "auc", "precision", "recall"),
    preferred_baseline_order: Sequence[str] = ("Mean", "SVM", "LightGBM", "MLP", "GNN"),
) -> List[Path]:
    """Generate all post-training plots and save them under `<run_dir>/figures`.

    Args:
        run_dir: Path to one training run directory (e.g., `results/binary/<timestamp>`).
        decision_threshold: Probability threshold used to binarize predictions for confusion matrices.
        models_for_cm: Optional explicit model order for confusion matrices.
        summary_file: Summary CSV filename expected under each strategy directory.
        figures_dir_name: Subdirectory name under `run_dir` for saved figures.
        candidate_metrics: Metrics to include in bar plots when present in summary files.
        preferred_baseline_order: Preferred display order for discovered baseline models.

    Returns:
        List of saved figure paths.
    """
    run_dir_path = Path(run_dir)
    strategy_dirs = _discover_strategy_dirs(run_dir_path, summary_file=summary_file)
    if not strategy_dirs:
        raise RuntimeError(f"No strategy folders with {summary_file} found under {run_dir_path}.")

    strategies = [path.name for path in strategy_dirs]
    results_df = _load_results_df(strategy_dirs, summary_file=summary_file, run_name=run_dir_path.name)

    figures_dir = run_dir_path / figures_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    saved_paths.append(
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
    if models:
        saved_paths.append(
            _plot_confusion_matrices(
                run_dir=run_dir_path,
                strategies=strategies,
                models_for_cm=models,
                decision_threshold=float(decision_threshold),
                output_path=figures_dir / "confusion_matrices.png",
            )
        )

    return saved_paths
