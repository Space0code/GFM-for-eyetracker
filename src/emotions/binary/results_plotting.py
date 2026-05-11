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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def _resolve_binary_outcome_labels(y_true: np.ndarray, y_pred_cls: np.ndarray) -> np.ndarray:
    """Map binary (true, pred) pairs to TP/FP/TN/FN labels."""
    y_true = y_true.astype(int).reshape(-1)
    y_pred_cls = y_pred_cls.astype(int).reshape(-1)
    labels = np.empty(shape=y_true.shape[0], dtype=object)

    labels[(y_true == 1) & (y_pred_cls == 1)] = "TP"
    labels[(y_true == 0) & (y_pred_cls == 1)] = "FP"
    labels[(y_true == 0) & (y_pred_cls == 0)] = "TN"
    labels[(y_true == 1) & (y_pred_cls == 0)] = "FN"
    return labels


def _load_gnn_analysis_artifacts_by_strategy(
    run_dir: Path,
    strategy: str,
    decision_threshold: float,
) -> dict[str, np.ndarray] | None:
    """Load and concatenate fold-level GNN analysis artifacts for one strategy."""
    strategy_dir = run_dir / strategy
    if not strategy_dir.exists():
        return None

    pred_parts: List[np.ndarray] = []
    true_parts: List[np.ndarray] = []
    raw_parts: List[np.ndarray] = []
    emb_parts: List[np.ndarray] = []
    subject_parts: List[np.ndarray] = []
    recording_parts: List[np.ndarray] = []

    for fold_dir in sorted([path for path in strategy_dir.iterdir() if path.is_dir()]):
        artifact_path = fold_dir / "gnn_test_analysis_artifacts.npz"
        if not artifact_path.exists():
            continue

        with np.load(artifact_path, allow_pickle=True) as payload:
            if "pred_proba" not in payload or "y_true" not in payload:
                continue
            pred = np.asarray(payload["pred_proba"]).reshape(-1)
            true = np.asarray(payload["y_true"]).reshape(-1)
            if pred.shape[0] != true.shape[0]:
                continue

            pred_parts.append(pred)
            true_parts.append(true)

            if "raw_window_means" in payload:
                raw = np.asarray(payload["raw_window_means"])
                if raw.ndim == 2 and raw.shape[0] == pred.shape[0]:
                    raw_parts.append(raw)

            if "graph_embeddings" in payload:
                emb = np.asarray(payload["graph_embeddings"])
                if emb.ndim == 2 and emb.shape[0] == pred.shape[0]:
                    emb_parts.append(emb)

            if "subjects" in payload:
                subjects = np.asarray(payload["subjects"]).reshape(-1)
                if subjects.shape[0] == pred.shape[0]:
                    subject_parts.append(subjects.astype(str))

            if "recordings" in payload:
                recordings = np.asarray(payload["recordings"]).reshape(-1)
                if recordings.shape[0] == pred.shape[0]:
                    recording_parts.append(recordings.astype(str))

    if not pred_parts:
        return None

    pred_all = np.concatenate(pred_parts, axis=0)
    true_all = np.concatenate(true_parts, axis=0)
    pred_cls = (pred_all >= float(decision_threshold)).astype(int)
    outcome = _resolve_binary_outcome_labels(y_true=true_all, y_pred_cls=pred_cls)

    data: dict[str, np.ndarray] = {
        "pred_proba": pred_all,
        "y_true": true_all.astype(int),
        "y_pred_cls": pred_cls,
        "outcome": outcome.astype(str),
    }
    if raw_parts:
        raw_all = np.concatenate(raw_parts, axis=0)
        if raw_all.shape[0] == pred_all.shape[0]:
            data["raw_window_means"] = raw_all
    if emb_parts:
        emb_all = np.concatenate(emb_parts, axis=0)
        if emb_all.shape[0] == pred_all.shape[0]:
            data["graph_embeddings"] = emb_all
    if subject_parts:
        subjects_all = np.concatenate(subject_parts, axis=0)
        if subjects_all.shape[0] == pred_all.shape[0]:
            data["subjects"] = subjects_all
    if recording_parts:
        recordings_all = np.concatenate(recording_parts, axis=0)
        if recordings_all.shape[0] == pred_all.shape[0]:
            data["recordings"] = recordings_all
    return data


def _project_embeddings_to_2d(
    embeddings: np.ndarray,
    method: str,
) -> np.ndarray:
    """Project high-dimensional embeddings to 2D for visualization."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={embeddings.shape}.")
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2), dtype=float)

    method_key = method.strip().lower()
    if method_key == "pca":
        projector = PCA(n_components=2, random_state=42)
        return projector.fit_transform(embeddings)
    if method_key == "tsne":
        n_samples = embeddings.shape[0]
        if n_samples < 3:
            return np.zeros((n_samples, 2), dtype=float)
        perplexity = max(2.0, min(30.0, float(n_samples - 1) / 3.0))
        projector = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        return projector.fit_transform(embeddings)
    raise ValueError(f"Unsupported embedding projection method '{method}'. Use 'pca' or 'tsne'.")


def _scatter_by_outcome(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    outcome_labels: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plot 2D points colored by TP/FP/TN/FN outcome."""
    color_map = {
        "TP": "#2ca02c",
        "FP": "#d62728",
        "TN": "#1f77b4",
        "FN": "#ff7f0e",
    }
    order = ["TP", "FP", "TN", "FN"]

    for label in order:
        mask = outcome_labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=16,
            alpha=0.75,
            c=color_map[label],
            label=f"{label} (n={int(np.sum(mask))})",
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.25)


def _plot_gnn_tp_fp_tn_fn_scatter(
    run_dir: Path,
    strategies: Sequence[str],
    decision_threshold: float,
    embedding_method: str,
    output_dir: Path,
) -> List[Path]:
    """Create GNN-only TP/FP/TN/FN scatter plots aggregated across folds by strategy."""
    saved_paths: List[Path] = []

    for strategy in strategies:
        payload = _load_gnn_analysis_artifacts_by_strategy(
            run_dir=run_dir,
            strategy=strategy,
            decision_threshold=decision_threshold,
        )
        if payload is None:
            continue

        outcome = payload["outcome"]
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))

        if "raw_window_means" in payload and payload["raw_window_means"].shape[1] >= 4:
            raw = payload["raw_window_means"]
            _scatter_by_outcome(
                axes[0],
                raw[:, 0],
                raw[:, 1],
                outcome,
                title=f"GNN {strategy} | Raw XY",
                x_label="window_mean(x-avg)",
                y_label="window_mean(y-avg)",
            )
            _scatter_by_outcome(
                axes[1],
                raw[:, 2],
                raw[:, 3],
                outcome,
                title=f"GNN {strategy} | Raw Pupil",
                x_label="window_mean(pupil-left)",
                y_label="window_mean(pupil-right)",
            )
        else:
            axes[0].set_axis_off()
            axes[0].set_title(f"GNN {strategy} | Raw XY unavailable")
            axes[1].set_axis_off()
            axes[1].set_title(f"GNN {strategy} | Raw Pupil unavailable")

        if "graph_embeddings" in payload:
            emb_2d = _project_embeddings_to_2d(
                embeddings=np.asarray(payload["graph_embeddings"], dtype=float),
                method=embedding_method,
            )
            _scatter_by_outcome(
                axes[2],
                emb_2d[:, 0],
                emb_2d[:, 1],
                outcome,
                title=f"GNN {strategy} | Embedding ({embedding_method.lower()})",
                x_label="component_1",
                y_label="component_2",
            )
        else:
            axes[2].set_axis_off()
            axes[2].set_title(f"GNN {strategy} | Embeddings unavailable")

        figure.tight_layout()
        output_path = output_dir / f"gnn_tp_fp_tn_fn_scatter_{strategy}.png"
        saved_paths.append(_save_figure(figure, output_path))

    return saved_paths


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
    embedding_method: str = "pca",
    summary_file: str = "summary.csv",
    figures_dir_name: str = "plots",
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

    saved_paths.extend(
        _plot_gnn_tp_fp_tn_fn_scatter(
            run_dir=run_dir_path,
            strategies=strategies,
            decision_threshold=float(decision_threshold),
            embedding_method=embedding_method,
            output_dir=figures_dir,
        )
    )

    return saved_paths
