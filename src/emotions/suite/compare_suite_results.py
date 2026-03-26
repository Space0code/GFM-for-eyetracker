"""Suite-level aggregation and comparison plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CLASSIFICATION_TASK_TYPES = {"binary", "multiclass"}


def _discover_strategy_summaries(run_dir: Path) -> List[Path]:
    if not run_dir.exists() or not run_dir.is_dir():
        return []
    return sorted(
        [path / "summary.csv" for path in run_dir.iterdir() if path.is_dir() and (path / "summary.csv").exists()]
    )


def _read_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if "metric_type" in df.columns:
        df = df[df["metric_type"] == "aggregated"].copy()
    return df


def _build_comparable_classification_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["f1_comparable"] = out.get("f1")
    if "f1_comparable" in out.columns:
        out["f1_comparable"] = out["f1_comparable"].where(out["f1_comparable"].notna(), out.get("macro_f1"))
        out["f1_comparable"] = out["f1_comparable"].where(out["f1_comparable"].notna(), out.get("weighted_f1"))

    out["auc_comparable"] = out.get("auc")
    if "auc_comparable" in out.columns:
        out["auc_comparable"] = out["auc_comparable"].where(out["auc_comparable"].notna(), out.get("macro_auc_ovr"))
        out["auc_comparable"] = out["auc_comparable"].where(out["auc_comparable"].notna(), out.get("weighted_auc_ovr"))

    if "balanced_accuracy" not in out.columns:
        out["balanced_accuracy"] = np.nan
    if "accuracy" not in out.columns:
        out["accuracy"] = np.nan
    return out


def _save_heatmap(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if df.empty or value_col not in df.columns:
        return

    numeric_df = df[["suite_experiment_id", "model", value_col]].copy()
    numeric_df[value_col] = pd.to_numeric(numeric_df[value_col], errors="coerce")
    numeric_df = numeric_df.dropna(subset=[value_col])
    if numeric_df.empty:
        return

    pivot = numeric_df.pivot_table(
        index="suite_experiment_id",
        columns="model",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.apply(pd.to_numeric, errors="coerce")
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.3), max(4, pivot.shape[0] * 0.5)))
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".3f",
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("model")
    ax.set_ylabel("experiment")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_group_ranking_barplot(
    df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
    higher_is_better: bool,
) -> None:
    if df.empty or metric_col not in df.columns:
        return

    agg = (
        df.groupby(["experiment_group", "model"], as_index=False)[metric_col]
        .mean()
        .dropna(subset=[metric_col])
    )
    if agg.empty:
        return

    agg = agg.sort_values(["experiment_group", metric_col], ascending=[True, not higher_is_better])

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(agg))))
    sns.barplot(data=agg, x=metric_col, y="model", hue="experiment_group", ax=ax, orient="h")
    ax.set_title(title)
    ax.set_xlabel(metric_col)
    ax.set_ylabel("model")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_entropy_scatter(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return

    metric_col = "balanced_accuracy"
    if metric_col not in df.columns:
        return

    filtered = df[["suite_experiment_id", "label_entropy", metric_col]].copy()
    filtered = filtered.dropna(subset=["label_entropy", metric_col])
    if filtered.empty:
        return

    best = filtered.groupby("suite_experiment_id", as_index=False).agg(
        label_entropy=("label_entropy", "first"),
        best_metric=(metric_col, "max"),
    )
    if best.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=best, x="label_entropy", y="best_metric", ax=ax)
    ax.set_title("Label Entropy vs Best Balanced Accuracy")
    ax.set_xlabel("label_entropy")
    ax.set_ylabel("best_balanced_accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_suite_comparison_artifacts(
    registry_csv_path: str,
    output_root_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build suite-level master comparison CSVs and plots from experiment registry."""
    registry_path = Path(registry_csv_path)
    output_root = Path(output_root_dir)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry CSV not found: {registry_csv_path}")

    registry_df = pd.read_csv(registry_path)
    if registry_df.empty:
        cls_out = output_root / "classification_master_comparison.csv"
        reg_out = output_root / "regression_master_comparison.csv"
        pd.DataFrame().to_csv(cls_out, index=False)
        pd.DataFrame().to_csv(reg_out, index=False)
        return pd.DataFrame(), pd.DataFrame()

    classification_rows: List[pd.DataFrame] = []
    regression_rows: List[pd.DataFrame] = []

    for _, row in registry_df.iterrows():
        if str(row.get("status", "")).lower() != "success":
            continue

        run_dir = Path(str(row.get("trainer_run_dir", "")))
        if not run_dir.exists():
            continue

        summaries = _discover_strategy_summaries(run_dir=run_dir)
        for summary_path in summaries:
            summary_df = _read_summary(summary_path)
            if summary_df.empty:
                continue

            summary_df = summary_df.copy()
            summary_df["suite_experiment_id"] = row.get("suite_experiment_id")
            summary_df["experiment_id"] = row.get("experiment_id")
            summary_df["scope"] = row.get("scope")
            summary_df["task_type"] = row.get("task_type")
            summary_df["experiment_group"] = row.get("experiment_group")
            summary_df["strategy"] = summary_path.parent.name
            summary_df["trainer_run_dir"] = str(run_dir)
            summary_df["label_entropy"] = row.get("label_entropy")

            if str(row.get("task_type")) in CLASSIFICATION_TASK_TYPES:
                classification_rows.append(summary_df)
            elif str(row.get("task_type")) == "regression":
                regression_rows.append(summary_df)

    classification_df = pd.concat(classification_rows, ignore_index=True) if classification_rows else pd.DataFrame()
    regression_df = pd.concat(regression_rows, ignore_index=True) if regression_rows else pd.DataFrame()

    if not classification_df.empty:
        classification_df = _build_comparable_classification_columns(classification_df)

    cls_out = output_root / "classification_master_comparison.csv"
    reg_out = output_root / "regression_master_comparison.csv"
    classification_df.to_csv(cls_out, index=False)
    regression_df.to_csv(reg_out, index=False)

    if not classification_df.empty:
        _save_heatmap(
            classification_df,
            value_col="accuracy",
            output_path=plots_dir / "classification_heatmap_accuracy.png",
            title="Classification Accuracy (experiment x model)",
            vmin=0.0,
            vmax=1.0,
        )
        _save_heatmap(
            classification_df,
            value_col="balanced_accuracy",
            output_path=plots_dir / "classification_heatmap_balanced_accuracy.png",
            title="Classification Balanced Accuracy (experiment x model)",
            vmin=0.0,
            vmax=1.0,
        )
        _save_heatmap(
            classification_df,
            value_col="f1_comparable",
            output_path=plots_dir / "classification_heatmap_f1.png",
            title="Classification F1 (experiment x model)",
            vmin=0.0,
            vmax=1.0,
        )
        _save_heatmap(
            classification_df,
            value_col="auc_comparable",
            output_path=plots_dir / "classification_heatmap_auc.png",
            title="Classification AUC (experiment x model)",
            vmin=0.0,
            vmax=1.0,
        )

        ranking_metric = "balanced_accuracy"
        if ranking_metric not in classification_df.columns:
            ranking_metric = "accuracy"
        _save_group_ranking_barplot(
            classification_df,
            metric_col=ranking_metric,
            output_path=plots_dir / "classification_group_model_ranking.png",
            title="Classification Model Ranking by Experiment Group",
            higher_is_better=True,
        )

        _save_entropy_scatter(
            classification_df,
            output_path=plots_dir / "classification_entropy_vs_best_metric.png",
        )

    if not regression_df.empty:
        _save_heatmap(
            regression_df,
            value_col="mae",
            output_path=plots_dir / "regression_heatmap_mae.png",
            title="Regression MAE (experiment x model)",
            cmap="Blues",
        )
        _save_heatmap(
            regression_df,
            value_col="ccc",
            output_path=plots_dir / "regression_heatmap_ccc.png",
            title="Regression CCC (experiment x model)",
        )
        _save_heatmap(
            regression_df,
            value_col="spearman",
            output_path=plots_dir / "regression_heatmap_spearman.png",
            title="Regression Spearman (experiment x model)",
        )
        _save_group_ranking_barplot(
            regression_df,
            metric_col="ccc",
            output_path=plots_dir / "regression_group_model_ranking.png",
            title="Regression Model Ranking by Experiment Group (CCC)",
            higher_is_better=True,
        )

    return classification_df, regression_df
