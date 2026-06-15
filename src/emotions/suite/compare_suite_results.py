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
PREFERRED_MODEL_ORDER = [
    "Random",
    "random",
    "Majority",
    "majority",
    "MajorityClassifier",
    "majority_classifier",
    "SVM",
    "LightGBM",
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
    "GeteroGCNMLP",
    "HeteroGCNMLPWeights",
    "GeteroGCNMLPWeights",
]
MODEL_DISPLAY_NAMES = {
    "Random": "Naključni",
    "random": "Naključni",
    "Majority": "Večinski",
    "majority": "Večinski",
    "MajorityClassifier": "Večinski",
    "majority_classifier": "Večinski",
    "GazeMAE_MLP": "GazeMAE+MLP",
    "MOMENT_gaze": "MOMENT+MLP",
    "MOMENT_pupil": "MOMENT+MLP",
    "MOMENT_gaze_pupil": "MOMENT+MLP",
    "MOMENT_all_signals": "MOMENT+MLP",
    "MOMENT_GazeMAE_gaze_pupil": "MOMENT+GazeMAE+MLP",
    "MOMENT_GazeMAE_all_signals": "MOMENT+GazeMAE+MLP",
    "BasicGCN": "GCN",
    "HeteroGCNMean": "HeteroGCN-mean",
    "HeteroGCNMLP": "HeteroGCN-MLP",
    "GeteroGCNMLP": "HeteroGCN-MLP",
    "HeteroGCNMLPWeights": "HeteroGCN-MLP-w",
    "GeteroGCNMLPWeights": "HeteroGCN-MLP-w",
}
METRIC_DISPLAY_NAMES = {
    "accuracy": "točnost",
    "balanced_accuracy": "uravnotežena točnost",
    "macro_f1": "makro F1",
    "weighted_f1": "utežen F1",
    "f1_comparable": "F1",
    "auc_comparable": "AUC",
    "auc": "AUC",
}


def _model_display_name(model_name: str) -> str:
    """Return the thesis-facing model display name."""
    name = str(model_name)
    if name in set(MODEL_DISPLAY_NAMES.values()):
        return name
    return MODEL_DISPLAY_NAMES.get(name, name)


def _model_order_index(model_name: str) -> int:
    """Return fixed simple-to-complex model order index."""
    preferred_idx = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    name = str(model_name)
    if name in preferred_idx:
        return preferred_idx[name]
    display_idx = {
        _model_display_name(preferred_name): idx
        for idx, preferred_name in enumerate(PREFERRED_MODEL_ORDER)
    }
    return display_idx.get(name, len(preferred_idx))


def _ordered_model_labels(model_names: List[str]) -> List[str]:
    """Return unique model labels in fixed simple-to-complex order."""
    unique = list(dict.fromkeys(str(model_name) for model_name in model_names))
    ordered = sorted(unique, key=lambda name: (_model_order_index(name), name.lower()))
    return list(dict.fromkeys(_model_display_name(name) for name in ordered))


def _metric_display_name(metric_name: str) -> str:
    """Return the thesis-facing metric display name."""
    return METRIC_DISPLAY_NAMES.get(str(metric_name), str(metric_name))


def _insert_model_display_column(df: pd.DataFrame) -> pd.DataFrame:
    """Insert or refresh thesis-facing model labels in a comparison table."""
    if df.empty or "model" not in df.columns:
        return df
    result = df.drop(columns=["model_display"], errors="ignore").copy()
    insert_at = result.columns.get_loc("model") + 1
    result.insert(insert_at, "model_display", result["model"].map(_model_display_name))
    return result


def _sort_by_model_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort comparison rows by experiment metadata and fixed model order."""
    if df.empty or "model" not in df.columns:
        return df
    result = df.copy()
    result["_model_order"] = result["model"].map(_model_order_index)
    sort_columns = [
        column
        for column in [
            "suite_experiment_id",
            "experiment_id",
            "strategy",
            "_model_order",
            "model",
        ]
        if column in result.columns
    ]
    return result.sort_values(sort_columns).drop(columns=["_model_order"]).reset_index(drop=True)


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
    numeric_df["model_display"] = numeric_df["model"].map(_model_display_name)
    numeric_df[value_col] = pd.to_numeric(numeric_df[value_col], errors="coerce")
    numeric_df = numeric_df.dropna(subset=[value_col])
    if numeric_df.empty:
        return

    pivot = numeric_df.pivot_table(
        index="suite_experiment_id",
        columns="model_display",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.apply(pd.to_numeric, errors="coerce")
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if pivot.empty:
        return
    model_order = _ordered_model_labels(numeric_df["model"].astype(str).tolist())
    pivot = pivot.reindex(columns=[model for model in model_order if model in pivot.columns])

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
    ax.set_ylabel("eksperiment")
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

    agg["model_display"] = agg["model"].map(_model_display_name)
    agg["_model_order"] = agg["model"].map(_model_order_index)
    agg = agg.sort_values(["experiment_group", "_model_order", "model"], ascending=True)
    model_order = _ordered_model_labels(agg["model"].astype(str).tolist())

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(agg))))
    sns.barplot(
        data=agg,
        x=metric_col,
        y="model_display",
        hue="experiment_group",
        order=model_order,
        ax=ax,
        orient="h",
    )
    ax.set_title(title)
    ax.set_xlabel(_metric_display_name(metric_col))
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
        classification_df = _insert_model_display_column(_sort_by_model_order(classification_df))
    if not regression_df.empty:
        regression_df = _insert_model_display_column(_sort_by_model_order(regression_df))

    cls_out = output_root / "classification_master_comparison.csv"
    reg_out = output_root / "regression_master_comparison.csv"
    classification_df.to_csv(cls_out, index=False)
    regression_df.to_csv(reg_out, index=False)

    if not classification_df.empty:
        _save_heatmap(
            classification_df,
            value_col="accuracy",
            output_path=plots_dir / "classification_heatmap_accuracy.png",
            title="Klasifikacijska točnost (eksperiment x model)",
            vmin=0.0,
            vmax=1.0,
        )
        _save_heatmap(
            classification_df,
            value_col="balanced_accuracy",
            output_path=plots_dir / "classification_heatmap_balanced_accuracy.png",
            title="Uravnotežena klasifikacijska točnost (eksperiment x model)",
            vmin=0.0,
            vmax=1.0,
        )
        _save_heatmap(
            classification_df,
            value_col="f1_comparable",
            output_path=plots_dir / "classification_heatmap_f1.png",
            title="Klasifikacijski F1 (eksperiment x model)",
            vmin=0.0,
            vmax=1.0,
        )
        _save_heatmap(
            classification_df,
            value_col="auc_comparable",
            output_path=plots_dir / "classification_heatmap_auc.png",
            title="Klasifikacijski AUC (eksperiment x model)",
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
            title="Primerjava modelov po skupinah eksperimentov",
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
