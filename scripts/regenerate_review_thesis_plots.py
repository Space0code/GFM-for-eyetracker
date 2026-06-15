"""Regenerate thesis-style review plots for selected quick-comparison results.

This is a focused post-processing script for the manual review pass requested
on the two June 2026 quick-comparison result folders. It rewrites plot images
from existing CSV artifacts only; it does not rerun model training.

Run from the repository root with the `gfm` environment active:

    python scripts/regenerate_review_thesis_plots.py
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RESULT_ROOTS = [
    Path("results/quick_v1_v2_comparison/RETAIN_2026-06-12_16-29-08"),
    Path("results/quick_v1_v2_comparison/RETAIN_2026-06-03_09-24-45"),
]

SIGNAL_SET_ORDER = ["gaze_only", "pupil_only", "gaze_pupil", "all_signals"]
SIGNAL_SET_NAMES = {
    "gaze_only": "samo pogled",
    "pupil_only": "samo zenici",
    "gaze_pupil": "pogled + zenici",
    "all_signals": "vsi signali",
}
EXPERIMENT_NAMES = {
    "multiclass_table6_valence_3class": "valenca",
    "multiclass_table6_arousal_3class": "vzburjenost",
}
CV_NAMES = {
    "subject_kfold": "prečno preverjanje po osebah",
    "subject_loo": "izpusti eno osebo",
    "recording_loo": "izpusti en posnetek",
    "recording_kfold": "prečno preverjanje po posnetkih",
}
METRIC_NAMES = {
    "accuracy": "točnost",
    "balanced_accuracy": "uravnotežena točnost",
    "macro_f1": "makro F1",
    "weighted_f1": "uteženi F1",
    "macro_auc_ovr": "AUC",
    "weighted_auc_ovr": "AUC",
    "auc": "AUC",
    "auc_comparable": "AUC",
    "f1_comparable": "makro F1",
    "train_loss": "učna izguba",
    "val_loss": "validacijska izguba",
    "test_loss": "testna izguba",
    "val_balanced_accuracy": "uravnotežena točnost",
    "val_macro_f1": "makro F1",
}

MODEL_ORDER = [
    "Random",
    "Majority",
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
    "BasicGCN_mean",
    "BasicGCN_attention",
    "GNN_v2",
    "GNN",
    "HeteroGCNMean",
    "HeteroGCNMLP",
    "HeteroGCNMLPWeights",
]
MODEL_NAMES = {
    "Random": "Naključni",
    "random": "Naključni",
    "Majority": "Večinski",
    "majority": "Večinski",
    "majority_classifier": "Večinski",
    "SVM": "SVM",
    "LightGBM": "LightGBM",
    "MLP": "MLP",
    "GazeMAE_MLP": "GazeMAE",
    "MOMENT_gaze": "MOMENT",
    "MOMENT_pupil": "MOMENT",
    "MOMENT_gaze_pupil": "MOMENT",
    "MOMENT_all_signals": "MOMENT",
    "MOMENT_GazeMAE_gaze_pupil": "MOMENTxGazeMAE",
    "MOMENT_GazeMAE_all_signals": "MOMENTxGazeMAE",
    "BasicGCN": "GCN",
    "BasicGCN_mean": "GCN-mean",
    "BasicGCN_attention": "GCN-att",
    "GNN_v2": "GNN v2",
    "GNN": "GNN",
    "HeteroGCNMean": "HeteroGCN-mean",
    "HeteroGCNMLP": "HeteroGCN-MLP",
    "HeteroGCNMLPWeights": "HeteroGCN-MLP-w",
}
OLD_DISPLAY_REPLACEMENTS = {
    "GazeMAE+MLP": "GazeMAE",
    "MOMENT+MLP": "MOMENT",
    "MOMENT+GazeMAE+MLP": "MOMENTxGazeMAE",
}
FROZEN_FAMILY = {
    "GazeMAE_MLP",
    "MOMENT_gaze",
    "MOMENT_pupil",
    "MOMENT_gaze_pupil",
    "MOMENT_all_signals",
    "MOMENT_GazeMAE_gaze_pupil",
    "MOMENT_GazeMAE_all_signals",
}
COLLAPSED_FROZEN_NAME = "GazeMAE/MOMENT"
MODEL_COLORS = {
    "Naključni": "#B8B8B8",
    "Večinski": "#D8B365",
    "SVM": "#B58D82",
    "LightGBM": "#8FD08B",
    "MLP": "#E8A06A",
    "GazeMAE": "#B58CC8",
    "MOMENT": "#B58CC8",
    "MOMENTxGazeMAE": "#B58CC8",
    COLLAPSED_FROZEN_NAME: "#B58CC8",
    "GCN": "#C6DBEF",
    "GCN-mean": "#C6DBEF",
    "GCN-att": "#9ECAE1",
    "GNN": "#9ECAE1",
    "GNN v2": "#6BAED6",
    "HeteroGCN-mean": "#9ECAE1",
    "HeteroGCN-MLP": "#6BAED6",
    "HeteroGCN-MLP-w": "#4292C6",
}
CLASS_NAMES = {
    "Low valence": "nizka valenca",
    "High valence": "visoka valenca",
    "Unpleasant": "neprijetno",
    "Neutral valence": "nevtralna valenca",
    "Pleasant": "prijetno",
    "Calm": "nizka vzburjenost",
    "Medium aroused": "srednja vzburjenost",
    "Excited/Activated": "visoka vzburjenost",
}


def model_order_key(model: object) -> tuple[int, str]:
    name = str(model)
    try:
        return MODEL_ORDER.index(name), name
    except ValueError:
        return len(MODEL_ORDER), name


def display_model(model: object, *, collapse_frozen: bool = False) -> str:
    name = str(model)
    if name in OLD_DISPLAY_REPLACEMENTS:
        return COLLAPSED_FROZEN_NAME if collapse_frozen else OLD_DISPLAY_REPLACEMENTS[name]
    if collapse_frozen and name in FROZEN_FAMILY:
        return COLLAPSED_FROZEN_NAME
    return MODEL_NAMES.get(name, name)


def heatmap_model_label(label: str) -> str:
    """Return a horizontal, compact multi-line label for heatmap columns."""
    replacements = {
        "HeteroGCN-mean": "HeteroGCN-\nmean",
        "HeteroGCN-MLP": "HeteroGCN-\nMLP",
        "HeteroGCN-MLP-w": "HeteroGCN-\nMLP-w",
        "MOMENTxGazeMAE": "MOMENTx\nGazeMAE",
        COLLAPSED_FROZEN_NAME: "GazeMAE/\nMOMENT",
    }
    return replacements.get(label, label)


def metric_label(metric: str) -> str:
    return METRIC_NAMES.get(metric, metric.replace("_", " "))


def experiment_label(experiment_id: object) -> str:
    return EXPERIMENT_NAMES.get(str(experiment_id), str(experiment_id))


def cv_label(cv_strategy: object) -> str:
    return CV_NAMES.get(str(cv_strategy), str(cv_strategy))


def panel_title(group: dict[str, object], *, include_signal: bool = True) -> str:
    bits: list[str] = []
    if include_signal and group.get("signal_set") is not None:
        bits.append(SIGNAL_SET_NAMES.get(str(group["signal_set"]), str(group["signal_set"])))
    if group.get("experiment_id") is not None:
        bits.append(experiment_label(group["experiment_id"]))
    if group.get("cv_strategy") is not None:
        bits.append(cv_label(group["cv_strategy"]))
    elif group.get("strategy") is not None:
        bits.append(cv_label(group["strategy"]))
    return " - ".join(bits)


def compact_panel_title(group: dict[str, object]) -> str:
    bits: list[str] = []
    if group.get("signal_set") is not None:
        bits.append(SIGNAL_SET_NAMES.get(str(group["signal_set"]), str(group["signal_set"])))
    if group.get("experiment_id") is not None:
        bits.append(experiment_label(group["experiment_id"]))
    strategy = group.get("cv_strategy", group.get("strategy"))
    if strategy is not None:
        bits.append(cv_label(strategy))
    return "\n".join(bits)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def delete_removed_plots(root: Path) -> None:
    for pattern in ["classification_entropy_vs_best_metric.png", "test_loss_by_model.png"]:
        for path in root.rglob(pattern):
            path.unlink()
            print(f"removed {path}")


def update_display_tables(root: Path) -> None:
    candidates = [
        *root.glob("quick_comparison_summary*.csv"),
        root / "tables" / "fold_metrics.csv",
        root / "tables" / "metric_summary_with_std.csv",
        root / "tables" / "model_benchmark_summary.csv",
        root / "tables" / "main_model_complexity_report.csv",
        root / "tables" / "thesis_signal_set_model_metrics.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        changed = False
        if "model_display" in df.columns and "model" in df.columns:
            df["model_display"] = df["model"].map(display_model)
            changed = True
        elif path.name == "thesis_signal_set_model_metrics.csv" and "model" in df.columns:
            df["model"] = df["model"].map(lambda value: display_model(value, collapse_frozen=True))
            changed = True
        elif path.name == "main_model_complexity_report.csv" and "model" in df.columns:
            df["model"] = df["model"].map(display_model)
            changed = True
        if changed:
            df.to_csv(path, index=False)
            print(f"updated display names in {path}")


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str) == "success"]
    if "metric_type" in out.columns:
        out = out[out["metric_type"].astype(str) == "aggregated"]
    return out


def ordered_models(models: Iterable[object]) -> list[str]:
    return sorted({str(model) for model in models if pd.notna(model)}, key=model_order_key)


def ordered_groups(df: pd.DataFrame, group_cols: list[str]) -> list[tuple[object, ...]]:
    if not group_cols:
        return [tuple()]
    groups = list(df[group_cols].drop_duplicates().itertuples(index=False, name=None))

    def key(values: tuple[object, ...]) -> tuple[object, ...]:
        group = dict(zip(group_cols, values))
        signal_rank = SIGNAL_SET_ORDER.index(str(group["signal_set"])) if str(group.get("signal_set")) in SIGNAL_SET_ORDER else 999
        experiment_rank = ["multiclass_table6_valence_3class", "multiclass_table6_arousal_3class"].index(str(group["experiment_id"])) if str(group.get("experiment_id")) in {"multiclass_table6_valence_3class", "multiclass_table6_arousal_3class"} else 999
        cv_rank = ["subject_kfold", "subject_loo", "recording_loo", "recording_kfold"].index(str(group.get("cv_strategy", group.get("strategy", "")))) if str(group.get("cv_strategy", group.get("strategy", ""))) in {"subject_kfold", "subject_loo", "recording_loo", "recording_kfold"} else 999
        return (signal_rank, experiment_rank, cv_rank, tuple(str(value) for value in values))

    return sorted(groups, key=key)


def ranking_metric_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1_comparable",
        "weighted_f1",
        "auc",
        "macro_auc_ovr",
        "auc_comparable",
    ]
    selected: list[str] = []
    seen_labels: set[str] = set()
    for metric in candidates:
        if metric not in df.columns:
            continue
        label = metric_label(metric)
        if label == "AUC" and "AUC" in seen_labels:
            continue
        selected.append(metric)
        seen_labels.add(label)
    return selected


def plot_grouped_model_ranking(
    df: pd.DataFrame,
    output_path: Path,
    group_cols: list[str],
    title: str,
    *,
    inside_label_color: str = "#111111",
) -> None:
    if df.empty or "model" not in df.columns:
        return
    plot_df = successful_rows(df)
    metrics = ranking_metric_columns(plot_df)
    if not metrics:
        return
    for metric in metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=metrics, how="all")
    if plot_df.empty:
        return
    groups = ordered_groups(plot_df, group_cols)
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(14.5, max(4.0, 4.35 * len(groups))),
        squeeze=False,
    )
    for idx, values in enumerate(groups):
        group = dict(zip(group_cols, values))
        group_df = plot_df.copy()
        for col, value in group.items():
            group_df = group_df[group_df[col].astype(str) == str(value)]
        models = ordered_models(group_df["model"])
        ax = axes[idx, 0]
        x = np.arange(len(metrics))
        group_width = 0.84
        bar_width = group_width / max(1, len(models))
        for model_idx, model in enumerate(models):
            model_df = group_df[group_df["model"].astype(str) == str(model)]
            values_for_model = [
                pd.to_numeric(model_df[metric], errors="coerce").dropna().iloc[0]
                if not model_df.empty and not pd.to_numeric(model_df[metric], errors="coerce").dropna().empty
                else np.nan
                for metric in metrics
            ]
            label = display_model(model)
            offset = -group_width / 2 + bar_width / 2 + model_idx * bar_width
            bars = ax.bar(
                x + offset,
                values_for_model,
                width=bar_width * 0.94,
                color=MODEL_COLORS.get(label, "#999999"),
                edgecolor="white",
                linewidth=0.4,
            )
            for bar, value in zip(bars, values_for_model):
                if not pd.notna(value):
                    continue
                x_pos = bar.get_x() + bar.get_width() / 2
                if value >= 0.18:
                    ax.text(
                        x_pos,
                        value / 2,
                        label,
                        ha="center",
                        va="center",
                        rotation=90,
                        fontsize=6.6,
                        color=inside_label_color,
                        fontweight="normal",
                        clip_on=True,
                    )
                else:
                    ax.text(
                        x_pos,
                        value + 0.012,
                        label,
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6.0,
                        color="#111111",
                        fontweight="normal",
                        clip_on=True,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([metric_label(metric) for metric in metrics], rotation=0, ha="center")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("")
        ax.set_title(panel_title(group) or title, fontsize=11, pad=8)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.4)
    save_figure(fig, output_path)


def plot_metrics_barplots(summary_paths: Iterable[Path]) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auc_ovr"]
    seen_run_dirs: set[Path] = set()
    for summary_path in summary_paths:
        run_dir = summary_path.parents[1]
        if run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir)
        figures_dir = run_dir / "plots"
        rows: list[pd.DataFrame] = []
        for strategy_dir in sorted(summary_path.parents[1].iterdir()):
            candidate = strategy_dir / "summary.csv"
            if candidate.exists():
                frame = pd.read_csv(candidate)
                frame["strategy"] = strategy_dir.name
                rows.append(frame)
        if not rows:
            continue
        df = successful_rows(pd.concat(rows, ignore_index=True))
        available = [metric for metric in metrics if metric in df.columns]
        if not available:
            continue
        for metric in available:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
        strategies = list(dict.fromkeys(df["strategy"].astype(str)))
        fig, axes = plt.subplots(len(strategies), 1, figsize=(13.0, max(3.8, 3.6 * len(strategies))), squeeze=False)
        for row_idx, strategy in enumerate(strategies):
            ax = axes[row_idx, 0]
            strategy_df = df[df["strategy"].astype(str) == strategy].copy()
            models = ordered_models(strategy_df["model"])
            x = np.arange(len(available))
            group_width = 0.86
            bar_width = group_width / max(1, len(models))
            for model_idx, model in enumerate(models):
                model_df = strategy_df[strategy_df["model"].astype(str) == str(model)]
                values = [pd.to_numeric(model_df[metric], errors="coerce").dropna().iloc[0] if not model_df[metric].dropna().empty else np.nan for metric in available]
                offset = -group_width / 2 + bar_width / 2 + model_idx * bar_width
                label = display_model(model)
                bars = ax.bar(
                    x + offset,
                    values,
                    width=bar_width * 0.94,
                    color=MODEL_COLORS.get(label, "#777777"),
                )
                for bar, value in zip(bars, values):
                    if not pd.notna(value):
                        continue
                    if value > 0.18:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value / 2,
                            label,
                            ha="center",
                            va="center",
                            rotation=90,
                            fontsize=6.5,
                            color="white",
                            fontweight="bold",
                        )
            ax.set_xticks(x)
            ax.set_xticklabels([metric_label(metric) for metric in available], rotation=0)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("vrednost")
            if "experiment_id" in df and not df["experiment_id"].dropna().empty:
                experiment_id = str(df["experiment_id"].dropna().iloc[0])
            elif "valence" in run_dir.name:
                experiment_id = "multiclass_table6_valence_3class"
            elif "arousal" in run_dir.name:
                experiment_id = "multiclass_table6_arousal_3class"
            else:
                experiment_id = ""
            ax.set_title(f"{experiment_label(experiment_id)} - {cv_label(strategy)}", fontsize=11)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Primerjava modelov po metrikah", y=0.995, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_figure(fig, figures_dir / "metrics_barplots.png")


def plot_combined_suite_heatmap(master_csv: Path) -> None:
    df = successful_rows(pd.read_csv(master_csv))
    if df.empty:
        return
    metric_specs = [
        ("accuracy", "točnost"),
        ("balanced_accuracy", "uravnotežena točnost"),
        ("f1_comparable", "makro F1"),
        ("auc_comparable", "AUC"),
    ]
    metric_specs = [(metric, label) for metric, label in metric_specs if metric in df.columns]
    if not metric_specs:
        return
    group_cols = [col for col in ["experiment_id", "strategy"] if col in df.columns]
    groups = ordered_groups(df, group_cols)
    n_cols = 2 if len(groups) > 1 else 1
    n_rows = ceil(len(groups) / n_cols)
    max_model_count = max((len(ordered_models(df[df[group_cols[0]].astype(str) == str(values[0])]["model"])) if group_cols else len(ordered_models(df["model"])) for values in groups), default=1)
    figure_width = max(10.5, 6.8 * n_cols, 1.15 * max_model_count * n_cols + 2.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figure_width, max(3.8, 3.4 * n_rows)), squeeze=False)
    cbar_ax = fig.add_axes([0.93, 0.18, 0.018, 0.66])
    for ax in axes.ravel()[len(groups):]:
        ax.set_axis_off()
    for idx, values in enumerate(groups):
        ax = axes.ravel()[idx]
        group = dict(zip(group_cols, values))
        group_df = df.copy()
        for col, value in group.items():
            group_df = group_df[group_df[col].astype(str) == str(value)]
        models = ordered_models(group_df["model"])
        table = pd.DataFrame(index=[label for _, label in metric_specs])
        for model in models:
            label = display_model(model)
            row = group_df[group_df["model"].astype(str) == str(model)]
            if row.empty:
                continue
            table[label] = [pd.to_numeric(row[metric], errors="coerce").dropna().iloc[0] if not row[metric].dropna().empty else np.nan for metric, _ in metric_specs]
        display_columns = list(table.columns)
        table.columns = [heatmap_model_label(str(column)) for column in display_columns]
        sns.heatmap(
            table,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar=idx == 0,
            cbar_ax=cbar_ax if idx == 0 else None,
            cbar_kws={"label": "vrednost"},
            ax=ax,
        )
        subtitle = panel_title(group)
        ax.set_title(f"Pregled klasifikacijskih metrik - {subtitle}" if len(groups) == 1 else subtitle, fontsize=11, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.tick_params(axis="x", labelsize=9)
    if len(groups) > 1:
        fig.suptitle("Pregled klasifikacijskih metrik", y=0.985, fontsize=13)
        top = 0.90
    else:
        top = 0.86
    fig.subplots_adjust(left=0.10, right=0.89, top=top, bottom=0.14, hspace=0.36, wspace=0.34)
    output_path = master_csv.parent / "plots" / "classification_heatmap_metrics.png"
    save_figure(fig, output_path)
    for old in master_csv.parent.joinpath("plots").glob("classification_heatmap_*.png"):
        if old.name != output_path.name:
            old.unlink()
            print(f"removed {old}")


def plot_thesis_heatmaps(root: Path) -> None:
    table_path = root / "quick_comparison_summary.csv"
    if not table_path.exists():
        return
    df = successful_rows(pd.read_csv(table_path))
    if df.empty or "signal_set" not in df.columns:
        return
    for metric, filename_metric in [("accuracy", "accuracy"), ("macro_f1", "macro_f1")]:
        if metric not in df.columns:
            continue
        plot_df = df.copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        models = ordered_models(plot_df["model"])
        display_order = []
        for model in models:
            label = display_model(model, collapse_frozen=True)
            if label not in display_order:
                display_order.append(label)
        plot_df["model_display"] = plot_df["model"].map(lambda value: display_model(value, collapse_frozen=True))
        heatmap = plot_df.pivot_table(index="signal_set", columns="model_display", values=metric, aggfunc="first")
        heatmap = heatmap.reindex(index=[s for s in SIGNAL_SET_ORDER if s in heatmap.index], columns=display_order)
        heatmap.index = [SIGNAL_SET_NAMES.get(str(index), str(index)) for index in heatmap.index]
        width = max(8.5, 0.9 * len(display_order) + 2.2)
        fig, ax = plt.subplots(figsize=(width, 3.8))
        sns.heatmap(
            heatmap,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": metric_label(metric)},
            ax=ax,
        )
        ax.set_title(f"Primerjava modelov po signalnih naborih - {metric_label(metric)}", fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
        output = root / "plots" / f"thesis_{filename_metric}_heatmap_multiclass_table6_valence_3class_subject_kfold.png"
        save_figure(fig, output)


def plot_label_distributions(root: Path) -> None:
    table_path = root / "tables" / "label_distribution_aggregate.csv"
    if not table_path.exists():
        return
    df = pd.read_csv(table_path)
    if df.empty:
        return
    if "split" in df.columns:
        df = df[df["split"].astype(str) == "test"]
    group_col = "experiment_id"
    experiments = [exp for exp in ["multiclass_table6_valence_3class", "multiclass_table6_arousal_3class"] if exp in set(df[group_col].astype(str))]
    if not experiments:
        experiments = list(dict.fromkeys(df[group_col].astype(str)))
    representative_rows = []
    for experiment in experiments:
        exp_df = df[df[group_col].astype(str) == experiment].copy()
        for optional in ["signal_set", "cv_strategy", "source_model"]:
            if optional in exp_df.columns:
                order = SIGNAL_SET_ORDER if optional == "signal_set" else sorted(exp_df[optional].astype(str).unique())
                chosen = next((value for value in order if value in set(exp_df[optional].astype(str))), exp_df[optional].astype(str).iloc[0])
                exp_df = exp_df[exp_df[optional].astype(str) == str(chosen)]
        representative_rows.append(exp_df)
    plot_df = pd.concat(representative_rows, ignore_index=True)
    plot_df["class_display"] = plot_df["class_name"].map(lambda value: CLASS_NAMES.get(str(value), str(value)))

    for y_col, output_name, ylabel in [
        ("count", "label_distribution_counts.png", "število primerov"),
        ("proportion", "label_distribution_proportions.png", "delež"),
    ]:
        fig, axes = plt.subplots(len(experiments), 1, figsize=(8.5, max(3.2, 3.0 * len(experiments))), squeeze=False)
        for idx, experiment in enumerate(experiments):
            ax = axes[idx, 0]
            exp_df = plot_df[plot_df[group_col].astype(str) == experiment].sort_values("class_index")
            bars = ax.bar(exp_df["class_display"], exp_df[y_col], color="#4C78A8", width=0.72)
            if y_col == "proportion":
                ax.set_ylim(0, 1)
                labels = [f"{value:.2f}" for value in exp_df[y_col]]
            else:
                labels = [f"{int(value)}" for value in exp_df[y_col]]
            ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
            ax.set_title(f"Porazdelitev razredov: {experiment_label(experiment)}", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", visible=False)
            ax.xaxis.grid(False)
        fig.tight_layout()
        save_figure(fig, root / "plots" / output_name)


def aggregate_curve(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    curve = df[["epoch", "fold_id", metric]].copy()
    curve["epoch"] = pd.to_numeric(curve["epoch"], errors="coerce")
    curve[metric] = pd.to_numeric(curve[metric], errors="coerce")
    curve = curve.dropna(subset=["epoch", metric])
    if curve.empty:
        return pd.DataFrame()
    return curve.groupby("epoch", as_index=False)[metric].agg(["mean", "std"]).reset_index()


def plot_training_grid(root: Path, metrics: list[str], output_name: str, title: str) -> None:
    table_path = root / "tables" / "training_history.csv"
    if not table_path.exists():
        return
    df = pd.read_csv(table_path)
    metrics = [metric for metric in metrics if metric in df.columns]
    if not metrics or "model" not in df.columns:
        return
    group_cols = [col for col in ["signal_set", "experiment_id", "cv_strategy"] if col in df.columns]
    groups = ordered_groups(df, group_cols)
    row_models: list[list[str]] = []
    for values in groups:
        group = dict(zip(group_cols, values))
        group_df = df.copy()
        for col, value in group.items():
            group_df = group_df[group_df[col].astype(str) == str(value)]
        row_models.append(ordered_models(group_df["model"]))
    max_cols = max((len(models) for models in row_models), default=1)
    fig, axes = plt.subplots(
        len(groups),
        max_cols,
        figsize=(max(12, 3.15 * max_cols), max(4.2, 2.85 * len(groups))),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    metric_styles = {
        "train_loss": ("#0072B2", "-"),
        "val_loss": ("#E69F00", "-"),
        "test_loss": ("#666666", "--"),
        "val_balanced_accuracy": ("#009E73", "-"),
        "val_macro_f1": ("#CC79A7", "-"),
    }
    handles: dict[str, object] = {}
    for row_idx, values in enumerate(groups):
        group = dict(zip(group_cols, values))
        group_df = df.copy()
        for col, value in group.items():
            group_df = group_df[group_df[col].astype(str) == str(value)]
        models = row_models[row_idx]
        row_label = compact_panel_title(group)
        fig.text(
            0.018,
            0.83 - row_idx * (0.76 / max(1, len(groups) - 1)) if len(groups) > 1 else 0.50,
            row_label,
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="semibold",
            linespacing=1.2,
        )
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            model_df = group_df[group_df["model"].astype(str) == str(model)]
            if model_df.empty:
                ax.set_axis_off()
                continue
            for metric in metrics:
                curve = aggregate_curve(model_df, metric)
                if curve.empty:
                    continue
                color, linestyle = metric_styles.get(metric, ("#333333", "-"))
                line = ax.plot(curve["epoch"], curve["mean"], color=color, linestyle=linestyle, linewidth=1.4, label=metric_label(metric))[0]
                handles[metric] = line
                std = curve["std"].fillna(0.0)
                ax.fill_between(curve["epoch"].to_numpy(), (curve["mean"] - std).to_numpy(), (curve["mean"] + std).to_numpy(), color=color, alpha=0.12, linewidth=0)
            if row_idx == 0:
                ax.set_title(display_model(model), fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel("izguba" if any("loss" in metric for metric in metrics) else "vrednost", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(alpha=0.20)
            if any("loss" in metric for metric in metrics):
                values_for_ylim = []
                for metric in metrics:
                    values_for_ylim.extend(pd.to_numeric(model_df[metric], errors="coerce").dropna().tolist())
                upper = np.nanpercentile(values_for_ylim, 98) * 1.15 if values_for_ylim else 1.0
                ax.set_ylim(0, min(max(upper, 0.8), 2.0))
            else:
                ax.set_ylim(0, 1)
        for col_idx in range(len(models), max_cols):
            axes[row_idx, col_idx].set_axis_off()
    legend_handles = [handles[metric] for metric in metrics if metric in handles]
    legend_labels = [metric_label(metric) for metric in metrics if metric in handles]
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, y=0.992, fontsize=13)
    fig.text(0.5, 0.035, "epoha", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.995, top=0.93, bottom=0.10, hspace=0.55, wspace=0.25)
    save_figure(fig, root / "plots" / output_name)


def process_root(root: Path) -> None:
    delete_removed_plots(root)
    update_display_tables(root)

    summary_candidates = [root / "quick_comparison_summary_baselines_added_post_hoc.csv", root / "quick_comparison_summary.csv"]
    summary_path = next((path for path in summary_candidates if path.exists()), None)
    if summary_path is not None:
        summary = pd.read_csv(summary_path)
        group_cols = [col for col in ["signal_set", "experiment_id", "cv_strategy"] if col in summary.columns]
        plot_grouped_model_ranking(summary, root / "plots" / "classification_group_model_ranking.png", group_cols, "Primerjava modelov po metrikah")
        if (root / "plots" / "classification_group_model_ranking_baselines_added_post_hoc.png").exists():
            plot_grouped_model_ranking(
                summary,
                root / "plots" / "classification_group_model_ranking_baselines_added_post_hoc.png",
                group_cols,
                "Primerjava modelov po metrikah",
                inside_label_color="black",
            )

    plot_thesis_heatmaps(root)
    plot_label_distributions(root)
    plot_training_grid(root, ["train_loss", "val_loss"], "training_progress_loss.png", "Potek učne in validacijske izgube")
    plot_training_grid(root, ["train_loss", "val_loss", "test_loss"], "training_progress_loss_combined.png", "Potek izgube po delitvah podatkov")
    plot_training_grid(root, ["val_balanced_accuracy", "val_macro_f1"], "training_progress_validation_metrics.png", "Potek validacijskih metrik")

    for master_csv in root.glob("model_runs/**/classification_master_comparison.csv"):
        plot_combined_suite_heatmap(master_csv)
        df = pd.read_csv(master_csv)
        group_cols = [col for col in ["experiment_id", "strategy"] if col in df.columns]
        plot_grouped_model_ranking(df, master_csv.parent / "plots" / "classification_group_model_ranking.png", group_cols, "Primerjava modelov po metrikah")

    plot_metrics_barplots(root.glob("model_runs/**/summary.csv"))


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    for root in RESULT_ROOTS:
        if root.exists():
            print(f"processing {root}")
            process_root(root)


if __name__ == "__main__":
    main()
