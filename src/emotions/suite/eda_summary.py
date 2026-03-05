"""Compact EDA text summary generation for experiment snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = ["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"]


@dataclass
class EdaSummaryResult:
    """EDA summary result including text and key scalar stats."""

    text: str
    stats: Dict[str, Any]


def _resolve_threshold_value(threshold_spec: Any, values: Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        raise ValueError("Cannot resolve threshold from empty numeric values.")

    if isinstance(threshold_spec, str):
        mode = threshold_spec.strip().lower()
        if mode == "mean":
            return float(series.mean())
        if mode == "median":
            return float(series.median())
        try:
            return float(mode)
        except ValueError as exc:
            raise ValueError(
                f"Invalid threshold '{threshold_spec}'. Use numeric, 'mean', or 'median'."
            ) from exc

    return float(threshold_spec)


def _distribution_stats(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {key: float("nan") for key in ["min", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "max", "mean", "std"]}

    return {
        "min": float(numeric.min()),
        "q01": float(numeric.quantile(0.01)),
        "q05": float(numeric.quantile(0.05)),
        "q25": float(numeric.quantile(0.25)),
        "q50": float(numeric.quantile(0.50)),
        "q75": float(numeric.quantile(0.75)),
        "q95": float(numeric.quantile(0.95)),
        "q99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
    }


def _format_distribution(name: str, stats: Dict[str, float]) -> str:
    ordered_keys = ["min", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "max", "mean", "std"]
    values = ", ".join(f"{key}={stats[key]:.6g}" for key in ordered_keys)
    return f"- {name}: {values}"


def _compute_label_entropy(labels: pd.Series) -> float:
    counts = labels.value_counts(dropna=False)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    probs = counts.astype(float) / float(total)
    entropy = -np.sum([float(p) * math.log(float(p), 2) for p in probs if p > 0])
    return float(entropy)


def _format_class_distribution_label(value: Any, label_name_mapping: Mapping[int, str] | None) -> str:
    """Format class value with optional human-readable class name."""
    if label_name_mapping:
        try:
            as_float = float(value)
            if as_float.is_integer():
                class_name = label_name_mapping.get(int(as_float))
                if class_name:
                    return f"class={value} ({class_name})"
        except (TypeError, ValueError):
            pass
    return f"class={value}"


def _classification_distribution_text(
    labels: pd.Series,
    label_name_mapping: Mapping[int, str] | None = None,
) -> Tuple[List[str], float, float]:
    counts = labels.value_counts(dropna=False).sort_index()
    total = float(counts.sum()) if len(counts) > 0 else 0.0
    lines = []
    minority_ratio = float("nan")

    if total > 0:
        minority_ratio = float(counts.min() / total)
        for value, count in counts.items():
            proportion = float(count) / total
            class_label = _format_class_distribution_label(
                value=value,
                label_name_mapping=label_name_mapping,
            )
            lines.append(f"- {class_label}: count={int(count)}, proportion={proportion:.4f}")

    return lines, _compute_label_entropy(labels), minority_ratio


def _regression_distribution_text(values: pd.Series) -> List[str]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ["- no valid numeric labels"]

    bins = pd.cut(numeric, bins=5, include_lowest=True)
    counts = bins.value_counts(sort=False)
    total = float(counts.sum())
    lines = []
    for bin_range, count in counts.items():
        lines.append(
            f"- bin={bin_range}: count={int(count)}, proportion={float(count) / total:.4f}"
        )
    return lines


def _derive_final_labels(
    snapshot_df: pd.DataFrame,
    task_type: str,
    experiment_cfg: Dict[str, Any],
) -> Tuple[pd.Series | None, Dict[str, Any]]:
    details: Dict[str, Any] = {}

    if task_type == "binary":
        target_column = experiment_cfg["target_column"]
        threshold_spec = experiment_cfg.get("threshold", 0.0)
        target = pd.to_numeric(snapshot_df[target_column], errors="coerce")
        threshold = _resolve_threshold_value(threshold_spec, target.dropna().tolist())
        labels = (target > threshold).astype(float)
        details = {
            "target_column": target_column,
            "threshold_spec": threshold_spec,
            "threshold_value": threshold,
        }
        return labels, details

    if task_type == "multiclass":
        task_name = str(experiment_cfg.get("task_name", "")).strip().lower()
        if task_name in {"emotion-id", "emotion_id", "feltemo"}:
            target_column = experiment_cfg.get("target_column", "emotion-id")
            labels = pd.to_numeric(snapshot_df[target_column], errors="coerce").dropna().astype(int)
            details = {
                "target_column": target_column,
                "task_name": "emotion-id",
            }
            return labels, details

        # VA quadrant multiclass
        source_columns = experiment_cfg.get("target_columns") or ["emotion-valence", "emotion-arousal"]
        if len(source_columns) != 2:
            raise ValueError("VA quadrant multiclass requires two target_columns.")
        val_col, ar_col = source_columns
        valence = pd.to_numeric(snapshot_df[val_col], errors="coerce")
        arousal = pd.to_numeric(snapshot_df[ar_col], errors="coerce")

        threshold_cfg = experiment_cfg.get("thresholds", {})
        v_spec = threshold_cfg.get("valence", experiment_cfg.get("threshold", "mean"))
        a_spec = threshold_cfg.get("arousal", experiment_cfg.get("threshold", "mean"))
        v_thr = _resolve_threshold_value(v_spec, valence.dropna().tolist())
        a_thr = _resolve_threshold_value(a_spec, arousal.dropna().tolist())

        valid_mask = valence.notna() & arousal.notna()
        valence = valence[valid_mask]
        arousal = arousal[valid_mask]

        ll = (valence <= v_thr) & (arousal <= a_thr)
        lh = (valence <= v_thr) & (arousal > a_thr)
        hl = (valence > v_thr) & (arousal <= a_thr)
        hh = (valence > v_thr) & (arousal > a_thr)

        labels = pd.Series(np.where(ll, 0, np.where(lh, 1, np.where(hl, 2, 3))), index=valence.index)
        details = {
            "task_name": "va_quadrant",
            "source_columns": source_columns,
            "thresholds": {
                "valence_spec": v_spec,
                "valence_value": v_thr,
                "arousal_spec": a_spec,
                "arousal_value": a_thr,
            },
            "label_mapping": {0: "LL", 1: "LH", 2: "HL", 3: "HH"},
        }
        return labels, details

    if task_type == "regression":
        target_column = experiment_cfg["target_column"]
        values = pd.to_numeric(snapshot_df[target_column], errors="coerce")
        return values, {"target_column": target_column}

    return None, {}


def _window_count_summary(values: List[int]) -> Tuple[int, float, int]:
    if not values:
        return 0, 0.0, 0
    return int(np.min(values)), float(np.median(values)), int(np.max(values))


def _robust_group_outlier_warnings(snapshot_df: pd.DataFrame, signal_columns: List[str]) -> List[str]:
    if snapshot_df.empty:
        return []
    required = ["subject", "recording"] + [col for col in signal_columns if col in snapshot_df.columns]
    if len(required) <= 2:
        return []

    grouped = (
        snapshot_df[required]
        .groupby(["subject", "recording"], dropna=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    if grouped.empty:
        return []

    numeric_cols = [col for col in grouped.columns if col not in {"subject", "recording"}]
    warnings: List[str] = []
    if not numeric_cols:
        return warnings

    medians = grouped[numeric_cols].median(axis=0)
    mad = (grouped[numeric_cols] - medians).abs().median(axis=0)
    mad = mad.replace(0.0, np.nan)

    robust_z = 0.6745 * (grouped[numeric_cols] - medians) / mad
    max_abs_robust_z = robust_z.abs().max(axis=1).fillna(0.0)

    outlier_rows = grouped[max_abs_robust_z > 5.0].copy()
    outlier_rows["max_abs_robust_z"] = max_abs_robust_z[max_abs_robust_z > 5.0]
    for _, row in outlier_rows.sort_values("max_abs_robust_z", ascending=False).iterrows():
        warnings.append(
            "- outlier subject-recording "
            f"({row['subject']}, {row['recording']}), "
            f"max_abs_robust_z={float(row['max_abs_robust_z']):.3f}"
        )
    return warnings


def build_eda_summary(
    snapshot_df: pd.DataFrame,
    task_type: str,
    experiment_cfg: Dict[str, Any],
    label_name_mapping: Mapping[int, str] | None = None,
    window_subject_counts: Dict[str, int] | None = None,
    window_recording_counts: Dict[str, int] | None = None,
    snapshot_manifest: Dict[str, Any] | None = None,
) -> EdaSummaryResult:
    """Build compact EDA summary text and key metrics from cleaned snapshot."""
    window_subject_counts = window_subject_counts or {}
    window_recording_counts = window_recording_counts or {}
    snapshot_manifest = snapshot_manifest or {}

    lines: List[str] = []
    warnings: List[str] = []

    rows, cols = snapshot_df.shape
    subject_count = int(snapshot_df["subject"].nunique()) if "subject" in snapshot_df.columns else 0
    recording_count = int(snapshot_df["recording"].nunique()) if "recording" in snapshot_df.columns else 0

    signal_outlier_info = snapshot_manifest.get("signal_outlier_filter", {})
    lines.append("data_source:")
    lines.append("- basis=cleaned snapshot used for both EDA and training (not raw source signals)")
    if signal_outlier_info:
        lines.append(
            "- signal_outlier_filter="
            f"enabled:{bool(signal_outlier_info.get('enabled', False))}, "
            f"lower_q:{signal_outlier_info.get('lower_quantile')}, "
            f"upper_q:{signal_outlier_info.get('upper_quantile')}, "
            f"rows_dropped:{signal_outlier_info.get('rows_dropped')}"
        )
    else:
        lines.append("- signal_outlier_filter=not_reported")
    lines.append("")

    lines.append(f"shape: rows={rows}, cols={cols}")
    lines.append(f"subjects: {subject_count}")
    lines.append(f"recordings: {recording_count}")
    lines.append("")

    lines.append("signal_distribution (cleaned snapshot after cleaning/dropna/filters):")
    for col in SIGNAL_COLUMNS:
        if col not in snapshot_df.columns:
            lines.append(f"- {col}: missing")
            continue
        stats = _distribution_stats(snapshot_df[col])
        lines.append(_format_distribution(col, stats))
    lines.append("")

    final_labels, label_details = _derive_final_labels(
        snapshot_df=snapshot_df,
        task_type=task_type,
        experiment_cfg=experiment_cfg,
    )

    label_entropy = float("nan")
    minority_ratio = float("nan")

    lines.append("final_label_distribution:")
    if final_labels is None:
        lines.append("- unavailable")
    elif task_type == "regression":
        lines.extend(_regression_distribution_text(final_labels))
        label_entropy = float("nan")
    else:
        class_lines, label_entropy, minority_ratio = _classification_distribution_text(
            final_labels,
            label_name_mapping=label_name_mapping,
        )
        if class_lines:
            lines.extend(class_lines)
        else:
            lines.append("- unavailable")
        lines.append(f"- label_entropy={label_entropy:.6f}")
    lines.append("")

    if window_subject_counts:
        subject_values = list(window_subject_counts.values())
        min_w, med_w, max_w = _window_count_summary(subject_values)
        lines.append(
            "windows_per_subject: "
            f"min={min_w}, median={med_w:.3f}, max={max_w}"
        )

        median_value = float(np.median(subject_values)) if subject_values else 0.0
        low_support = [
            f"{subject}({count})"
            for subject, count in sorted(window_subject_counts.items())
            if median_value > 0 and count < 0.5 * median_value
        ]
        if low_support:
            warnings.append(
                "- low-support subjects (<50% median windows): " + ", ".join(low_support)
            )

    if window_recording_counts:
        rec_values = list(window_recording_counts.values())
        min_w, med_w, max_w = _window_count_summary(rec_values)
        lines.append(
            "windows_per_recording: "
            f"min={min_w}, median={med_w:.3f}, max={max_w}"
        )

        median_value = float(np.median(rec_values)) if rec_values else 0.0
        low_support = [
            f"{recording}({count})"
            for recording, count in sorted(window_recording_counts.items())
            if median_value > 0 and count < 0.5 * median_value
        ]
        if low_support:
            warnings.append(
                "- low-support recordings (<50% median windows): " + ", ".join(low_support)
            )

    if window_subject_counts or window_recording_counts:
        lines.append("")

    warnings.extend(_robust_group_outlier_warnings(snapshot_df=snapshot_df, signal_columns=SIGNAL_COLUMNS))

    if task_type in {"binary", "multiclass"} and np.isfinite(minority_ratio) and minority_ratio < 0.2:
        warnings.append(
            f"- severe class imbalance: minority proportion={minority_ratio:.4f} (<0.2)"
        )

    lines.append("warnings:")
    if warnings:
        lines.extend(warnings)
    else:
        lines.append("- none")

    stats = {
        "rows": rows,
        "cols": cols,
        "subjects": subject_count,
        "recordings": recording_count,
        "label_entropy": label_entropy,
        "minority_ratio": minority_ratio,
        "label_details": label_details,
        "warnings": warnings,
    }

    return EdaSummaryResult(text="\n".join(lines).strip() + "\n", stats=stats)
