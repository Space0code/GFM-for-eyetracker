"""Cleaned per-experiment snapshot builder for HCI experiment suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from data.data import clean_dataset


DEFAULT_FEATURE_COLUMNS = [
    "x-avg",
    "y-avg",
    "pupil-size-left-avg",
    "pupil-size-right-avg",
]

SNAPSHOT_CACHE_VERSION = "snapshot-cache-v1"

_SCOPE_DIRS = {
    "emotion-elicitation": ["emotion-elicitation"],
    "image-tagging-1": ["image-tagging-1"],
    "image-tagging-2": ["image-tagging-2"],
    "video-tagging": ["video-tagging"],
    "pooled-tagging": ["image-tagging-1", "image-tagging-2", "video-tagging"],
}


@dataclass
class SnapshotBuildResult:
    """Result bundle for one snapshot build."""

    dataframe: pd.DataFrame
    manifest: Dict[str, Any]
    source_files: List[str]


def _resolve_scope_directories(source_root: Path, scope: str) -> List[Path]:
    if scope not in _SCOPE_DIRS:
        raise ValueError(
            f"Unsupported scope '{scope}'. Supported scopes: {sorted(_SCOPE_DIRS.keys())}"
        )
    return [source_root / rel for rel in _SCOPE_DIRS[scope]]


def _collect_scope_files(scope_dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for scope_dir in scope_dirs:
        if not scope_dir.exists():
            continue
        files.extend(sorted(scope_dir.glob("*.csv")))
    return files


def _apply_global_row_filters(df: pd.DataFrame, dataset_cfg: Dict[str, Any]) -> pd.DataFrame:
    experiment_type_column = dataset_cfg.get("experiment_type_column", "experiment-type")
    allowed_experiment_types = dataset_cfg.get("allowed_experiment_types")
    if allowed_experiment_types and experiment_type_column in df.columns:
        df = df[df[experiment_type_column].isin(allowed_experiment_types)]

    label_quality_column = dataset_cfg.get("label_quality_column")
    allowed_quality_values = dataset_cfg.get("allowed_label_quality_values")
    if label_quality_column and allowed_quality_values and label_quality_column in df.columns:
        df = df[df[label_quality_column].isin(allowed_quality_values)]

    filter_subjects = dataset_cfg.get("filter_subjects")
    if filter_subjects is not None and "subject" in df.columns:
        df = df[df["subject"].isin(filter_subjects)]

    exclude_subjects = dataset_cfg.get("exclude_subjects")
    if exclude_subjects is not None and "subject" in df.columns:
        df = df[~df["subject"].isin(exclude_subjects)]

    filter_recordings = dataset_cfg.get("filter_recordings")
    if filter_recordings is not None and "recording" in df.columns:
        df = df[df["recording"].isin(filter_recordings)]

    return df.reset_index(drop=True)


def _clean_group(
    group_df: pd.DataFrame,
    feature_columns: List[str],
    dropna_columns: List[str] | None,
) -> pd.DataFrame:
    if len(group_df) == 0:
        return group_df.reset_index(drop=True)

    group_df = group_df.sort_values("time-rel-seconds").reset_index(drop=True)

    required_clean_cols = ["time-rel-seconds"] + feature_columns
    missing_required = [col for col in required_clean_cols if col not in group_df.columns]
    if missing_required:
        raise ValueError(f"Missing required signal columns for cleaning: {missing_required}")

    cleaned = clean_dataset(
        group_df,
        required_cols=required_clean_cols,
        interpolation_cols=feature_columns,
    )

    if dropna_columns is None:
        cleaned = cleaned.dropna()
    else:
        missing_dropna = [col for col in dropna_columns if col not in cleaned.columns]
        if missing_dropna:
            raise ValueError(
                "Missing configured dropna columns in snapshot group: "
                f"{missing_dropna}"
            )
        cleaned = cleaned.dropna(subset=dropna_columns)

    return cleaned.reset_index(drop=True)


def _should_drop_group_by_threshold(
    group_df: pd.DataFrame,
    target_columns: List[str],
    dropping_threshold: float,
) -> bool:
    if not target_columns:
        return False

    present_targets = [col for col in target_columns if col in group_df.columns]
    if not present_targets:
        return False

    numeric_df = group_df[present_targets].apply(pd.to_numeric, errors="coerce")
    if numeric_df.empty:
        return False

    all_below_or_equal = (numeric_df <= dropping_threshold).all(axis=1).all()
    return bool(all_below_or_equal)


def _compute_dataframe_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"").hexdigest()

    ordered = df.copy()
    ordered = ordered.sort_values([col for col in ["subject", "recording", "time-rel-seconds"] if col in ordered.columns])
    hashed = pd.util.hash_pandas_object(ordered, index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 fingerprint for one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sequence_for_key(values: Any) -> Any:
    """Normalize sequence-like values so order-insensitive params hash consistently."""
    if values is None:
        return None
    if not isinstance(values, list):
        return values
    return sorted([str(item) for item in values])


def _dataset_affecting_cache_payload(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only dataset parameters that affect snapshot rows/columns."""
    signal_outlier_cfg = dataset_cfg.get("signal_outlier_filter") or {}
    if not isinstance(signal_outlier_cfg, dict):
        signal_outlier_cfg = {}

    return {
        "feature_columns": _normalize_sequence_for_key(dataset_cfg.get("feature_columns") or DEFAULT_FEATURE_COLUMNS),
        "dropna_columns": _normalize_sequence_for_key(dataset_cfg.get("dropna_columns")),
        "filter_subjects": _normalize_sequence_for_key(dataset_cfg.get("filter_subjects")),
        "filter_recordings": _normalize_sequence_for_key(dataset_cfg.get("filter_recordings")),
        "exclude_subjects": _normalize_sequence_for_key(dataset_cfg.get("exclude_subjects")),
        "experiment_type_column": dataset_cfg.get("experiment_type_column", "experiment-type"),
        "allowed_experiment_types": _normalize_sequence_for_key(dataset_cfg.get("allowed_experiment_types")),
        "label_quality_column": dataset_cfg.get("label_quality_column"),
        "allowed_label_quality_values": _normalize_sequence_for_key(dataset_cfg.get("allowed_label_quality_values")),
        "dropping_emotion_threshold": float(dataset_cfg.get("dropping_emotion_threshold", -np.inf)),
        "signal_outlier_filter": {
            "enabled": bool(signal_outlier_cfg.get("enabled", True)),
            "lower_quantile": float(signal_outlier_cfg.get("lower_quantile", 0.01)),
            "upper_quantile": float(signal_outlier_cfg.get("upper_quantile", 0.99)),
            "columns": _normalize_sequence_for_key(signal_outlier_cfg.get("columns") or DEFAULT_FEATURE_COLUMNS),
        },
    }


def _build_source_signatures(files: List[Path], source_root: Path) -> List[Dict[str, Any]]:
    """Build deterministic signatures for source file change detection."""
    signatures: List[Dict[str, Any]] = []
    for file_path in sorted(files):
        stat = file_path.stat()
        try:
            rel_path = str(file_path.resolve().relative_to(source_root.resolve()))
        except ValueError:
            rel_path = str(file_path.resolve())
        signatures.append(
            {
                "path": rel_path,
                "size_bytes": int(stat.st_size),
                "sha256": _file_sha256(file_path),
            }
        )
    return signatures


def _build_snapshot_cache_key(
    source_root: Path,
    scope: str,
    dataset_cfg: Dict[str, Any],
    target_columns: List[str],
    source_signatures: List[Dict[str, Any]],
) -> str:
    """Create deterministic cache key for one cleaned snapshot definition."""
    payload = {
        "cache_version": SNAPSHOT_CACHE_VERSION,
        "source_root": str(source_root.resolve()),
        "scope": scope,
        "target_columns": _normalize_sequence_for_key(target_columns),
        "dataset_affecting_params": _dataset_affecting_cache_payload(dataset_cfg),
        "source_file_signatures": source_signatures,
    }
    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def _apply_signal_quantile_filter(
    snapshot_df: pd.DataFrame,
    dataset_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop rows outside configured signal quantile bounds.

    By default, this trims to [q01, q99] on all four core signals.
    """
    signal_cfg = dataset_cfg.get("signal_outlier_filter") or {}
    if not isinstance(signal_cfg, dict):
        raise ValueError("dataset.signal_outlier_filter must be a dictionary when provided.")

    enabled = bool(signal_cfg.get("enabled", True))
    lower_q = float(signal_cfg.get("lower_quantile", 0.01))
    upper_q = float(signal_cfg.get("upper_quantile", 0.99))
    columns = signal_cfg.get("columns") or DEFAULT_FEATURE_COLUMNS

    if lower_q < 0 or lower_q > 1 or upper_q < 0 or upper_q > 1:
        raise ValueError("signal_outlier_filter quantiles must be within [0, 1].")
    if lower_q >= upper_q:
        raise ValueError("signal_outlier_filter.lower_quantile must be < upper_quantile.")

    present_columns = [col for col in columns if col in snapshot_df.columns]
    info: Dict[str, Any] = {
        "enabled": enabled,
        "lower_quantile": lower_q,
        "upper_quantile": upper_q,
        "columns": list(columns),
        "present_columns": present_columns,
        "thresholds": {},
        "rows_before": int(len(snapshot_df)),
        "rows_after": int(len(snapshot_df)),
        "rows_dropped": 0,
    }

    if not enabled or snapshot_df.empty or not present_columns:
        return snapshot_df, info

    numeric_df = snapshot_df[present_columns].apply(pd.to_numeric, errors="coerce")
    keep_mask = np.ones(len(snapshot_df), dtype=bool)

    for col in present_columns:
        series = numeric_df[col]
        lower = float(series.quantile(lower_q))
        upper = float(series.quantile(upper_q))
        info["thresholds"][col] = {"lower": lower, "upper": upper}
        keep_mask &= series.ge(lower).to_numpy() & series.le(upper).to_numpy()

    filtered_df = snapshot_df.loc[keep_mask].reset_index(drop=True)
    info["rows_after"] = int(len(filtered_df))
    info["rows_dropped"] = int(info["rows_before"] - info["rows_after"])
    return filtered_df, info


def build_clean_snapshot_dataframe(
    source_data_root: str,
    scope: str,
    dataset_cfg: Dict[str, Any],
    target_columns: List[str],
    experiment_id: str,
    threshold_description: Dict[str, Any] | None = None,
    use_cache: bool = False,
    cache_dir: str | None = None,
) -> SnapshotBuildResult:
    """Build cleaned snapshot dataframe aligned with trainer cleaning/filtering rules."""
    source_root = Path(source_data_root)
    scope_dirs = _resolve_scope_directories(source_root=source_root, scope=scope)
    files = _collect_scope_files(scope_dirs)
    source_files = [str(path) for path in files]

    source_signatures: List[Dict[str, Any]] = []
    cache_key: str | None = None
    cache_csv_path: Path | None = None
    cache_manifest_path: Path | None = None

    if files and use_cache:
        source_signatures = _build_source_signatures(files=files, source_root=source_root)
        cache_key = _build_snapshot_cache_key(
            source_root=source_root,
            scope=scope,
            dataset_cfg=dataset_cfg,
            target_columns=target_columns,
            source_signatures=source_signatures,
        )
        resolved_cache_dir = Path(cache_dir) if cache_dir else (source_root / ".snapshot_cache")
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_csv_path = resolved_cache_dir / f"{cache_key}.csv"
        cache_manifest_path = resolved_cache_dir / f"{cache_key}.yaml"

        if cache_csv_path.exists() and cache_manifest_path.exists():
            cached_df = pd.read_csv(cache_csv_path)
            with cache_manifest_path.open("r", encoding="utf-8") as handle:
                cached_manifest = yaml.safe_load(handle) or {}

            if not isinstance(cached_manifest, dict):
                cached_manifest = {}

            cached_manifest["experiment_id"] = experiment_id
            cached_manifest["scope"] = scope
            cached_manifest["threshold"] = threshold_description
            cached_manifest["target_columns"] = target_columns
            cached_manifest["source_files"] = source_files
            cached_manifest["source_file_count"] = len(files)
            cached_manifest["source_file_signatures"] = source_signatures
            cached_manifest["cache"] = {
                "enabled": True,
                "cache_key": cache_key,
                "cache_hit": True,
                "cache_csv_path": str(cache_csv_path),
                "cache_manifest_path": str(cache_manifest_path),
                "cache_version": SNAPSHOT_CACHE_VERSION,
            }

            return SnapshotBuildResult(
                dataframe=cached_df,
                manifest=cached_manifest,
                source_files=source_files,
            )

    if not files:
        manifest = {
            "experiment_id": experiment_id,
            "scope": scope,
            "status": "empty",
            "reason": "No source files found for scope.",
            "source_directories": [str(path) for path in scope_dirs],
            "source_file_count": 0,
            "row_count": 0,
            "column_count": 0,
            "subjects": [],
            "recordings": [],
            "target_columns": target_columns,
            "threshold": threshold_description,
            "snapshot_hash": _compute_dataframe_hash(pd.DataFrame()),
            "source_file_signatures": [],
            "cache": {
                "enabled": bool(use_cache),
                "cache_key": cache_key,
                "cache_hit": False,
                "cache_csv_path": str(cache_csv_path) if cache_csv_path else None,
                "cache_manifest_path": str(cache_manifest_path) if cache_manifest_path else None,
                "cache_version": SNAPSHOT_CACHE_VERSION,
            },
        }
        return SnapshotBuildResult(dataframe=pd.DataFrame(), manifest=manifest, source_files=[])

    feature_columns = dataset_cfg.get("feature_columns") or DEFAULT_FEATURE_COLUMNS
    dropna_columns = dataset_cfg.get("dropna_columns")
    dropping_threshold = float(dataset_cfg.get("dropping_emotion_threshold", -np.inf))

    cleaned_groups: List[pd.DataFrame] = []

    for file_path in files:
        df = pd.read_csv(file_path)
        if df.empty:
            continue

        df = _apply_global_row_filters(df=df, dataset_cfg=dataset_cfg)
        if df.empty:
            continue

        # Match data_filepath mode behavior from existing loaders.
        if "subject" not in df.columns or "recording" not in df.columns:
            raise ValueError(
                f"Snapshot source file missing required 'subject'/'recording' columns: {file_path}"
            )

        df = df.dropna(subset=["subject", "recording"]).reset_index(drop=True)
        if df.empty:
            continue

        grouped = df.groupby(["subject", "recording"], sort=True)
        for _, group_df in grouped:
            cleaned = _clean_group(
                group_df=group_df,
                feature_columns=feature_columns,
                dropna_columns=dropna_columns,
            )
            if cleaned.empty:
                continue

            if _should_drop_group_by_threshold(
                group_df=cleaned,
                target_columns=target_columns,
                dropping_threshold=dropping_threshold,
            ):
                continue

            cleaned_groups.append(cleaned)

    if cleaned_groups:
        snapshot_df = pd.concat(cleaned_groups, axis=0, ignore_index=True)
        sort_cols = [col for col in ["subject", "recording", "time-rel-seconds"] if col in snapshot_df.columns]
        if sort_cols:
            snapshot_df = snapshot_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        snapshot_df = pd.DataFrame()

    snapshot_df, signal_outlier_info = _apply_signal_quantile_filter(
        snapshot_df=snapshot_df,
        dataset_cfg=dataset_cfg,
    )

    subjects = sorted(snapshot_df["subject"].dropna().astype(str).unique().tolist()) if "subject" in snapshot_df.columns else []
    recordings = (
        sorted(snapshot_df["recording"].dropna().astype(str).unique().tolist())
        if "recording" in snapshot_df.columns
        else []
    )

    manifest = {
        "experiment_id": experiment_id,
        "scope": scope,
        "status": "ok" if len(snapshot_df) > 0 else "empty",
        "source_directories": [str(path) for path in scope_dirs],
        "source_file_count": len(files),
        "source_files": source_files,
        "row_count": int(len(snapshot_df)),
        "column_count": int(snapshot_df.shape[1]) if not snapshot_df.empty else 0,
        "subjects": subjects,
        "subject_count": len(subjects),
        "recordings": recordings,
        "recording_count": len(recordings),
        "target_columns": target_columns,
        "threshold": threshold_description,
        "dataset_filters": {
            "filter_subjects": dataset_cfg.get("filter_subjects"),
            "filter_recordings": dataset_cfg.get("filter_recordings"),
            "exclude_subjects": dataset_cfg.get("exclude_subjects"),
            "allowed_experiment_types": dataset_cfg.get("allowed_experiment_types"),
            "label_quality_column": dataset_cfg.get("label_quality_column"),
            "allowed_label_quality_values": dataset_cfg.get("allowed_label_quality_values"),
            "dropna_columns": dropna_columns,
            "feature_columns": feature_columns,
            "dropping_emotion_threshold": dropping_threshold,
            "signal_outlier_filter": {
                "enabled": signal_outlier_info["enabled"],
                "lower_quantile": signal_outlier_info["lower_quantile"],
                "upper_quantile": signal_outlier_info["upper_quantile"],
                "columns": signal_outlier_info["columns"],
            },
        },
        "signal_outlier_filter": signal_outlier_info,
        "snapshot_hash": _compute_dataframe_hash(snapshot_df),
        "source_file_signatures": source_signatures,
        "cache": {
            "enabled": bool(use_cache),
            "cache_key": cache_key,
            "cache_hit": False,
            "cache_csv_path": str(cache_csv_path) if cache_csv_path else None,
            "cache_manifest_path": str(cache_manifest_path) if cache_manifest_path else None,
            "cache_version": SNAPSHOT_CACHE_VERSION,
        },
    }

    if use_cache and cache_csv_path and cache_manifest_path:
        try:
            snapshot_df.to_csv(cache_csv_path, index=False)
            with cache_manifest_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(manifest, handle, sort_keys=False)
        except Exception:
            # Snapshot cache is an optimization only; do not fail experiment build.
            pass

    return SnapshotBuildResult(
        dataframe=snapshot_df,
        manifest=manifest,
        source_files=source_files,
    )


def write_snapshot_artifacts(
    snapshot: SnapshotBuildResult,
    snapshot_csv_path: str,
    manifest_yaml_path: str,
) -> Tuple[str, str]:
    """Persist snapshot dataframe and manifest to disk."""
    csv_path = Path(snapshot_csv_path)
    yaml_path = Path(manifest_yaml_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot.dataframe.to_csv(csv_path, index=False)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot.manifest, handle, sort_keys=False)

    return str(csv_path), str(yaml_path)
