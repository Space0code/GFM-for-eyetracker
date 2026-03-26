"""Dataset/config defaults shared across training scripts and suite wrapper."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Sequence

DEFAULT_FEATURE_COLUMNS = [
    "x-avg",
    "y-avg",
    "pupil-size-left-avg",
    "pupil-size-right-avg",
]

DEFAULT_BASE_DROPNA_COLUMNS = [
    "time-rel-seconds",
    "x-avg",
    "y-avg",
    "pupil-size-left-avg",
    "pupil-size-right-avg",
    "subject",
    "recording",
]


def resolve_feature_columns(dataset_cfg: Mapping[str, Any]) -> list[str]:
    """Resolve feature columns from config with a validated default."""
    raw = dataset_cfg.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    if not isinstance(raw, list) or not raw:
        raise ValueError("dataset.feature_columns must be a non-empty list.")
    return [str(column) for column in raw]


def resolve_dropna_columns(
    dataset_cfg: MutableMapping[str, Any],
    target_columns: Sequence[str],
) -> list[str]:
    """Resolve dropna columns, ensuring all target columns are present once."""
    raw = dataset_cfg.get("dropna_columns")
    if raw is None:
        dropna_columns = list(DEFAULT_BASE_DROPNA_COLUMNS)
    elif isinstance(raw, list):
        dropna_columns = [str(column) for column in raw]
    else:
        raise ValueError("dataset.dropna_columns must be a list when provided.")

    for target_column in target_columns:
        token = str(target_column)
        if token not in dropna_columns:
            dropna_columns.append(token)

    dataset_cfg["dropna_columns"] = dropna_columns
    return dropna_columns


def resolve_min_samples_per_window(dataset_cfg: Mapping[str, Any]) -> int:
    """Resolve `min_samples_per_window` with a deterministic fallback."""
    kt = int(dataset_cfg.get("kt", 1))
    ks = int(dataset_cfg.get("ks", 1))
    default_min = max(kt, ks) + 1
    return int(dataset_cfg.get("min_samples_per_window", default_min))


def apply_defaults_when_missing(target_cfg: MutableMapping[str, Any], defaults: Mapping[str, Any]) -> None:
    """Set default key/value pairs only when keys are absent."""
    for key, value in defaults.items():
        target_cfg.setdefault(key, value)


def build_graph_dataset_kwargs(
    dataset_cfg: Mapping[str, Any],
    target_columns: Sequence[str],
    feature_columns: Sequence[str],
    dropna_columns: Sequence[str],
) -> Dict[str, Any]:
    """Build keyword arguments for `SpacioTemporalDataset`."""
    return {
        "root_dir": dataset_cfg.get("data_dir"),
        "data_filepath": dataset_cfg.get("data_filepath"),
        "filter_subjects": dataset_cfg.get("filter_subjects"),
        "filter_recordings": dataset_cfg.get("filter_recordings"),
        "file_list": dataset_cfg.get("file_list"),
        "recursive": dataset_cfg["recursive"],
        "ignore_dirs": dataset_cfg.get("ignore_dirs", []),
        "window_length": dataset_cfg["window_length"],
        "window_overlap": dataset_cfg["window_overlap"],
        "kt": dataset_cfg["kt"],
        "ks": dataset_cfg["ks"],
        "use_edge_weights": dataset_cfg["use_edge_weights"],
        "tau": dataset_cfg["tau"],
        "cache_dir": dataset_cfg.get("cache_dir"),
        "use_cache": dataset_cfg.get("use_cache", True),
        "dropping_emotion_threshold": dataset_cfg.get("dropping_emotion_threshold", -1),
        "feature_columns": list(feature_columns),
        "target_columns": [str(column) for column in target_columns],
        "dropna_columns": list(dropna_columns),
        "experiment_type_column": dataset_cfg.get("experiment_type_column", "experiment-type"),
        "allowed_experiment_types": dataset_cfg.get("allowed_experiment_types"),
        "label_quality_column": dataset_cfg.get("label_quality_column"),
        "allowed_label_quality_values": dataset_cfg.get("allowed_label_quality_values"),
        "target_aggregation": dataset_cfg.get("target_aggregation", "mean"),
    }


def build_tabular_samples_kwargs(
    dataset_cfg: Mapping[str, Any],
    target_columns: Sequence[str],
    feature_columns: Sequence[str],
    dropna_columns: Sequence[str],
    min_samples_per_window: int,
) -> Dict[str, Any]:
    """Build keyword arguments for `build_tabular_samples`."""
    return {
        "data_dir": dataset_cfg.get("data_dir"),
        "data_filepath": dataset_cfg.get("data_filepath"),
        "filter_subjects": dataset_cfg.get("filter_subjects"),
        "filter_recordings": dataset_cfg.get("filter_recordings"),
        "file_list": dataset_cfg.get("file_list"),
        "window_length": dataset_cfg.get("window_length", 10),
        "window_overlap": dataset_cfg.get("window_overlap", 0.0),
        "min_samples_per_window": int(min_samples_per_window),
        "dropping_emotion_threshold": dataset_cfg.get("dropping_emotion_threshold", -1),
        "feature_columns": list(feature_columns),
        "target_columns": [str(column) for column in target_columns],
        "target_aggregation": dataset_cfg.get("target_aggregation", "mean"),
        "dropna_columns": list(dropna_columns),
        "experiment_type_column": dataset_cfg.get("experiment_type_column", "experiment-type"),
        "allowed_experiment_types": dataset_cfg.get("allowed_experiment_types"),
        "label_quality_column": dataset_cfg.get("label_quality_column"),
        "allowed_label_quality_values": dataset_cfg.get("allowed_label_quality_values"),
    }
