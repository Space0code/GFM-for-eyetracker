"""Utilities for preserving and deriving HCI eye-tracking signals.

These helpers keep optional HCI signals consistent across preprocessing,
snapshot building, tabular baselines, and graph construction. They are meant to
be called before cleaning/interpolation so membership columns such as
``fixation-index`` are preserved as identifiers rather than treated as
continuous signals.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


BASE_NODE_FEATURE_COLUMNS = [
    "x-avg",
    "y-avg",
    "pupil-size-left-avg",
    "pupil-size-right-avg",
]
DISTANCE_AVG_COLUMN = "distance-avg"
FIXATION_DURATION_COLUMN = "fixation-duration"
FIXATION_INDEX_COLUMN = "fixation-index"
FIXATION_COLUMN = "fixation"
TIME_WINDOW_NORMALIZED_COLUMN = "time-window-normalized"
DISTANCE_SOURCE_COLUMNS = ["distance-left", "distance-right"]


def resolve_optional_hci_feature_columns(
    feature_columns: Sequence[str] | None,
    *,
    use_distance_avg: bool = True,
    use_fixation_duration: bool = True,
    use_relative_time: bool = False,
) -> list[str]:
    """Resolve node feature columns with optional HCI signals appended once."""
    resolved = list(feature_columns) if feature_columns is not None else list(BASE_NODE_FEATURE_COLUMNS)
    if use_relative_time and TIME_WINDOW_NORMALIZED_COLUMN not in resolved:
        resolved.append(TIME_WINDOW_NORMALIZED_COLUMN)
    if use_distance_avg and DISTANCE_AVG_COLUMN not in resolved:
        resolved.append(DISTANCE_AVG_COLUMN)
    if use_fixation_duration and FIXATION_DURATION_COLUMN not in resolved:
        resolved.append(FIXATION_DURATION_COLUMN)
    return list(dict.fromkeys(str(column) for column in resolved))


def feature_interpolation_columns(feature_columns: Sequence[str]) -> list[str]:
    """Return feature columns that should be linearly interpolated."""
    return [
        column
        for column in feature_columns
        if column not in {FIXATION_DURATION_COLUMN, TIME_WINDOW_NORMALIZED_COLUMN}
    ]


def raw_signal_feature_columns(feature_columns: Sequence[str]) -> list[str]:
    """Return feature columns that must exist before window-local derivation."""
    return [column for column in feature_columns if column != TIME_WINDOW_NORMALIZED_COLUMN]


def _coerce_fixation_bool(series: pd.Series) -> pd.Series:
    """Convert common fixation encodings to nullable boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.gt(0).astype("boolean")

    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "t": True,
            "yes": True,
            "y": True,
            "1": True,
            "fixation": True,
            "false": False,
            "f": False,
            "no": False,
            "n": False,
            "0": False,
            "nan": False,
            "": False,
        }
    )
    return mapped.astype("boolean")


def prepare_hci_eye_tracking_signals(
    df: pd.DataFrame,
    *,
    fixation_duration_fill_value: float = 0.0,
) -> pd.DataFrame:
    """Derive distance average and make fixation duration safe for features.

    ``distance-avg`` is the row-wise mean of ``distance-left`` and
    ``distance-right`` with pandas' default skip-NaN behavior. ``fixation-index``
    is intentionally left uninterpolated and is used only as group membership.
    Missing or non-fixation ``fixation-duration`` values are filled with zero.
    """
    if (
        DISTANCE_AVG_COLUMN not in df.columns or df[DISTANCE_AVG_COLUMN].isna().all()
    ) and any(column in df.columns for column in DISTANCE_SOURCE_COLUMNS):
        present_distance_cols = [column for column in DISTANCE_SOURCE_COLUMNS if column in df.columns]
        for column in present_distance_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df[DISTANCE_AVG_COLUMN] = df[present_distance_cols].mean(axis=1, skipna=True)

    if FIXATION_DURATION_COLUMN in df.columns:
        duration = pd.to_numeric(df[FIXATION_DURATION_COLUMN], errors="coerce")

        if FIXATION_COLUMN in df.columns:
            fixation_mask = _coerce_fixation_bool(df[FIXATION_COLUMN]).fillna(False)
            duration = duration.mask(~fixation_mask, fixation_duration_fill_value)
            df[FIXATION_COLUMN] = fixation_mask.astype("boolean")
        elif FIXATION_INDEX_COLUMN in df.columns:
            fixation_mask = pd.to_numeric(df[FIXATION_INDEX_COLUMN], errors="coerce").notna() & duration.gt(0)
            df[FIXATION_COLUMN] = fixation_mask.astype("boolean")

        df[FIXATION_DURATION_COLUMN] = duration.fillna(fixation_duration_fill_value)
    elif FIXATION_COLUMN in df.columns:
        df[FIXATION_COLUMN] = _coerce_fixation_bool(df[FIXATION_COLUMN]).fillna(False).astype("boolean")

    return df
