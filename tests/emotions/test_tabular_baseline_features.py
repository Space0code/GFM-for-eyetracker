from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.hci_signals import DISTANCE_AVG_COLUMN, FIXATION_DURATION_COLUMN
from emotions.train_baseline import (
    FIXATION_SUMMARY_FEATURE_COLUMNS,
    TABULAR_AGGREGATE_SUFFIXES,
    build_tabular_samples,
    select_tabular_feature_columns,
)


def _baseline_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time-rel-seconds": [0.0, 0.1, 0.2, 0.3, 0.4],
            "x-avg": [1.0, 2.0, 4.0, 8.0, 16.0],
            "y-avg": [2.0, 3.0, 5.0, 9.0, 17.0],
            "pupil-size-left-avg": [3.0, 3.1, 3.2, 3.3, 3.4],
            "pupil-size-right-avg": [3.4, 3.3, 3.2, 3.1, 3.0],
            "distance-left": [60.0, 61.0, 62.0, 63.0, 64.0],
            "distance-right": [64.0, 65.0, 66.0, 67.0, 68.0],
            "fixation-index": [1, 1, 2, 2, pd.NA],
            "fixation-duration": [100.0, 100.0, 200.0, 200.0, 999.0],
            "fixation": [True, True, True, True, False],
            "subject": ["P1"] * 5,
            "recording": ["r1"] * 5,
            "emotion-id": [1] * 5,
            "experiment-type": ["emotion-elicitation"] * 5,
            "emotion-derivation-status": ["ok"] * 5,
        }
    )


def test_tabular_baseline_samples_include_extended_signal_and_fixation_features(tmp_path: Path) -> None:
    data_csv = tmp_path / "hci_baseline.csv"
    _baseline_feature_frame().to_csv(data_csv, index=False)

    feature_columns = [
        "x-avg",
        "y-avg",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
        DISTANCE_AVG_COLUMN,
        FIXATION_DURATION_COLUMN,
    ]
    samples = build_tabular_samples(
        data_filepath=str(data_csv),
        window_length=1,
        window_overlap=0.0,
        min_samples_per_window=2,
        feature_columns=feature_columns,
        target_columns=["emotion-id"],
        target_aggregation="mean",
        dropna_columns=[
            "time-rel-seconds",
            "x-avg",
            "y-avg",
            "pupil-size-left-avg",
            "pupil-size-right-avg",
            "emotion-id",
        ],
    )

    assert len(samples) == 1
    features = samples[0].features
    for signal in feature_columns:
        for suffix in TABULAR_AGGREGATE_SUFFIXES:
            assert f"{signal}_{suffix}" in features
    for feature_name in FIXATION_SUMMARY_FEATURE_COLUMNS:
        assert feature_name in features

    assert np.isclose(features["x-avg_range"], 15.0)
    assert np.isclose(features["x-avg_std"], np.std([1.0, 2.0, 4.0, 8.0, 16.0], ddof=0))
    assert np.isclose(features["distance-avg_mean"], 64.0)
    assert np.isclose(features["fixation-duration_mean"], 120.0)
    assert np.isclose(features["fixation-duration_max"], 200.0)
    assert np.isclose(features["fixation_count"], 2.0)
    assert np.isclose(features["fixation_sample_fraction"], 0.8)
    assert np.isclose(features["fixation-duration_fixation_sum"], 600.0)
    assert np.isclose(features["fixation-duration_fixation_mean"], 150.0)
    assert np.isclose(features["fixation-duration_fixation_max"], 200.0)


def test_tabular_constant_target_aggregation_rejects_mixed_labels(tmp_path: Path) -> None:
    data_csv = tmp_path / "hci_mixed_labels.csv"
    frame = _baseline_feature_frame()
    frame["emotion-id"] = [1, 1, 2, 2, 2]
    frame.to_csv(data_csv, index=False)

    with pytest.raises(ValueError, match="Expected constant target column 'emotion-id'"):
        build_tabular_samples(
            data_filepath=str(data_csv),
            window_length=1,
            window_overlap=0.0,
            min_samples_per_window=2,
            feature_columns=["x-avg", "y-avg"],
            target_columns=["emotion-id"],
            target_aggregation="constant",
            dropna_columns=["time-rel-seconds", "x-avg", "y-avg", "emotion-id"],
        )


def test_select_tabular_feature_columns_keeps_all_aggregates_and_exact_embedding_columns() -> None:
    X = pd.DataFrame(
        columns=[
            "x-avg_mean",
            "x-avg_std",
            "x-avg_iqr",
            "y-avg_mean",
            "distance-avg_q75",
            "fixation-duration_mean",
            "fixation-duration_iqr",
            "fixation_count",
            "fixation_sample_fraction",
            "fixation-duration_fixation_sum",
            "fixation-duration_fixation_mean",
            "fixation-duration_fixation_max",
            "unrelated_mean",
        ]
    )

    selected = select_tabular_feature_columns(
        X,
        ["x-avg", "y-avg", DISTANCE_AVG_COLUMN, FIXATION_DURATION_COLUMN],
    )

    assert selected == [
        "x-avg_mean",
        "x-avg_std",
        "x-avg_iqr",
        "y-avg_mean",
        "distance-avg_q75",
        "fixation-duration_mean",
        "fixation-duration_iqr",
        "fixation_count",
        "fixation_sample_fraction",
        "fixation-duration_fixation_sum",
        "fixation-duration_fixation_mean",
        "fixation-duration_fixation_max",
    ]

    gazemae_X = pd.DataFrame(columns=["gazemae_z_000", "gazemae_z_001", "x-avg_mean"])
    assert select_tabular_feature_columns(gazemae_X, ["gazemae_z_000", "gazemae_z_001"]) == [
        "gazemae_z_000",
        "gazemae_z_001",
    ]
