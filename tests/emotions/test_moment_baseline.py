from __future__ import annotations

from emotions.gazemae_baseline import GAZEMAE_FEATURE_COLUMNS
from emotions.moment_baseline import (
    MOMENT_FEATURE_COLUMNS,
    MOMENT_GAZE_MODEL_NAME,
    MOMENT_GAZE_PUPIL_MODEL_NAME,
    MOMENT_PUPIL_MODEL_NAME,
    resolve_moment_feature_columns,
)


def test_moment_signal_subset_columns_include_requested_sources() -> None:
    assert resolve_moment_feature_columns(MOMENT_GAZE_MODEL_NAME) == ["x-avg", "y-avg"]
    assert resolve_moment_feature_columns(MOMENT_PUPIL_MODEL_NAME) == [
        "pupil-size-left-avg",
        "pupil-size-right-avg",
    ]
    assert resolve_moment_feature_columns(MOMENT_GAZE_PUPIL_MODEL_NAME) == [
        "x-avg",
        "y-avg",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
    ]
    assert resolve_moment_feature_columns("all_signals") == [
        "x-avg",
        "y-avg",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
        "distance-avg",
        "fixation-duration",
    ]


def test_moment_and_fusion_feature_dimensions_are_fixed() -> None:
    assert len(MOMENT_FEATURE_COLUMNS) == 768
    assert len([*MOMENT_FEATURE_COLUMNS, *GAZEMAE_FEATURE_COLUMNS]) == 1280
