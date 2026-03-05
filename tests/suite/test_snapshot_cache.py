from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from emotions.suite.data_snapshot import build_clean_snapshot_dataframe


def _write_emotion_scope_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "time-rel-seconds": [0.0, 0.2, 0.4, 0.6, 0.0, 0.2, 0.4, 0.6],
            "x-avg": [1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3],
            "y-avg": [3.0, 3.1, 3.2, 3.3, 4.0, 4.1, 4.2, 4.3],
            "pupil-size-left-avg": [5.0, 5.1, 5.2, 5.3, 6.0, 6.1, 6.2, 6.3],
            "pupil-size-right-avg": [5.5, 5.6, 5.7, 5.8, 6.5, 6.6, 6.7, 6.8],
            "subject": ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2"],
            "recording": ["r1", "r1", "r1", "r1", "r2", "r2", "r2", "r2"],
            "experiment-type": ["emotion-elicitation"] * 8,
            "emotion-derivation-status": ["ok"] * 8,
            "emotion-valence": [6.0, 6.0, 6.0, 6.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    df.to_csv(path, index=False)


def _dataset_cfg() -> dict:
    return {
        "feature_columns": ["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"],
        "dropna_columns": [
            "time-rel-seconds",
            "x-avg",
            "y-avg",
            "pupil-size-left-avg",
            "pupil-size-right-avg",
            "subject",
            "recording",
            "emotion-valence",
        ],
        "experiment_type_column": "experiment-type",
        "allowed_experiment_types": ["emotion-elicitation"],
        "label_quality_column": "emotion-derivation-status",
        "allowed_label_quality_values": ["ok"],
        "dropping_emotion_threshold": -1.0,
        "filter_subjects": None,
        "filter_recordings": None,
    }


def test_snapshot_cache_hit_ignores_non_data_params(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "processed" / "hci-tagging"
    _write_emotion_scope_csv(source_root / "emotion-elicitation" / "sample.csv")
    cache_dir = tmp_path / "snapshot_cache"

    cfg = _dataset_cfg()
    first = build_clean_snapshot_dataframe(
        source_data_root=str(source_root),
        scope="emotion-elicitation",
        dataset_cfg=cfg,
        target_columns=["emotion-valence"],
        experiment_id="exp_a",
        threshold_description={"threshold": "mean"},
        use_cache=True,
        cache_dir=str(cache_dir),
    )
    second = build_clean_snapshot_dataframe(
        source_data_root=str(source_root),
        scope="emotion-elicitation",
        dataset_cfg=cfg,
        target_columns=["emotion-valence"],
        experiment_id="exp_b",
        threshold_description={"threshold": 0.5},
        use_cache=True,
        cache_dir=str(cache_dir),
    )

    assert first.manifest["cache"]["enabled"] is True
    assert first.manifest["cache"]["cache_hit"] is False
    assert second.manifest["cache"]["cache_hit"] is True
    assert first.manifest["cache"]["cache_key"] == second.manifest["cache"]["cache_key"]
    assert_frame_equal(first.dataframe, second.dataframe, check_dtype=False)


def test_snapshot_cache_invalidates_on_data_affecting_filter_change(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "processed" / "hci-tagging"
    _write_emotion_scope_csv(source_root / "emotion-elicitation" / "sample.csv")
    cache_dir = tmp_path / "snapshot_cache"

    cfg_all = _dataset_cfg()
    cfg_subset = deepcopy(cfg_all)
    cfg_subset["filter_subjects"] = ["P1"]

    all_subjects = build_clean_snapshot_dataframe(
        source_data_root=str(source_root),
        scope="emotion-elicitation",
        dataset_cfg=cfg_all,
        target_columns=["emotion-valence"],
        experiment_id="exp_all",
        threshold_description={"threshold": "mean"},
        use_cache=True,
        cache_dir=str(cache_dir),
    )
    subset = build_clean_snapshot_dataframe(
        source_data_root=str(source_root),
        scope="emotion-elicitation",
        dataset_cfg=cfg_subset,
        target_columns=["emotion-valence"],
        experiment_id="exp_subset",
        threshold_description={"threshold": "mean"},
        use_cache=True,
        cache_dir=str(cache_dir),
    )

    assert subset.manifest["cache"]["cache_hit"] is False
    assert all_subjects.manifest["cache"]["cache_key"] != subset.manifest["cache"]["cache_key"]
    assert len(subset.dataframe) < len(all_subjects.dataframe)


def test_snapshot_cache_key_is_stable_for_filter_order(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "processed" / "hci-tagging"
    _write_emotion_scope_csv(source_root / "emotion-elicitation" / "sample.csv")
    cache_dir = tmp_path / "snapshot_cache"

    cfg_a = _dataset_cfg()
    cfg_b = _dataset_cfg()
    cfg_a["filter_subjects"] = ["P1", "P2"]
    cfg_b["filter_subjects"] = ["P2", "P1"]

    first = build_clean_snapshot_dataframe(
        source_data_root=str(source_root),
        scope="emotion-elicitation",
        dataset_cfg=cfg_a,
        target_columns=["emotion-valence"],
        experiment_id="exp_order_a",
        threshold_description={"threshold": "mean"},
        use_cache=True,
        cache_dir=str(cache_dir),
    )
    second = build_clean_snapshot_dataframe(
        source_data_root=str(source_root),
        scope="emotion-elicitation",
        dataset_cfg=cfg_b,
        target_columns=["emotion-valence"],
        experiment_id="exp_order_b",
        threshold_description={"threshold": "mean"},
        use_cache=True,
        cache_dir=str(cache_dir),
    )

    assert first.manifest["cache"]["cache_key"] == second.manifest["cache"]["cache_key"]
    assert second.manifest["cache"]["cache_hit"] is True
