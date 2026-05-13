import numpy as np

from emotions.multiclass.train_multiclass import (
    _raw_to_label,
    _resolve_fold_context,
    _resolve_task_definition,
    _select_class_downsample_indices,
)


def test_va_quadrant_encoder_uses_strict_greater_than_threshold() -> None:
    task_def = {
        "mode": "va-quadrant",
        "threshold_specs": {"valence": "mean", "arousal": "mean"},
    }

    train_raw = np.array(
        [
            [3.0, 3.0],
            [7.0, 7.0],
            [3.0, 7.0],
            [7.0, 3.0],
        ],
        dtype=float,
    )

    context = _resolve_fold_context(task_def=task_def, train_raw_targets=train_raw)
    assert context["valence_threshold"] == 5.0
    assert context["arousal_threshold"] == 5.0

    raw = np.array(
        [
            [5.0, 5.0],  # equal threshold => low-low (0)
            [5.0, 6.0],  # low-high (1)
            [6.0, 5.0],  # high-low (2)
            [6.0, 6.0],  # high-high (3)
        ],
        dtype=float,
    )

    labels = _raw_to_label(task_def=task_def, raw_values=raw, fold_context=context)
    assert labels.tolist() == [0, 1, 2, 3]


def test_table6_task_definition_requires_enable_flag() -> None:
    multiclass_task_cfg = {
        "task_name": "table6-arousal-3class",
        "target_column": "emotion-id",
        "table6_class_mapping": {0: 0, 1: 2},
    }
    dataset_cfg = {}

    try:
        _resolve_task_definition(multiclass_task_cfg=multiclass_task_cfg, dataset_cfg=dataset_cfg)
    except ValueError as exc:
        assert "Table-6 multiclass task requested but disabled" in str(exc)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("Expected Table-6 task to require an enable flag.")


def test_table6_label_mapping_maps_unknown_to_negative_one() -> None:
    task_def = {
        "mode": "table6-3class",
        "table6_class_mapping": {0: 0, 1: 2, 4: 1},
    }
    raw = np.array([[0.0], [1.0], [4.0], [10.0]], dtype=float)
    labels = _raw_to_label(task_def=task_def, raw_values=raw, fold_context={})
    assert labels.tolist() == [0, 2, 1, -1]


def test_table6_class_downsampling_matches_target_class_count() -> None:
    task_def = {
        "mode": "table6-3class",
        "table6_class_mapping": {0: 0, 1: 2, 2: 0, 3: 2, 4: 1},
        "drop_unmapped_labels": True,
    }
    raw = np.array([[0.0], [0.0], [2.0], [2.0], [1.0], [3.0], [4.0], [99.0]], dtype=float)

    selected, metadata = _select_class_downsample_indices(
        raw_targets=raw,
        task_def=task_def,
        downsampling_cfg={
            "enabled": True,
            "strategy": "match_class_count",
            "source_class": 0,
            "target_class": 2,
            "random_state": 42,
        },
    )

    labels = _raw_to_label(task_def=task_def, raw_values=raw[selected], fold_context={})
    assert np.bincount(labels, minlength=3).tolist() == [2, 1, 2]
    assert metadata["before_counts"] == {0: 4, 1: 1, 2: 2}
    assert metadata["after_counts"] == {0: 2, 1: 1, 2: 2}
