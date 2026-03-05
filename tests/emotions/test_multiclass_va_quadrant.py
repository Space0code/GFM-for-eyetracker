import numpy as np

from emotions.multiclass.train_multiclass import _raw_to_label, _resolve_fold_context


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
