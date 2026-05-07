from __future__ import annotations

import numpy as np

from emotions.multiclass.baseline_model_multiclass import get_multiclass_baseline_by_name
from emotions.utils import validate_config


def test_majority_multiclass_classifier_predicts_training_majority() -> None:
    model = get_multiclass_baseline_by_name("Majority")
    X_train = np.zeros((5, 2), dtype=float)
    y_train = np.asarray([2, 1, 2, 0, 2])
    X_test = np.ones((3, 2), dtype=float)
    all_classes = np.asarray([0, 1, 2])

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test, all_classes=all_classes)

    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba[:, 2] == 1.0)


def test_random_multiclass_classifier_returns_reproducible_one_hot_predictions() -> None:
    model = get_multiclass_baseline_by_name("Random", random_state=7)
    X_train = np.zeros((5, 2), dtype=float)
    y_train = np.asarray([0, 1, 2, 1, 2])
    X_test = np.ones((8, 2), dtype=float)
    all_classes = np.asarray([0, 1, 2])

    model.fit(X_train, y_train)
    first = model.predict_proba(X_test, all_classes=all_classes)
    second = model.predict_proba(X_test, all_classes=all_classes)

    assert first.shape == (8, 3)
    assert np.allclose(first, second)
    assert np.allclose(first.sum(axis=1), 1.0)
    assert set(np.unique(first)).issubset({0.0, 1.0})


def test_validate_config_accepts_multiclass_random_and_majority_baselines() -> None:
    config = {
        "multiclass_task": {"task_name": "table6-arousal-3class"},
        "dataset": {"data_filepath": "unused.csv"},
        "cross_validation": {"strategies": ["recording_kfold"]},
        "logging": {"results_dir": "unused"},
        "metrics": ["accuracy"],
        "baselines": {"models": ["Random", "Majority"]},
    }

    validate_config(config)
