from __future__ import annotations

import numpy as np
import pytest
import torch

from emotions.multiclass.baseline_model_multiclass import get_multiclass_baseline_by_name
from emotions.multiclass import train_multiclass
from emotions.moment_baseline import MOMENT_MODEL_NAMES
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


def test_multiclass_factory_accepts_moment_embedding_baselines() -> None:
    for model_name in sorted(MOMENT_MODEL_NAMES):
        model = get_multiclass_baseline_by_name(model_name)

        assert model.name == model_name


def test_resolve_gnn_model_names_accepts_list_and_canonicalizes() -> None:
    config = {
        "gnn": {
            "models": [
                "basic-gcn",
                "HeteroGCNMean",
                "hetero_gcn_mlp",
                "HeteroGCNMLPWeights",
                "basicgcn",
            ]
        }
    }

    assert train_multiclass.resolve_gnn_model_names(config) == [
        "BasicGCN",
        "HeteroGCNMean",
        "HeteroGCNMLP",
        "HeteroGCNMLPWeights",
    ]


def test_resolve_gnn_model_names_falls_back_to_single_model_version() -> None:
    config = {"gnn": {"model": {"model_version": "HeteroGCNMLP"}}}

    assert train_multiclass.resolve_gnn_model_names(config) == ["HeteroGCNMLP"]


def test_resolve_gnn_model_names_rejects_invalid_names() -> None:
    with pytest.raises(ValueError, match="Unsupported GNN model"):
        train_multiclass.resolve_gnn_model_names({"gnn": {"models": ["ImaginaryGNN"]}})


def test_gnn_fold_failure_policy_continues_after_one_variant_fails(tmp_path, monkeypatch) -> None:
    config = {
        "gnn": {
            "models": ["BasicGCN", "HeteroGCNMean"],
            "model": {"model_version": "BasicGCN"},
            "training": {"batch_size": 1},
        }
    }

    monkeypatch.setattr(
        train_multiclass,
        "_prepare_gnn_fold_data",
        lambda **kwargs: (["train"], ["val"], ["test"], ["train_loader"], ["val_loader"], ["test_loader"]),
    )

    trained_models: list[str] = []

    def fake_train_one_gnn_model(**kwargs):
        model_name = kwargs["model_name"]
        trained_models.append(model_name)
        if model_name == "BasicGCN":
            raise RuntimeError("intentional variant failure")
        return {"standard": {"aggregated": {"accuracy": 1.0}}}

    monkeypatch.setattr(train_multiclass, "_train_one_gnn_model", fake_train_one_gnn_model)

    results = train_multiclass._train_gnn_fold(
        config=config,
        train_idx=np.asarray([0]),
        val_idx=np.asarray([1]),
        test_idx=np.asarray([2]),
        dataset=[],
        fold_dir=str(tmp_path / "fold_0"),
        test_name="fold_0",
        device=torch.device("cpu"),
        task_def={},
        fold_context={},
        class_to_index={0: 0},
        class_labels=[0],
        standardize_features=False,
        verbose=False,
    )

    assert trained_models == ["BasicGCN", "HeteroGCNMean"]
    assert list(results) == ["HeteroGCNMean"]
    failures = (tmp_path / "fold_0" / "gnn" / "gnn_failures.csv").read_text(encoding="utf-8")
    assert "BasicGCN" in failures
    assert "intentional variant failure" in failures
