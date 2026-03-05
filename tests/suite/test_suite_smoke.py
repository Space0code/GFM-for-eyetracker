from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from emotions.suite.run_hci_experiment_suite import run_suite


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _build_base_binary_config(path: Path) -> None:
    _write_yaml(
        path,
        {
            "run_experiments": {"baselines": True, "gnn": True},
            "binary_task": {"target_column": "emotion-valence", "threshold": 0.5, "decision_threshold": 0.5},
            "dataset": {
                "data_filepath": "placeholder.csv",
                "recursive": True,
                "window_length": 1,
                "window_overlap": 0,
                "min_samples_per_window": 1,
                "kt": 1,
                "ks": 1,
                "use_edge_weights": False,
                "tau": 0.05,
                "use_cache": False,
                "target_aggregation": "mean",
                "feature_columns": ["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"],
                "dropna_columns": [
                    "time-rel-seconds",
                    "x-avg",
                    "y-avg",
                    "pupil-size-left-avg",
                    "pupil-size-right-avg",
                    "subject",
                    "recording",
                ],
                "experiment_type_column": "experiment-type",
            },
            "cross_validation": {"strategies": ["recording_loo"], "val_size": 1, "random_state": 42},
            "baselines": {"models": ["Mean"], "hyperparameters": {"Mean": {}}},
            "gnn": {
                "model": {"in_channels": 4, "hidden_channels": 8},
                "training": {"num_epochs": 1, "batch_size": 2, "learning_rate": 0.001, "device": "cpu"},
            },
            "metrics": ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "auc"],
            "logging": {"results_dir": "results/binary", "verbose": False},
        },
    )


def _build_base_multiclass_config(path: Path) -> None:
    _write_yaml(
        path,
        {
            "run_experiments": {"baselines": True, "gnn": True},
            "multiclass_task": {"task_name": "emotion-id", "target_column": "emotion-id"},
            "dataset": {
                "data_filepath": "placeholder.csv",
                "recursive": True,
                "window_length": 1,
                "window_overlap": 0,
                "min_samples_per_window": 1,
                "kt": 1,
                "ks": 1,
                "use_edge_weights": False,
                "tau": 0.05,
                "use_cache": False,
                "target_aggregation": "mean",
                "feature_columns": ["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"],
                "dropna_columns": [
                    "time-rel-seconds",
                    "x-avg",
                    "y-avg",
                    "pupil-size-left-avg",
                    "pupil-size-right-avg",
                    "emotion-id",
                    "subject",
                    "recording",
                ],
                "experiment_type_column": "experiment-type",
            },
            "cross_validation": {"strategies": ["recording_loo"], "val_size": 1, "random_state": 42},
            "baselines": {"models": ["Mean"], "hyperparameters": {"Mean": {}}},
            "gnn": {
                "model": {"in_channels": 4, "hidden_channels": 8},
                "training": {"num_epochs": 1, "batch_size": 2, "learning_rate": 0.001, "device": "cpu"},
            },
            "metrics": ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr"],
            "logging": {"results_dir": "results/multiclass", "verbose": False},
        },
    )


def _build_base_regression_config(path: Path) -> None:
    _write_yaml(
        path,
        {
            "run_experiments": {"baselines": True, "gnn": True},
            "regression_task": {"target_column": "emotion-valence"},
            "dataset": {
                "data_filepath": "placeholder.csv",
                "recursive": True,
                "window_length": 1,
                "window_overlap": 0,
                "min_samples_per_window": 1,
                "kt": 1,
                "ks": 1,
                "use_edge_weights": False,
                "tau": 0.05,
                "use_cache": False,
                "target_aggregation": "mean",
                "feature_columns": ["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"],
                "dropna_columns": [
                    "time-rel-seconds",
                    "x-avg",
                    "y-avg",
                    "pupil-size-left-avg",
                    "pupil-size-right-avg",
                    "emotion-valence",
                    "subject",
                    "recording",
                ],
                "experiment_type_column": "experiment-type",
            },
            "cross_validation": {"strategies": ["recording_loo"], "val_size": 1, "random_state": 42},
            "baselines": {"models": ["Mean"], "hyperparameters": {"Mean": {}}},
            "gnn": {
                "model": {"in_channels": 4, "hidden_channels": 8},
                "training": {"num_epochs": 1, "batch_size": 2, "learning_rate": 0.001, "device": "cpu"},
            },
            "metrics": ["mae", "ccc", "spearman"],
            "logging": {"results_dir": "results/regression", "verbose": False},
        },
    )


def _create_scope_csv(path: Path, scope: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if scope == "emotion-elicitation":
        df = pd.DataFrame(
            {
                "time-rel-seconds": [0.0, 0.4, 0.8, 1.2],
                "x-avg": [1.0, 1.2, 1.1, 1.3],
                "y-avg": [2.0, 2.1, 2.2, 2.1],
                "pupil-size-left-avg": [3.0, 3.1, 3.2, 3.3],
                "pupil-size-right-avg": [3.2, 3.3, 3.4, 3.5],
                "subject": ["P1", "P1", "P1", "P1"],
                "recording": ["emo_r1", "emo_r1", "emo_r1", "emo_r1"],
                "experiment-type": ["emotion-elicitation"] * 4,
                "emotion-id": [4, 4, 4, 4],
                "emotion-valence": [6, 6, 6, 6],
                "emotion-arousal": [7, 7, 7, 7],
                "emotion-control": [5, 5, 5, 5],
                "emotion-predictability": [4, 4, 4, 4],
                "emotion-derivation-status": ["ok"] * 4,
            }
        )
    else:
        df = pd.DataFrame(
            {
                "time-rel-seconds": [0.0, 0.4, 0.8, 1.2],
                "x-avg": [1.0, 1.2, 1.1, 1.3],
                "y-avg": [2.0, 2.1, 2.2, 2.1],
                "pupil-size-left-avg": [3.0, 3.1, 3.2, 3.3],
                "pupil-size-right-avg": [3.2, 3.3, 3.4, 3.5],
                "subject": ["P1", "P1", "P1", "P1"],
                "recording": [f"{scope}_r1"] * 4,
                "experiment-type": [scope] * 4,
                "tag-valid": [0, 1, 0, 1],
                "tag-agree": [1, 1, 0, 1],
                "tag-derivation-status": ["ok"] * 4,
            }
        )
    df.to_csv(path, index=False)


def test_suite_smoke_runs_selected_experiments(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "processed" / "hci-tagging"
    for scope in ["emotion-elicitation", "image-tagging-1", "image-tagging-2", "video-tagging"]:
        _create_scope_csv(source_root / scope / f"{scope}_sample.csv", scope)

    config_root = tmp_path / "configs"
    binary_base = config_root / "binary.yaml"
    multiclass_base = config_root / "multiclass.yaml"
    regression_base = config_root / "regression.yaml"
    _build_base_binary_config(binary_base)
    _build_base_multiclass_config(multiclass_base)
    _build_base_regression_config(regression_base)

    def fake_train_dispatch(task_type: str, config_path: str) -> str:
        run_dir = Path(config_path).parent / "fake_trainer_run"
        strategy_dir = run_dir / "recording_loo"
        strategy_dir.mkdir(parents=True, exist_ok=True)

        if task_type in {"binary", "multiclass"}:
            if task_type == "binary":
                summary = pd.DataFrame(
                    [
                        {
                            "model": "GNN",
                            "metric_type": "aggregated",
                            "accuracy": 0.8,
                            "balanced_accuracy": 0.75,
                            "f1": 0.7,
                            "precision": 0.72,
                            "recall": 0.69,
                            "auc": 0.81,
                        }
                    ]
                )
            else:
                summary = pd.DataFrame(
                    [
                        {
                            "model": "GNN",
                            "metric_type": "aggregated",
                            "accuracy": 0.65,
                            "balanced_accuracy": 0.6,
                            "macro_f1": 0.58,
                            "weighted_f1": 0.62,
                            "macro_auc_ovr": 0.7,
                            "weighted_auc_ovr": 0.72,
                        }
                    ]
                )
        else:
            summary = pd.DataFrame(
                [
                    {
                        "model": "GNN",
                        "metric_type": "aggregated",
                        "mae": 0.9,
                        "ccc": 0.5,
                        "spearman": 0.55,
                    }
                ]
            )

        summary.to_csv(strategy_dir / "summary.csv", index=False)
        return str(run_dir)

    monkeypatch.setattr(
        "emotions.suite.run_hci_experiment_suite._train_dispatch",
        fake_train_dispatch,
    )

    wrapper_config = {
        "suite": {
            "results_dir": str(tmp_path / "suite_results"),
            "seed": 42,
            "source_data_root": str(source_root),
            "base_configs": {
                "binary": str(binary_base),
                "multiclass": str(multiclass_base),
                "regression": str(regression_base),
            },
        },
        "global_overrides": {
            "cross_validation": {"strategies": ["recording_loo"], "val_size": 1, "random_state": 42},
            "dataset": {"target_aggregation": "mean"},
        },
        "experiments": {
            "binary_valence": {
                "enabled": True,
                "task_type": "binary",
                "scope": "emotion-elicitation",
                "target_column": "emotion-valence",
                "threshold": 0.5,
            },
            "binary_tag_agree_pooled": {
                "enabled": True,
                "task_type": "binary",
                "scope": "pooled-tagging",
                "target_column": "tag-agree",
                "threshold": 0.5,
            },
            "multiclass_emotion_id": {
                "enabled": True,
                "task_type": "multiclass",
                "scope": "emotion-elicitation",
                "task_name": "emotion-id",
                "target_column": "emotion-id",
            },
            "regression_valence": {
                "enabled": True,
                "task_type": "regression",
                "scope": "emotion-elicitation",
                "target_column": "emotion-valence",
            },
        },
    }

    wrapper_path = tmp_path / "wrapper.yaml"
    _write_yaml(wrapper_path, wrapper_config)

    suite_dir = Path(run_suite(str(wrapper_path)))

    registry_path = suite_dir / "suite_experiment_registry.csv"
    assert registry_path.exists()

    registry = pd.read_csv(registry_path)
    assert len(registry) == 4
    assert set(registry["status"].tolist()) == {"success"}

    for _, row in registry.iterrows():
        assert Path(row["snapshot_csv"]).exists()
        assert Path(row["eda_summary_path"]).exists()
        assert Path(row["resolved_config_path"]).exists()

    assert (suite_dir / "classification_master_comparison.csv").exists()
    assert (suite_dir / "regression_master_comparison.csv").exists()
