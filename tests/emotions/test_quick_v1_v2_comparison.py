from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import numpy as np

from emotions.gnn_improvement_experiments.run_quick_v1_v2_comparison import (
    AROUSAL_EXPERIMENT_ID,
    build_fixed_overrides,
    build_quick_runs,
    build_variant,
    _build_payload,
    _ordered_models,
    _parse_cv_strategies,
    _parse_models,
    _save_combined_confusion_matrices,
    _save_group_model_ranking,
    _save_label_distribution_outputs,
)


def _args() -> Namespace:
    return Namespace(
        output_root="results/test_quick",
        seed=None,
        cv_strategy=None,
        n_splits=None,
        val_size=None,
        num_epochs=None,
        use_torch_compile=False,
    )


def test_quick_variants_set_expected_gnn_versions() -> None:
    v1 = build_variant("GNN_v1")
    v2 = build_variant("GNN_v2")

    assert v1.overrides["global_overrides"]["dataset"]["graph_version"] == "v1"
    assert v1.overrides["global_overrides"]["dataset"]["edge_weight_mode"] == "handcrafted"
    assert v1.overrides["global_overrides"]["gnn"]["model"]["model_version"] == "v1"

    assert v2.overrides["global_overrides"]["dataset"]["graph_version"] == "v2"
    assert v2.overrides["global_overrides"]["dataset"]["edge_weight_mode"] == "learned_signed"
    assert v2.overrides["global_overrides"]["gnn"]["model"]["model_version"] == "v2"


def test_quick_payload_enables_table6_arousal_and_valence() -> None:
    base_cfg = {
        "experiments": {
            AROUSAL_EXPERIMENT_ID: {"enabled": False},
            "multiclass_table6_valence_3class": {"enabled": True},
        }
    }
    payload = _build_payload(
        base_cfg=base_cfg,
        fixed_overrides=build_fixed_overrides(_args()),
        variant=build_variant("LightGBM"),
    )

    assert payload["experiments"][AROUSAL_EXPERIMENT_ID]["enabled"] is True
    assert payload["experiments"]["multiclass_table6_valence_3class"]["enabled"] is True
    assert payload["global_overrides"]["baselines"]["models"] == ["LightGBM"]
    assert payload["global_overrides"]["run_experiments"] == {"baselines": True, "gnn": False}


def test_quick_variants_support_random_and_majority() -> None:
    random_variant = build_variant("Random")
    majority_variant = build_variant("Majority")

    assert random_variant.overrides["global_overrides"]["baselines"]["models"] == ["Random"]
    assert majority_variant.overrides["global_overrides"]["baselines"]["models"] == ["Majority"]
    assert random_variant.summary_model_name == "Random"
    assert majority_variant.summary_model_name == "Majority"


def test_quick_model_parser_accepts_common_aliases() -> None:
    parsed = _parse_models("random,majority,gnn1,gnn2,lgbm")

    assert parsed == ["Random", "Majority", "GNN_v1", "GNN_v2", "LightGBM"]


def test_quick_cv_parser_accepts_multiple_strategies() -> None:
    parsed = _parse_cv_strategies("subject_loo,recording_loo,subject_kfold,recording_kfold")

    assert parsed == ["subject_loo", "recording_loo", "subject_kfold", "recording_kfold"]


def test_quick_plot_model_order_keeps_sanity_baselines_first() -> None:
    ordered = _ordered_models(["LightGBM", "GNN_v2", "Random", "MLP", "Majority", "SVM", "GNN_v1"])

    assert ordered == ["Random", "Majority", "GNN_v1", "GNN_v2", "MLP", "LightGBM", "SVM"]


def test_quick_runs_group_baselines_into_one_suite_invocation() -> None:
    runs = build_quick_runs(["Random", "Majority", "GNN_v1", "GNN_v2", "LightGBM"])

    assert [run.run_name for run in runs] == ["Baselines", "GNN_v1", "GNN_v2"]
    assert runs[0].model_names == ["Random", "Majority", "LightGBM"]
    assert runs[0].overrides["global_overrides"]["baselines"]["models"] == [
        "Random",
        "Majority",
        "LightGBM",
    ]


def test_fixed_overrides_can_target_command_output_dir() -> None:
    output_dir = Path("results/test_quick/command_timestamp")
    overrides = build_fixed_overrides(_args(), run_output_dir=output_dir)

    assert overrides["suite"]["results_dir"] == str(output_dir / "model_runs")
    assert overrides["global_overrides"]["gnn"]["training"]["use_torch_compile"] is False


def test_fixed_overrides_only_include_explicit_cli_overrides() -> None:
    args = Namespace(
        output_root="results/test_quick",
        seed=7,
        cv_strategy="subject_kfold,recording_kfold",
        n_splits=2,
        val_size=1,
        num_epochs=3,
        use_torch_compile=False,
    )
    overrides = build_fixed_overrides(args)

    assert overrides["suite"]["seed"] == 7
    assert overrides["global_overrides"]["cross_validation"] == {
        "random_state": 7,
        "strategies": ["subject_kfold", "recording_kfold"],
        "n_splits": 2,
        "val_size": 1,
    }
    assert overrides["global_overrides"]["gnn"]["training"]["num_epochs"] == 3
    assert overrides["global_overrides"]["gnn"]["training"]["use_torch_compile"] is False


def test_fixed_overrides_can_explicitly_enable_torch_compile() -> None:
    args = _args()
    args.use_torch_compile = True

    overrides = build_fixed_overrides(args)

    assert overrides["global_overrides"]["gnn"]["training"]["use_torch_compile"] is True


def test_group_model_ranking_plot_is_written(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "GNN_v2",
                "status": "success",
                "accuracy": 0.39,
                "balanced_accuracy": 0.41,
                "macro_f1": 0.37,
                "weighted_f1": 0.38,
                "auc": 0.52,
            },
            {
                "model": "Random",
                "status": "success",
                "accuracy": 0.27,
                "balanced_accuracy": 0.29,
                "macro_f1": 0.26,
                "weighted_f1": 0.27,
                "auc": 0.50,
            },
            {
                "model": "Majority",
                "status": "success",
                "accuracy": 0.31,
                "balanced_accuracy": 0.33,
                "macro_f1": 0.25,
                "weighted_f1": 0.29,
                "auc": 0.50,
            },
            {
                "model": "GNN_v1",
                "status": "success",
                "accuracy": 0.34,
                "balanced_accuracy": 0.35,
                "macro_f1": 0.31,
                "weighted_f1": 0.32,
                "auc": 0.49,
            },
            {
                "model": "LightGBM",
                "status": "success",
                "accuracy": 0.37,
                "balanced_accuracy": 0.38,
                "macro_f1": 0.35,
                "weighted_f1": 0.36,
                "auc": 0.51,
            },
        ]
    )

    output_path = _save_group_model_ranking(summary=summary, output_dir=tmp_path)

    assert output_path == tmp_path / "plots" / "classification_group_model_ranking.png"
    assert output_path.exists()


def test_combined_confusion_matrices_include_random_and_majority(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "subject_loo" / "fold_0"
    (strategy_dir / "baselines" / "Random").mkdir(parents=True)
    (strategy_dir / "baselines" / "Majority").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)

    targets = np.asarray([0, 1, 2, 0])
    random_predictions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    majority_predictions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    for model_name, predictions in {
        "Random": random_predictions,
        "Majority": majority_predictions,
    }.items():
        model_dir = strategy_dir / "baselines" / model_name
        np.save(model_dir / "test_targets.npy", targets)
        np.save(model_dir / "test_predictions.npy", predictions)

    rows = [
        {
            "model": "Random",
            "experiment_id": AROUSAL_EXPERIMENT_ID,
            "cv_strategy": "subject_loo",
            "status": "success",
            "suite_run_dir": str(suite_run_dir),
            "summary_model_name": "Random",
        },
        {
            "model": "Majority",
            "experiment_id": AROUSAL_EXPERIMENT_ID,
            "cv_strategy": "subject_loo",
            "status": "success",
            "suite_run_dir": str(suite_run_dir),
            "summary_model_name": "Majority",
        },
    ]

    output_path = _save_combined_confusion_matrices(
        rows=rows,
        output_dir=tmp_path,
        experiment_id=AROUSAL_EXPERIMENT_ID,
        cv_strategy="subject_loo",
        use_strategy_suffix=False,
    )

    assert output_path == tmp_path / "figures" / "confusion_matrices_table6_arousal.png"
    assert output_path.exists()


def test_combined_confusion_matrices_can_use_strategy_suffix(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "recording_loo" / "fold_0"
    model_dir = strategy_dir / "baselines" / "Random"
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    np.save(model_dir / "test_targets.npy", np.asarray([0, 1]))
    np.save(model_dir / "test_predictions.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]]))

    output_path = _save_combined_confusion_matrices(
        rows=[
            {
                "model": "Random",
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "cv_strategy": "recording_loo",
                "status": "success",
                "suite_run_dir": str(suite_run_dir),
                "summary_model_name": "Random",
            }
        ],
        output_dir=tmp_path,
        experiment_id=AROUSAL_EXPERIMENT_ID,
        cv_strategy="recording_loo",
        use_strategy_suffix=True,
    )

    assert output_path == tmp_path / "figures" / "confusion_matrices_table6_arousal_recording_loo.png"
    assert output_path.exists()


def test_label_distribution_outputs_include_tables_and_plots(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "subject_kfold" / "fold_0"
    model_dir = strategy_dir / "baselines" / "Random"
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    np.save(model_dir / "test_targets.npy", np.asarray([0, 0, 1, 2]))

    paths = _save_label_distribution_outputs(
        rows=[
            {
                "model": "Random",
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "cv_strategy": "subject_kfold",
                "status": "success",
                "suite_run_dir": str(suite_run_dir),
                "summary_model_name": "Random",
            }
        ],
        output_dir=tmp_path,
    )

    expected_paths = {
        tmp_path / "tables" / "label_distribution_by_fold.csv",
        tmp_path / "tables" / "label_distribution_aggregate.csv",
        tmp_path / "plots" / "label_distribution_counts.png",
        tmp_path / "plots" / "label_distribution_proportions.png",
    }
    assert expected_paths.issubset(set(paths))
    for path in expected_paths:
        assert path.exists()

    aggregate = pd.read_csv(tmp_path / "tables" / "label_distribution_aggregate.csv")
    assert aggregate["count"].tolist() == [2, 1, 1]
    assert np.allclose(aggregate["proportion"].to_numpy(), np.asarray([0.5, 0.25, 0.25]))
