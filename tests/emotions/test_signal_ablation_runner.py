from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from emotions.gnn_improvement_experiments.ablations.run_signal_ablation_suite import (
    DEFAULT_VARIANTS,
    build_command_overrides,
    build_signal_ablation_variant,
    build_variant_payload,
    _suite_registry_status,
)


def _args() -> Namespace:
    return Namespace(
        seed=42,
        n_splits=5,
        val_size=1,
        num_epochs=None,
    )


def test_signal_ablation_variants_are_gnn_v2_only() -> None:
    for variant_name in DEFAULT_VARIANTS:
        variant = build_signal_ablation_variant(variant_name)
        global_overrides = variant.overrides["global_overrides"]

        assert global_overrides["run_experiments"] == {"baselines": False, "gnn": True}
        assert global_overrides["dataset"]["graph_version"] == "v2"
        assert global_overrides["dataset"]["edge_weight_mode"] == "learned_signed"
        assert global_overrides["gnn"]["model"]["model_version"] == "v2"
        assert global_overrides["gnn"]["model"]["use_fixation_edges"] is True


def test_signal_ablation_variant_flags_remove_expected_sources() -> None:
    without_temporal = build_signal_ablation_variant("without_temporal").overrides["global_overrides"]["dataset"]
    assert without_temporal["use_temporal_node_feature"] is False
    assert without_temporal["use_temporal_edge_features"] is False
    assert without_temporal["use_temporal_edges"] is False

    without_gaze = build_signal_ablation_variant("without_spatial_gaze").overrides["global_overrides"]["dataset"]
    assert without_gaze["use_gaze_node_features"] is False
    assert without_gaze["use_gaze_edge_features"] is False
    assert without_gaze["use_spatial_edges"] is False
    assert "x-avg" not in without_gaze["dropna_columns"]
    assert "y-avg" not in without_gaze["dropna_columns"]

    without_fixation = build_signal_ablation_variant("without_fixation").overrides["global_overrides"]["dataset"]
    assert without_fixation["use_fixation_node_feature"] is False
    assert without_fixation["use_fixation_edges"] is False

    without_pupil = build_signal_ablation_variant("without_pupil").overrides["global_overrides"]["dataset"]
    assert without_pupil["use_pupil_node_features"] is False
    assert "pupil-size-left-avg" not in without_pupil["dropna_columns"]
    assert "pupil-size-right-avg" not in without_pupil["dropna_columns"]

    without_distance = build_signal_ablation_variant("without_screen_distance").overrides["global_overrides"]["dataset"]
    assert without_distance["use_screen_distance_node_feature"] is False
    assert without_distance["use_screen_distance_edge_feature"] is False
    assert without_distance["use_delta_distance_edge_feature"] is False


def test_signal_ablation_payload_forces_subject_kfold_five_splits() -> None:
    base_cfg = {
        "suite": {"results_dir": "results/base"},
        "global_overrides": {
            "cross_validation": {
                "strategies": ["subject_loo", "recording_loo"],
                "n_splits": 99,
                "val_size": 2,
            }
        },
    }
    command_overrides = build_command_overrides(_args(), run_output_dir=Path("results/test_signal_ablation/run"))
    payload = build_variant_payload(
        base_cfg=base_cfg,
        command_overrides=command_overrides,
        variant=build_signal_ablation_variant("baseline_full"),
        variant_output_dir=Path("results/test_signal_ablation/run/baseline_full"),
    )

    assert payload["global_overrides"]["cross_validation"] == {
        "strategies": ["subject_kfold"],
        "n_splits": 5,
        "val_size": 1,
        "random_state": 42,
    }
    assert payload["global_overrides"]["run_experiments"] == {"baselines": False, "gnn": True}
    assert payload["suite"]["results_dir"] == "results/test_signal_ablation/run/baseline_full/suite"


def test_suite_registry_status_reports_failed_experiments(tmp_path: Path) -> None:
    registry = tmp_path / "suite_experiment_registry.csv"
    registry.write_text(
        "experiment_id,status\n"
        "multiclass_table6_valence_3class,success\n"
        "multiclass_table6_arousal_3class,failed\n",
        encoding="utf-8",
    )

    status, error = _suite_registry_status(tmp_path)

    assert status == "failed"
    assert "multiclass_table6_arousal_3class" in error


def test_suite_registry_status_accepts_all_success(tmp_path: Path) -> None:
    registry = tmp_path / "suite_experiment_registry.csv"
    registry.write_text(
        "experiment_id,status\n"
        "multiclass_table6_valence_3class,success\n",
        encoding="utf-8",
    )

    status, error = _suite_registry_status(tmp_path)

    assert status == "success"
    assert error == ""
