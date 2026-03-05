from __future__ import annotations

from typing import Any, Dict, Set

from emotions.gnn_improvement_experiments.run_gnn_ablation_suite import (
    AROUSAL_EXPERIMENT_ID,
    VALENCE_EXPERIMENT_ID,
    _set_only_binary_valence_arousal,
    build_fixed_wrapper_overrides,
    build_pass1_variants,
)


def test_build_pass1_variants_has_expected_size_and_families() -> None:
    variants = build_pass1_variants()
    variant_ids = [variant.variant_id for variant in variants]
    families = [variant.family for variant in variants]

    assert len(variants) == 20
    assert len(set(variant_ids)) == 20
    assert "baseline_default" in variant_ids
    assert families.count("kt_ks_grid") == 9
    assert families.count("num_layers") == 5
    assert families.count("pooling") == 1
    assert families.count("edge_weights") == 1
    assert families.count("conv_type") == 1
    assert families.count("target_aggregation") == 1
    assert families.count("early_stopping") == 1


def test_set_only_binary_valence_arousal_for_dict_experiments() -> None:
    wrapper_cfg = {
        "experiments": {
            VALENCE_EXPERIMENT_ID: {"enabled": False},
            AROUSAL_EXPERIMENT_ID: {"enabled": False},
            "binary_emotion_control": {"enabled": True},
            "regression_emotion_valence": {"enabled": True},
        }
    }

    _set_only_binary_valence_arousal(wrapper_cfg)
    experiments = wrapper_cfg["experiments"]
    assert experiments[VALENCE_EXPERIMENT_ID]["enabled"] is True
    assert experiments[AROUSAL_EXPERIMENT_ID]["enabled"] is True
    assert experiments["binary_emotion_control"]["enabled"] is False
    assert experiments["regression_emotion_valence"]["enabled"] is False


def _flatten_leaf_paths(payload: Dict[str, Any], prefix: str = "") -> Set[str]:
    """Return dotted leaf-key paths for nested dictionaries."""
    leaf_paths: Set[str] = set()
    for key, value in payload.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            leaf_paths.update(_flatten_leaf_paths(value, current))
        else:
            leaf_paths.add(current)
    return leaf_paths


def test_each_pass1_variant_changes_only_its_intended_factor() -> None:
    variants = build_pass1_variants()
    expected_paths = {
        "kt_ks_grid": {
            "global_overrides.dataset.kt",
            "global_overrides.dataset.ks",
        },
        "pooling": {"global_overrides.gnn.model.pooling"},
        "edge_weights": {"global_overrides.dataset.use_edge_weights"},
        "conv_type": {"global_overrides.gnn.model.conv_type"},
        "target_aggregation": {"global_overrides.dataset.target_aggregation"},
        "early_stopping": {
            "global_overrides.gnn.training.early_stopping_enabled",
            "global_overrides.gnn.training.early_stopping_patience",
            "global_overrides.gnn.training.early_stopping_min_delta",
            "global_overrides.gnn.training.early_stopping_restore_best",
        },
        "num_layers": {"global_overrides.gnn.model.num_layers"},
    }

    for variant in variants:
        paths = _flatten_leaf_paths(variant.overrides)
        if variant.family == "baseline":
            assert paths == set()
            continue
        assert variant.family in expected_paths
        assert paths == expected_paths[variant.family]


def test_fixed_overrides_lock_baseline_defaults() -> None:
    fixed = build_fixed_wrapper_overrides(seed=42)
    dataset = fixed["global_overrides"]["dataset"]
    training = fixed["global_overrides"]["gnn"]["training"]
    cv = fixed["global_overrides"]["cross_validation"]

    assert cv["strategies"] == ["recording_kfold"]
    assert cv["n_splits"] == 3
    assert cv["val_size"] == 1
    assert cv["random_state"] == 42

    assert dataset["kt"] == 2
    assert dataset["ks"] == 2
    assert dataset["use_edge_weights"] is False
    assert dataset["target_aggregation"] == "mean"

    assert training["num_epochs"] == 20
    assert training["num_workers"] == 0
    assert training["persistent_workers"] is False
    assert training["use_torch_compile"] is False
    assert training["early_stopping_enabled"] is False
