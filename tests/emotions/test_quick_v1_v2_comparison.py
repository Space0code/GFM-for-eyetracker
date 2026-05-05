from __future__ import annotations

from argparse import Namespace

from emotions.gnn_improvement_experiments.run_quick_v1_v2_comparison import (
    AROUSAL_EXPERIMENT_ID,
    build_fixed_overrides,
    build_variant,
    _build_payload,
)


def _args() -> Namespace:
    return Namespace(
        output_root="results/test_quick",
        seed=42,
        cv_strategy="recording_kfold",
        n_splits=3,
        val_size=1,
        num_epochs=2,
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


def test_quick_payload_enables_only_table6_arousal() -> None:
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
    assert payload["experiments"]["multiclass_table6_valence_3class"]["enabled"] is False
    assert payload["global_overrides"]["baselines"]["models"] == ["LightGBM"]
    assert payload["global_overrides"]["run_experiments"] == {"baselines": True, "gnn": False}
