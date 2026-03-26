from emotions.suite.run_hci_experiment_suite import _apply_scope_defaults


def test_scope_defaults_preserve_explicit_dataset_overrides() -> None:
    dataset_cfg = {
        "allowed_experiment_types": ["custom-scope"],
        "label_quality_column": "custom-quality-col",
        "allowed_label_quality_values": ["keep"],
    }

    _apply_scope_defaults(dataset_cfg=dataset_cfg, scope="emotion-elicitation")

    assert dataset_cfg["allowed_experiment_types"] == ["custom-scope"]
    assert dataset_cfg["label_quality_column"] == "custom-quality-col"
    assert dataset_cfg["allowed_label_quality_values"] == ["keep"]


def test_scope_defaults_fill_missing_keys() -> None:
    dataset_cfg = {}

    _apply_scope_defaults(dataset_cfg=dataset_cfg, scope="image-tagging-1")

    assert dataset_cfg["allowed_experiment_types"] == ["image-tagging-1"]
    assert dataset_cfg["label_quality_column"] == "tag-derivation-status"
    assert dataset_cfg["allowed_label_quality_values"] == ["ok"]
