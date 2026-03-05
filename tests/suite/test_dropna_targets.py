from emotions.suite.run_hci_experiment_suite import _ensure_dropna_targets


def test_dropna_targets_appended_without_duplicates() -> None:
    dataset_cfg = {
        "dropna_columns": [
            "time-rel-seconds",
            "x-avg",
            "y-avg",
            "subject",
            "recording",
        ]
    }

    _ensure_dropna_targets(dataset_cfg, ["emotion-valence", "emotion-arousal", "emotion-valence"])

    cols = dataset_cfg["dropna_columns"]
    assert cols.count("emotion-valence") == 1
    assert cols.count("emotion-arousal") == 1
    assert "subject" in cols


def test_dropna_targets_initializes_default_when_missing() -> None:
    dataset_cfg = {}
    _ensure_dropna_targets(dataset_cfg, ["emotion-id"])
    assert "emotion-id" in dataset_cfg["dropna_columns"]
    assert "time-rel-seconds" in dataset_cfg["dropna_columns"]
