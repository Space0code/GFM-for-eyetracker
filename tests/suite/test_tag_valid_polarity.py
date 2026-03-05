import pandas as pd

from emotions.suite.eda_summary import _derive_final_labels


def test_tag_valid_positive_class_remains_raw_one_with_threshold_half() -> None:
    snapshot_df = pd.DataFrame(
        {
            "tag-valid": [0, 1, 0, 1],
            "time-rel-seconds": [0.0, 0.1, 0.2, 0.3],
            "x-avg": [1.0, 1.0, 1.0, 1.0],
            "y-avg": [1.0, 1.0, 1.0, 1.0],
            "pupil-size-left-avg": [1.0, 1.0, 1.0, 1.0],
            "pupil-size-right-avg": [1.0, 1.0, 1.0, 1.0],
            "subject": ["P1", "P1", "P1", "P1"],
            "recording": ["R1", "R1", "R1", "R1"],
        }
    )

    labels, details = _derive_final_labels(
        snapshot_df=snapshot_df,
        task_type="binary",
        experiment_cfg={"target_column": "tag-valid", "threshold": 0.5},
    )

    assert details["threshold_value"] == 0.5
    assert labels.tolist() == [0.0, 1.0, 0.0, 1.0]
