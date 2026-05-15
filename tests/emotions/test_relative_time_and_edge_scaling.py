from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.data import SpacioTemporalDataset
from data.hci_signals import TIME_WINDOW_NORMALIZED_COLUMN
from emotions.common.edge_scaling import apply_edge_feature_scalers, fit_edge_feature_scalers
from emotions.train_baseline import build_tabular_samples


def _frame(offset: float, subject: str, recording: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time-rel-seconds": [offset, offset + 0.1, offset + 0.2, offset + 0.3],
            "x-avg": [1.0, 2.0, 4.0, 8.0],
            "y-avg": [2.0, 3.0, 5.0, 9.0],
            "pupil-size-left-avg": [3.0, 3.1, 3.2, 3.3],
            "pupil-size-right-avg": [3.4, 3.3, 3.2, 3.1],
            "distance-left": [60.0, 61.0, 62.0, 63.0],
            "distance-right": [64.0, 65.0, 66.0, 67.0],
            "fixation-index": [1, 1, 2, 2],
            "fixation-duration": [100.0, 100.0, 200.0, 200.0],
            "fixation": [True, True, True, True],
            "subject": [subject] * 4,
            "recording": [recording] * 4,
            "emotion-id": [1] * 4,
            "experiment-type": ["emotion-elicitation"] * 4,
            "emotion-derivation-status": ["ok"] * 4,
        }
    )


def test_tabular_samples_include_window_local_normalized_time_mean(tmp_path: Path) -> None:
    data_csv = tmp_path / "hci.csv"
    _frame(offset=20.0, subject="P1", recording="r1").to_csv(data_csv, index=False)

    samples = build_tabular_samples(
        data_filepath=str(data_csv),
        window_length=1,
        window_overlap=0.0,
        min_samples_per_window=2,
        feature_columns=[
            "x-avg",
            "y-avg",
            "pupil-size-left-avg",
            "pupil-size-right-avg",
            TIME_WINDOW_NORMALIZED_COLUMN,
        ],
        target_columns=["emotion-id"],
        target_aggregation="mean",
        dropna_columns=[
            "time-rel-seconds",
            "x-avg",
            "y-avg",
            "pupil-size-left-avg",
            "pupil-size-right-avg",
            "emotion-id",
        ],
    )

    assert len(samples) == 1
    assert np.isclose(
        samples[0].features[f"{TIME_WINDOW_NORMALIZED_COLUMN}_mean"],
        np.mean([0.0, 0.1, 0.2, 0.3]),
    )


def test_edge_feature_scalers_are_train_only_relation_specific_and_keep_direction(tmp_path: Path) -> None:
    data_csv = tmp_path / "hci_edges.csv"
    pd.concat(
        [
            _frame(offset=10.0, subject="P1", recording="r1"),
            _frame(offset=30.0, subject="P2", recording="r2"),
        ],
        ignore_index=True,
    ).to_csv(data_csv, index=False)

    dataset = SpacioTemporalDataset(
        data_filepath=str(data_csv),
        recursive=False,
        kt=1,
        ks=1,
        window_length=1,
        window_overlap=0.0,
        min_samples_per_window=2,
        use_edge_weights=True,
        graph_version="v2",
        edge_weight_mode="learned_signed",
        use_cache=False,
        target_columns=["emotion-id"],
        dropna_columns=[
            "time-rel-seconds",
            "x-avg",
            "y-avg",
            "pupil-size-left-avg",
            "pupil-size-right-avg",
            "emotion-id",
        ],
        feature_columns=[
            "x-avg",
            "y-avg",
            "pupil-size-left-avg",
            "pupil-size-right-avg",
        ],
        use_relative_time=True,
        use_distance_avg=True,
        use_fixation_duration=True,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )

    scalers = fit_edge_feature_scalers(dataset=dataset, train_idx=np.asarray([0]))
    assert set(scalers) == {"temporal", "spatial", "fixation"}
    assert scalers["temporal"] is not scalers["spatial"]
    assert scalers["fixation"] is not scalers["spatial"]

    graph = dataset[1].clone()
    original_forward = graph["node", "temporal_forward", "node"].edge_attr.clone()
    original_backward = graph["node", "temporal_backward", "node"].edge_attr.clone()
    apply_edge_feature_scalers(graph, scalers)

    scaled_forward = graph["node", "temporal_forward", "node"].edge_attr
    scaled_backward = graph["node", "temporal_backward", "node"].edge_attr

    assert torch.allclose(scaled_forward[:, -1], original_forward[:, -1])
    assert torch.allclose(scaled_backward[:, -1], original_backward[:, -1])
    assert not torch.allclose(scaled_forward[:, :-1], original_forward[:, :-1])
