from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from data.data import SpacioTemporalDataset, _fixation_dilation_offsets
from data.data_preprocess import preprocess_file
from data.hci_signals import TIME_WINDOW_NORMALIZED_COLUMN
from emotions.model import SpatioTemporalHeteroGNN


def _raw_hci_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time-rel-seconds": [10.0, 10.1, 10.2, 10.3, 10.4],
            "x-avg": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y-avg": [1.0, 1.5, 2.0, 2.5, 3.0],
            "confidence-gaze-left": [1, 1, 1, 1, 1],
            "confidence-gaze-right": [1, 1, 1, 1, 1],
            "pupil-size-left-avg": [3.0, 3.1, 3.2, 3.3, 3.4],
            "pupil-size-right-avg": [3.4, 3.3, 3.2, 3.1, 3.0],
            "distance-left": [60.0, 62.0, float("nan"), 66.0, 68.0],
            "distance-right": [64.0, 66.0, 70.0, 70.0, 72.0],
            "fixation-index": [10, 10, 11, 11, pd.NA],
            "fixation-duration": [100.0, 100.0, 200.0, 200.0, pd.NA],
            "fixation": [True, True, True, True, False],
            "subject": ["P1"] * 5,
            "recording": ["r1"] * 5,
            "emotion-id": [0] * 5,
            "emotion-derivation-status": ["ok"] * 5,
            "experiment-type": ["emotion-elicitation"] * 5,
        }
    )


def _single_fixation_frame(n: int) -> pd.DataFrame:
    """Create one contiguous fixation run for graph-construction tests."""
    return pd.DataFrame(
        {
            "time-rel-seconds": [idx * 0.1 for idx in range(n)],
            "x-avg": [float(idx) for idx in range(n)],
            "y-avg": [float(idx % 5) for idx in range(n)],
            "confidence-gaze-left": [1] * n,
            "confidence-gaze-right": [1] * n,
            "pupil-size-left-avg": [3.0 + idx * 0.01 for idx in range(n)],
            "pupil-size-right-avg": [3.2 + idx * 0.01 for idx in range(n)],
            "distance-left": [60.0] * n,
            "distance-right": [62.0] * n,
            "fixation-index": [10] * n,
            "fixation-duration": [300.0] * n,
            "fixation": [True] * n,
            "subject": ["P1"] * n,
            "recording": ["r1"] * n,
            "emotion-id": [0] * n,
            "emotion-derivation-status": ["ok"] * n,
            "experiment-type": ["emotion-elicitation"] * n,
        }
    )


def test_hci_preprocess_preserves_and_derives_optional_signals(tmp_path: Path) -> None:
    source_csv = tmp_path / "P1_sample.csv"
    dest_dir = tmp_path / "processed"
    _raw_hci_frame().to_csv(source_csv, index=False)

    preprocess_file(
        dir_name="hci-tagging/emotion-elicitation",
        filename=source_csv.name,
        file_path=str(source_csv),
        dest_data_dir=str(dest_dir),
    )

    out = pd.read_csv(dest_dir / source_csv.name)
    for column in [
        "distance-left",
        "distance-right",
        "distance-avg",
        "fixation-index",
        "fixation-duration",
        "fixation",
    ]:
        assert column in out.columns
    assert out.loc[0, "distance-avg"] == 62.0
    assert out.loc[2, "distance-avg"] == 70.0
    assert out.loc[4, "fixation-duration"] == 0.0


def test_dataset_builds_optional_node_and_edge_features_and_fixation_edges(tmp_path: Path) -> None:
    data_csv = tmp_path / "hci.csv"
    _raw_hci_frame().to_csv(data_csv, index=False)

    dataset = SpacioTemporalDataset(
        data_filepath=str(data_csv),
        recursive=False,
        kt=1,
        ks=1,
        window_length=10,
        window_overlap=0.0,
        min_samples_per_window=2,
        use_edge_weights=True,
        graph_version="v2",
        edge_weight_mode="learned_signed",
        use_cache=False,
        target_columns=["emotion-id"],
        dropna_columns=["time-rel-seconds", "x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg", "emotion-id"],
        use_distance_avg=True,
        use_fixation_duration=True,
        use_relative_time=False,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )

    graph = dataset[0]
    assert graph["node"].x.shape[1] == 6
    assert graph["node", "spatial", "node"].edge_attr.shape[1] == 7
    assert graph["node", "temporal_forward", "node"].edge_attr.shape[1] == 8
    assert graph["node", "temporal_backward", "node"].edge_attr.shape[1] == 8
    assert graph["node", "fixation", "node"].edge_attr.shape[1] == 7

    fixation_edges = set(map(tuple, graph["node", "fixation", "node"].edge_index.t().tolist()))
    assert fixation_edges == {(0, 1), (1, 0), (2, 3), (3, 2)}

    spatial_edges = graph["node", "spatial", "node"].edge_index
    spatial_edge_set = set(map(tuple, spatial_edges.t().tolist()))
    assert spatial_edges.shape[1] == len(spatial_edge_set)


def test_fixation_dilation_offsets_use_half_up_rounding_and_no_self_loops() -> None:
    assert _fixation_dilation_offsets(run_length=30, dilation_k=3) == (1, 11, 21)
    assert _fixation_dilation_offsets(run_length=30, dilation_k=10) == (
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22,
        25,
        28,
    )
    assert _fixation_dilation_offsets(run_length=2, dilation_k=3) == (1,)


def test_dataset_builds_default_dilated_intra_fixation_edges_without_duplicates(tmp_path: Path) -> None:
    data_csv = tmp_path / "hci_single_fixation.csv"
    _single_fixation_frame(30).to_csv(data_csv, index=False)

    dataset = SpacioTemporalDataset(
        data_filepath=str(data_csv),
        recursive=False,
        kt=1,
        ks=1,
        window_length=10,
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
        use_distance_avg=True,
        use_fixation_duration=True,
        use_relative_time=False,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )

    graph = dataset[0]
    edge_index = graph["node", "fixation", "node"].edge_index
    fixation_edges = set(map(tuple, edge_index.t().tolist()))
    expected_node_zero_targets = {1, 11, 21}

    assert all((0, target) in fixation_edges for target in expected_node_zero_targets)
    assert edge_index.shape[1] == 180
    assert edge_index.shape[1] == len(fixation_edges)
    assert not any(source == target for source, target in fixation_edges)


def test_dataset_uses_window_local_normalized_time_for_nodes_and_edges(tmp_path: Path) -> None:
    data_csv = tmp_path / "hci_relative_time.csv"
    _raw_hci_frame().to_csv(data_csv, index=False)

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
        use_relative_time=True,
        use_distance_avg=True,
        use_fixation_duration=True,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )

    graph = dataset[0]
    time_idx = dataset.feature_columns.index(TIME_WINDOW_NORMALIZED_COLUMN)
    node_time = graph["node"].x[:, time_idx]

    assert graph["node"].x.shape[1] == 7
    assert torch.allclose(node_time, torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]))
    assert float(node_time[-1]) < 1.0

    forward_edges = graph["node", "temporal_forward", "node"].edge_index
    first_forward = int(torch.nonzero((forward_edges[0] == 0) & (forward_edges[1] == 1))[0])
    forward_attr = graph["node", "temporal_forward", "node"].edge_attr[first_forward]

    assert torch.allclose(forward_attr[:3], torch.tensor([0.0, 0.1, 0.1]), atol=1e-6)
    assert not torch.any(torch.isclose(forward_attr[:3], torch.tensor(10.0)))


def test_gnn_v2_forward_supports_fixation_relation_and_extended_edge_attrs(tmp_path: Path) -> None:
    temp_csv = tmp_path / "hci_forward.csv"
    _raw_hci_frame().to_csv(temp_csv, index=False)
    dataset = SpacioTemporalDataset(
        data_filepath=str(temp_csv),
        recursive=False,
        kt=1,
        ks=1,
        window_length=10,
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
        use_distance_avg=True,
        use_fixation_duration=True,
        use_relative_time=False,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )
    batch = next(iter(DataLoader([dataset[0]], batch_size=1)))

    model = SpatioTemporalHeteroGNN(
        in_channels=6,
        hidden_channels=8,
        out_channels=1,
        output_scale=1.0,
        use_preprocess_mlp=False,
        use_edge_weights=True,
        conv_type="GCNConv",
        num_layers=2,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out.shape) == (1, 1)


def test_gnn_v2_forward_handles_empty_fixation_relation(tmp_path: Path) -> None:
    frame = _raw_hci_frame()
    frame["fixation-index"] = [10, 11, 12, 13, pd.NA]
    frame["fixation-duration"] = [100.0, 200.0, 300.0, 400.0, pd.NA]
    temp_csv = tmp_path / "hci_empty_fixation.csv"
    frame.to_csv(temp_csv, index=False)

    dataset = SpacioTemporalDataset(
        data_filepath=str(temp_csv),
        recursive=False,
        kt=1,
        ks=1,
        window_length=10,
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
        use_distance_avg=True,
        use_fixation_duration=True,
        use_relative_time=False,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )
    graph = dataset[0]
    assert graph["node", "fixation", "node"].edge_index.shape == (2, 0)
    assert graph["node", "fixation", "node"].edge_attr.shape == (0, 7)

    batch = next(iter(DataLoader([graph], batch_size=1)))
    model = SpatioTemporalHeteroGNN(
        in_channels=6,
        hidden_channels=8,
        out_channels=1,
        output_scale=1.0,
        use_preprocess_mlp=False,
        use_edge_weights=True,
        conv_type="GCNConv",
        num_layers=2,
        use_delta_distance_edge_feature=True,
        use_fixation_edges=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out.shape) == (1, 1)
