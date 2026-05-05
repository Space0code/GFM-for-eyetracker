from __future__ import annotations

from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from emotions.model import SpatioTemporalHeteroGNN, SpatioTemporalHeteroGNNV1
from emotions.utils import create_splitter


def _build_graph(subject: str, recording: str) -> HeteroData:
    graph = HeteroData()
    graph["node"].x = torch.randn(6, 4, dtype=torch.float32)
    graph["node"].num_nodes = 6

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
            [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    graph["node", "temporal", "node"].edge_index = edge_index
    graph["node", "spatial", "node"].edge_index = edge_index
    graph.y = torch.tensor([0.0], dtype=torch.float32)
    graph.subject = subject
    graph.recording = recording
    return graph


def _build_v2_graph(subject: str, recording: str) -> HeteroData:
    graph = HeteroData()
    graph["node"].x = torch.randn(6, 4, dtype=torch.float32)
    graph["node"].num_nodes = 6

    forward = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    backward = torch.flip(forward, dims=[0])
    spatial = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 2], [2, 3, 4, 5, 0, 1, 0, 1]],
        dtype=torch.long,
    )
    graph["node", "temporal_forward", "node"].edge_index = forward
    graph["node", "temporal_backward", "node"].edge_index = backward
    graph["node", "spatial", "node"].edge_index = spatial
    graph.y = torch.tensor([0.0], dtype=torch.float32)
    graph.subject = subject
    graph.recording = recording
    return graph


def test_recording_kfold_has_no_recording_leakage() -> None:
    samples = []
    for recording in ["r1", "r2", "r3", "r4", "r5", "r6"]:
        for subject in ["P1", "P2"]:
            samples.append(SimpleNamespace(subject=subject, recording=recording))

    splitter = create_splitter(
        strategy="recording_kfold",
        samples=samples,
        val_size=1,
        random_state=42,
        n_splits=3,
    )
    splits = list(splitter.split())
    assert len(splits) == 3

    for train_idx, val_idx, test_idx in splits:
        train_recordings = {samples[int(i)].recording for i in train_idx}
        val_recordings = {samples[int(i)].recording for i in val_idx}
        test_recordings = {samples[int(i)].recording for i in test_idx}

        assert len(test_recordings) >= 1
        assert train_recordings.isdisjoint(val_recordings)
        assert train_recordings.isdisjoint(test_recordings)
        assert val_recordings.isdisjoint(test_recordings)


def test_subject_kfold_has_no_subject_leakage() -> None:
    samples = []
    for subject in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        for recording in ["r1", "r2"]:
            samples.append(SimpleNamespace(subject=subject, recording=recording))

    splitter = create_splitter(
        strategy="subject_kfold",
        samples=samples,
        val_size=1,
        random_state=42,
        n_splits=3,
    )
    splits = list(splitter.split())
    assert len(splits) == 3

    for train_idx, val_idx, test_idx in splits:
        train_subjects = {samples[int(i)].subject for i in train_idx}
        val_subjects = {samples[int(i)].subject for i in val_idx}
        test_subjects = {samples[int(i)].subject for i in test_idx}

        assert len(test_subjects) >= 1
        assert train_subjects.isdisjoint(val_subjects)
        assert train_subjects.isdisjoint(test_subjects)
        assert val_subjects.isdisjoint(test_subjects)


def test_gnn_depth_pooling_output_shape_stays_consistent() -> None:
    loader = DataLoader(
        [_build_graph("P1", "r1"), _build_graph("P2", "r2")],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))

    for num_layers in [1, 2, 3, 5, 10, 50]:
        for pooling in ["mean", "mean_max"]:
            model = SpatioTemporalHeteroGNN(
                in_channels=4,
                hidden_channels=8,
                out_channels=1,
                output_scale=1.0,
                use_preprocess_mlp=False,
                use_edge_weights=False,
                conv_type="GCNConv",
                num_layers=num_layers,
                pooling=pooling,
            )
            model.eval()
            with torch.no_grad():
                out = model(batch)
            assert tuple(out.shape) == (2, 1)


def test_gnn_v1_forward_pass_accepts_legacy_edges() -> None:
    loader = DataLoader([_build_graph("P1", "r1"), _build_graph("P2", "r2")], batch_size=2)
    batch = next(iter(loader))
    model = SpatioTemporalHeteroGNNV1(
        in_channels=4,
        hidden_channels=8,
        out_channels=1,
        output_scale=1.0,
        use_preprocess_mlp=False,
        use_edge_weights=False,
        conv_type="GCNConv",
    )
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out.shape) == (2, 1)


def test_gnn_v2_forward_pass_accepts_split_temporal_edges() -> None:
    loader = DataLoader([_build_v2_graph("P1", "r1"), _build_v2_graph("P2", "r2")], batch_size=2)
    batch = next(iter(loader))
    model = SpatioTemporalHeteroGNN(
        in_channels=4,
        hidden_channels=8,
        out_channels=1,
        output_scale=1.0,
        use_preprocess_mlp=False,
        use_edge_weights=False,
        conv_type="GCNConv",
        num_layers=3,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out.shape) == (2, 1)
