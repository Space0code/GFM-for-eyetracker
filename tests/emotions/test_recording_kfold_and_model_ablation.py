from __future__ import annotations

from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from emotions.model import BasicGCN, SpatioTemporalHeteroGNN, SpatioTemporalHeteroGNNV1
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
    graph["node", "temporal_forward", "node"].edge_attr = torch.randn(forward.shape[1], 7)
    graph["node", "temporal_backward", "node"].edge_attr = torch.randn(backward.shape[1], 7)
    graph["node", "spatial", "node"].edge_attr = torch.randn(spatial.shape[1], 6)
    graph.y = torch.tensor([0.0], dtype=torch.float32)
    graph.subject = subject
    graph.recording = recording
    return graph


def _build_basic_gcn_graph(subject: str, recording: str) -> HeteroData:
    graph = HeteroData()
    graph["node"].x = torch.randn(5, 4, dtype=torch.float32)
    graph["node"].num_nodes = 5

    forward = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    backward = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long)
    spatial = torch.tensor([[0, 1, 4], [1, 2, 0]], dtype=torch.long)
    fixation = torch.tensor([[0, 3, 4], [1, 4, 0]], dtype=torch.long)
    for relation, edge_index, attr_dim in [
        ("temporal_forward", forward, 8),
        ("temporal_backward", backward, 8),
        ("spatial", spatial, 7),
        ("fixation", fixation, 7),
    ]:
        graph["node", relation, "node"].edge_index = edge_index
        graph["node", relation, "node"].edge_attr = torch.randn(edge_index.shape[1], attr_dim)

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
        for pooling in ["mean", "mean_max", "attention"]:
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
        use_edge_weights=True,
        conv_type="GCNConv",
        num_layers=3,
        use_delta_distance_edge_feature=False,
        use_fixation_edges=False,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out.shape) == (2, 1)


def test_basic_gcn_collapses_v2_relations_and_removes_duplicate_edges() -> None:
    graph = _build_basic_gcn_graph("P1", "r1")

    edge_index = BasicGCN.collapse_edge_index(graph)
    edge_pairs = set(map(tuple, edge_index.t().tolist()))

    assert edge_pairs == {
        (0, 1),
        (1, 2),
        (2, 3),
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 0),
        (3, 4),
    }
    assert edge_index.shape[1] == len(edge_pairs)


def test_basic_gcn_forward_uses_v2_graph_and_ignores_edge_attributes() -> None:
    loader = DataLoader(
        [_build_basic_gcn_graph("P1", "r1"), _build_basic_gcn_graph("P2", "r2")],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))
    model = BasicGCN(
        in_channels=4,
        hidden_channels=8,
        out_channels=3,
        output_scale=1.0,
        use_preprocess_mlp=False,
        use_edge_weights=True,
        conv_type="GCNConv",
        num_layers=2,
        readout="mean",
    )
    model.eval()

    with torch.no_grad():
        out, graph_emb = model(batch, return_graph_embedding=True)

    assert tuple(out.shape) == (2, 3)
    assert tuple(graph_emb.shape) == (2, 8)
    assert model.use_edge_weights is False


def test_signed_edge_weight_normalization_is_per_target_node() -> None:
    raw_scores = torch.tensor([0.5, -1.0, 2.0, 0.25], dtype=torch.float32)
    dst_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    weights = SpatioTemporalHeteroGNN.normalize_signed_edge_scores(
        raw_scores=raw_scores,
        dst_index=dst_index,
        num_nodes=2,
    )
    assert torch.isfinite(weights).all()
    for target in [0, 1]:
        target_weights = weights[dst_index == target]
        assert torch.isclose(target_weights.abs().sum(), torch.tensor(1.0), atol=1e-6)
