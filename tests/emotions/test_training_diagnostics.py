from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from emotions.common.training_diagnostics import (
    collect_gradient_norm_stats,
    collect_gnn_loader_diagnostics,
    save_gnn_fold_diagnostics,
)
from emotions.model import HeteroGCNMLPWeights


def _build_v2_graph(label: float) -> HeteroData:
    graph = HeteroData()
    graph["node"].x = torch.randn(5, 4, dtype=torch.float32)
    graph["node"].num_nodes = 5

    forward = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    backward = torch.flip(forward, dims=[0])
    spatial = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 0]], dtype=torch.long)
    fixation = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    graph["node", "temporal_forward", "node"].edge_index = forward
    graph["node", "temporal_backward", "node"].edge_index = backward
    graph["node", "spatial", "node"].edge_index = spatial
    graph["node", "fixation", "node"].edge_index = fixation
    graph["node", "temporal_forward", "node"].edge_attr = torch.randn(forward.shape[1], 7)
    graph["node", "temporal_backward", "node"].edge_attr = torch.randn(backward.shape[1], 7)
    graph["node", "spatial", "node"].edge_attr = torch.randn(spatial.shape[1], 6)
    graph["node", "fixation", "node"].edge_attr = torch.randn(fixation.shape[1], 6)
    graph.y = torch.tensor([label], dtype=torch.float32)
    return graph


def _build_model() -> HeteroGCNMLPWeights:
    model = HeteroGCNMLPWeights(
        in_channels=4,
        hidden_channels=8,
        out_channels=1,
        output_scale=1.0,
        use_preprocess_mlp=False,
        conv_type="GCNConv",
        num_layers=2,
        use_delta_distance_edge_feature=False,
        use_fixation_edges=True,
    )
    model.eval()
    return model


def test_gradient_norm_stats_are_collected_without_clipping() -> None:
    layer = torch.nn.Linear(2, 1)
    loss = layer(torch.ones(3, 2)).sum()
    loss.backward()

    stats = collect_gradient_norm_stats(layer.parameters())

    assert math.isfinite(stats.mean)
    assert math.isfinite(stats.max)
    assert stats.max >= stats.mean > 0.0


def test_gnn_loader_diagnostics_include_embeddings_logits_entropy_and_edge_weights() -> None:
    loader = DataLoader([_build_v2_graph(0.0), _build_v2_graph(1.0)], batch_size=2)
    diagnostics = collect_gnn_loader_diagnostics(
        model=_build_model(),
        loader=loader,
        device=torch.device("cpu"),
        split="val",
        task_kind="binary",
    )

    assert diagnostics["split"] == "val"
    assert math.isfinite(diagnostics["graph_embedding_variance"])
    assert math.isfinite(diagnostics["graph_embedding_mean_pairwise_cosine_similarity"])
    assert math.isfinite(diagnostics["logit_mean"])
    assert math.isfinite(diagnostics["logit_range"])
    assert math.isfinite(diagnostics["prediction_entropy_mean"])
    assert math.isfinite(diagnostics["temporal_forward_edge_weight_mean"])
    assert math.isfinite(diagnostics["temporal_backward_edge_weight_std"])
    assert math.isfinite(diagnostics["spatial_edge_weight_min"])
    assert math.isfinite(diagnostics["fixation_edge_weight_max"])


def test_save_gnn_fold_diagnostics_writes_one_row_per_split(tmp_path: Path) -> None:
    loader = DataLoader([_build_v2_graph(0.0), _build_v2_graph(1.0)], batch_size=2)
    output_path = tmp_path / "gnn_fold_diagnostics.csv"

    saved = save_gnn_fold_diagnostics(
        model=_build_model(),
        loaders={"train": loader, "val": loader},
        device=torch.device("cpu"),
        output_path=output_path,
        task_kind="binary",
        metadata={"best_epoch": 3},
    )

    reloaded = pd.read_csv(output_path)
    assert output_path.exists()
    assert list(saved["split"]) == ["train", "val"]
    assert list(reloaded["split"]) == ["train", "val"]
    assert set(reloaded["best_epoch"]) == {3}
