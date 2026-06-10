"""Spatio-temporal GCN building blocks for emotion tasks."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax


RelationFusion = Literal["mean", "mlp"]


class ConfigurableSpatioTemporalGCN(nn.Module):
    """Configurable GCN core used by the thesis graph-model variants.

    The class intentionally supports only the architectural degrees of freedom
    needed for the thesis comparison: homogeneous vs. heterogeneous message
    passing, mean vs. MLP relation fusion, and disabled vs. learned signed edge
    weights. All variants use attention graph readout.
    """

    RELATIONS = ("temporal_forward", "temporal_backward", "spatial", "fixation")

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        output_scale: float,
        *,
        homogeneous: bool,
        relation_fusion: RelationFusion = "mean",
        use_learned_edge_weights: bool = False,
        use_preprocess_mlp: bool = True,
        add_self_loops: bool = False,
        dropout_mlp: float = 0.1,
        dropout_gnn: float = 0.1,
        dropout_head: float = 0.1,
        conv_type: str = "GCNConv",
        num_layers: int = 2,
        use_fixation_edges: bool = True,
        use_delta_distance_edge_feature: bool = True,
        spatial_edge_attr_dim: int | None = None,
        temporal_edge_attr_dim: int | None = None,
        fixation_edge_attr_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if conv_type != "GCNConv":
            raise ValueError("Thesis GCN variants support only conv_type='GCNConv'.")
        if relation_fusion not in {"mean", "mlp"}:
            raise ValueError(f"Unsupported relation_fusion='{relation_fusion}'. Choose 'mean' or 'mlp'.")
        if homogeneous and relation_fusion != "mean":
            raise ValueError("Homogeneous BasicGCN must use relation_fusion='mean'.")
        if homogeneous and use_learned_edge_weights:
            raise ValueError("Homogeneous BasicGCN does not use learned edge weights.")
        if use_learned_edge_weights and relation_fusion != "mlp":
            raise ValueError("Learned edge weights are only used by the MLP-fusion hetero variant.")

        self.homogeneous = bool(homogeneous)
        self.relation_fusion = relation_fusion
        self.relation_pooling = relation_fusion
        self.use_learned_edge_weights = bool(use_learned_edge_weights)
        self.use_edge_weights = self.use_learned_edge_weights
        self.supports_scalar_edge_weights = self.use_learned_edge_weights
        self.edge_weight_mode = "learned_signed" if self.use_learned_edge_weights else "none"
        self.use_preprocess_mlp = bool(use_preprocess_mlp)
        self.num_layers = int(num_layers)
        self.pooling = "attention"
        self.readout = "attention"
        self.output_scale = output_scale
        self.use_delta_distance_edge_feature = bool(use_delta_distance_edge_feature)

        relations = ("temporal_forward", "temporal_backward", "spatial")
        if use_fixation_edges:
            relations = (*relations, "fixation")
        self.relations = relations

        self.spatial_edge_attr_dim = (
            int(spatial_edge_attr_dim)
            if spatial_edge_attr_dim is not None
            else 7 if self.use_delta_distance_edge_feature else 6
        )
        self.temporal_edge_attr_dim = (
            int(temporal_edge_attr_dim)
            if temporal_edge_attr_dim is not None
            else 8 if self.use_delta_distance_edge_feature else 7
        )
        self.fixation_edge_attr_dim = (
            int(fixation_edge_attr_dim)
            if fixation_edge_attr_dim is not None
            else self.spatial_edge_attr_dim
        )

        if self.use_preprocess_mlp:
            self.preprocess_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.LayerNorm(hidden_channels),
                nn.Dropout(p=dropout_mlp),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            conv1_in_channels = hidden_channels
        else:
            conv1_in_channels = in_channels

        conv_kwargs = {"add_self_loops": add_self_loops}
        if self.use_learned_edge_weights:
            conv_kwargs = {"add_self_loops": False, "normalize": False}

        self.layer_norms = nn.ModuleList()
        if self.homogeneous:
            self.convs = nn.ModuleList()
            for layer_idx in range(num_layers):
                layer_in_channels = conv1_in_channels if layer_idx == 0 else hidden_channels
                self.convs.append(GCNConv(layer_in_channels, hidden_channels, add_self_loops=add_self_loops))
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
        else:
            self.relation_convs = nn.ModuleList()
            self.relation_fusion_mlps = nn.ModuleList()
            for layer_idx in range(num_layers):
                layer_in_channels = conv1_in_channels if layer_idx == 0 else hidden_channels
                self.relation_convs.append(
                    nn.ModuleDict(
                        {
                            relation: GCNConv(layer_in_channels, hidden_channels, **conv_kwargs)
                            for relation in self.relations
                        }
                    )
                )
                if self.relation_fusion == "mlp":
                    self.relation_fusion_mlps.append(
                        nn.Sequential(
                            nn.Linear(len(self.relations) * hidden_channels, hidden_channels),
                            nn.GELU(),
                            nn.Dropout(p=dropout_gnn),
                            nn.Linear(hidden_channels, hidden_channels),
                        )
                    )
                self.layer_norms.append(nn.LayerNorm(hidden_channels))

        self.input_residual_proj = nn.Linear(conv1_in_channels, hidden_channels)
        self.gnn_activation = nn.GELU()
        self.gnn_dropout = nn.Dropout(p=dropout_gnn)

        if self.use_learned_edge_weights:
            self.spatial_edge_weight_mlp = self._build_edge_weight_mlp(self.spatial_edge_attr_dim)
            self.temporal_edge_weight_mlp = self._build_edge_weight_mlp(self.temporal_edge_attr_dim)
            self.fixation_edge_weight_mlp = self._build_edge_weight_mlp(self.fixation_edge_attr_dim)

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden_channels, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden_channels, out_channels),
        )

    @staticmethod
    def _edge_type(relation: str) -> tuple[str, str, str]:
        """Return the heterograph edge type tuple for one node-node relation."""
        return ("node", relation, "node")

    @classmethod
    def collapse_edge_index(cls, data) -> torch.Tensor:
        """Collapse v2 relations into one deduplicated homogeneous edge index."""
        edge_indices = []
        edge_index_dict = data.edge_index_dict
        device = data["node"].x.device
        for relation in cls.RELATIONS:
            edge_index = edge_index_dict.get(cls._edge_type(relation))
            if edge_index is not None and edge_index.numel() > 0:
                edge_indices.append(edge_index.to(device=device, dtype=torch.long))

        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_index = torch.cat(edge_indices, dim=1)
        return torch.unique(edge_index, dim=1)

    @staticmethod
    def _build_edge_weight_mlp(input_dim: int) -> nn.Sequential:
        """Build the small edge-weight MLP used for learned signed weights."""
        return nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.GELU(),
            nn.Linear(6, 4),
            nn.GELU(),
            nn.Linear(4, 2),
            nn.GELU(),
            nn.Linear(2, 1),
        )

    @staticmethod
    def normalize_signed_edge_scores(
        raw_scores: torch.Tensor,
        dst_index: torch.Tensor,
        num_nodes: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize signed edge scores by incoming absolute score magnitude."""
        signed_scores = torch.tanh(raw_scores.view(-1))
        denom = torch.zeros(num_nodes, dtype=signed_scores.dtype, device=signed_scores.device)
        denom.index_add_(0, dst_index, signed_scores.abs())
        return signed_scores / (denom[dst_index] + eps)

    def _edge_weight_from_attr(
        self,
        relation: str,
        edge_attr: torch.Tensor | None,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor | None:
        """Return learned scalar edge weights, or None for unweighted variants."""
        if not self.use_learned_edge_weights or edge_attr is None:
            return None

        if relation in {"temporal_forward", "temporal_backward"}:
            if edge_attr.shape[-1] != self.temporal_edge_attr_dim:
                raise ValueError(
                    "Temporal learned edge attributes must have "
                    f"{self.temporal_edge_attr_dim} features, got {edge_attr.shape[-1]}."
                )
            raw_scores = self.temporal_edge_weight_mlp(edge_attr)
        elif relation == "spatial":
            if edge_attr.shape[-1] != self.spatial_edge_attr_dim:
                raise ValueError(
                    "Spatial learned edge attributes must have "
                    f"{self.spatial_edge_attr_dim} features, got {edge_attr.shape[-1]}."
                )
            raw_scores = self.spatial_edge_weight_mlp(edge_attr)
        elif relation == "fixation":
            if edge_attr.shape[-1] != self.fixation_edge_attr_dim:
                raise ValueError(
                    "Fixation learned edge attributes must have "
                    f"{self.fixation_edge_attr_dim} features, got {edge_attr.shape[-1]}."
                )
            raw_scores = self.fixation_edge_weight_mlp(edge_attr)
        else:
            raise ValueError(f"Unsupported relation for learned edge weights: {relation}.")

        return self.normalize_signed_edge_scores(
            raw_scores=raw_scores,
            dst_index=edge_index[1],
            num_nodes=num_nodes,
        )

    def _apply_relation_conv(
        self,
        conv: GCNConv,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply GCNConv with optional scalar edge weights."""
        if edge_weight is None:
            return conv(x, edge_index)
        return conv(x, edge_index, edge_weight=edge_weight.view(-1))

    def _forward_homogeneous(self, x_node: torch.Tensor, data) -> torch.Tensor:
        """Run the homogeneous collapsed-edge GCN stack."""
        x0_node = x_node
        edge_index = self.collapse_edge_index(data)
        for layer_idx, conv in enumerate(self.convs):
            layer_out = self.gnn_activation(conv(x_node, edge_index))
            residual = self.input_residual_proj(x0_node) if layer_idx == 0 else x_node
            x_node = self.layer_norms[layer_idx](layer_out + residual)
            x_node = self.gnn_dropout(x_node)
        return x_node

    def _forward_heterogeneous(self, x_node: torch.Tensor, data) -> torch.Tensor:
        """Run the heterogeneous relation-specific GCN stack."""
        x0_node = x_node
        edge_index_dict = data.edge_index_dict

        for layer_idx, relation_convs in enumerate(self.relation_convs):
            relation_outputs = []
            for relation in self.relations:
                edge_type = self._edge_type(relation)
                edge_index = edge_index_dict.get(edge_type)
                if edge_index is None or edge_index.numel() == 0:
                    hidden_dim = self.layer_norms[layer_idx].normalized_shape[0]
                    relation_outputs.append(
                        torch.zeros(
                            x_node.shape[0],
                            hidden_dim,
                            dtype=x_node.dtype,
                            device=x_node.device,
                        )
                    )
                    continue

                edge_weight = None
                if self.use_learned_edge_weights and hasattr(data[edge_type], "edge_attr"):
                    edge_weight = self._edge_weight_from_attr(
                        relation=relation,
                        edge_attr=data[edge_type].edge_attr,
                        edge_index=edge_index,
                        num_nodes=x_node.shape[0],
                    )

                relation_out = self._apply_relation_conv(
                    conv=relation_convs[relation],
                    x=x_node,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
                relation_outputs.append(self.gnn_activation(relation_out))

            if self.relation_fusion == "mean":
                fused = torch.stack(relation_outputs, dim=0).mean(dim=0)
            elif self.relation_fusion == "mlp":
                fused = self.relation_fusion_mlps[layer_idx](torch.cat(relation_outputs, dim=1))
            else:
                raise ValueError(f"Unsupported relation_fusion='{self.relation_fusion}'.")

            residual = self.input_residual_proj(x0_node) if layer_idx == 0 else x_node
            x_node = self.layer_norms[layer_idx](fused + residual)
            x_node = self.gnn_dropout(x_node)

        return x_node

    def forward(self, data, return_graph_embedding: bool = False):
        x_node = data["node"].x
        if self.use_preprocess_mlp:
            x_node = self.preprocess_mlp(x_node)

        if self.homogeneous:
            x_node = self._forward_homogeneous(x_node=x_node, data=data)
        else:
            x_node = self._forward_heterogeneous(x_node=x_node, data=data)

        batch = data["node"].batch
        scores = self.attention_pool(x_node)
        alpha = softmax(scores, batch)
        graph_emb = global_add_pool(alpha * x_node, batch)

        out = self.head(graph_emb) * self.output_scale
        if return_graph_embedding:
            return out, graph_emb
        return out


class BasicGCN(ConfigurableSpatioTemporalGCN):
    """Homogeneous GCN with collapsed v2 relations and attention readout."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("use_edge_weights", None)
        kwargs.pop("edge_weight_mode", None)
        kwargs.pop("relation_fusion", None)
        kwargs.pop("use_learned_edge_weights", None)
        kwargs.pop("homogeneous", None)
        kwargs.pop("readout", None)
        kwargs.pop("pooling", None)
        kwargs.pop("graph_pooling", None)
        kwargs.pop("head_pooling", None)
        super().__init__(
            *args,
            homogeneous=True,
            relation_fusion="mean",
            use_learned_edge_weights=False,
            **kwargs,
        )


class HeteroGCNMean(ConfigurableSpatioTemporalGCN):
    """Heterogeneous GCN with non-learned mean relation fusion."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("use_edge_weights", None)
        kwargs.pop("edge_weight_mode", None)
        kwargs.pop("relation_fusion", None)
        kwargs.pop("use_learned_edge_weights", None)
        kwargs.pop("homogeneous", None)
        kwargs.pop("pooling", None)
        kwargs.pop("graph_pooling", None)
        kwargs.pop("head_pooling", None)
        super().__init__(
            *args,
            homogeneous=False,
            relation_fusion="mean",
            use_learned_edge_weights=False,
            **kwargs,
        )


class HeteroGCNMLP(ConfigurableSpatioTemporalGCN):
    """Heterogeneous GCN with learned MLP relation fusion and no edge weights."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("use_edge_weights", None)
        kwargs.pop("edge_weight_mode", None)
        kwargs.pop("relation_fusion", None)
        kwargs.pop("use_learned_edge_weights", None)
        kwargs.pop("homogeneous", None)
        kwargs.pop("pooling", None)
        kwargs.pop("graph_pooling", None)
        kwargs.pop("head_pooling", None)
        super().__init__(
            *args,
            homogeneous=False,
            relation_fusion="mlp",
            use_learned_edge_weights=False,
            **kwargs,
        )


class HeteroGCNMLPWeights(ConfigurableSpatioTemporalGCN):
    """Heterogeneous GCN with MLP relation fusion and learned signed edge weights."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("use_edge_weights", None)
        kwargs.pop("edge_weight_mode", None)
        kwargs.pop("relation_fusion", None)
        kwargs.pop("use_learned_edge_weights", None)
        kwargs.pop("homogeneous", None)
        kwargs.pop("pooling", None)
        kwargs.pop("graph_pooling", None)
        kwargs.pop("head_pooling", None)
        super().__init__(
            *args,
            homogeneous=False,
            relation_fusion="mlp",
            use_learned_edge_weights=True,
            **kwargs,
        )
