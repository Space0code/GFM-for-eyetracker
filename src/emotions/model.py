"""Spatio-temporal heterogenous GNN building block for emotion tasks."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

class SpatioTemporalHeteroGNNV1(nn.Module):
    """Frozen v1 spatio-temporal heterogeneous GNN architecture.

    This class preserves the pre-v2 architecture: one temporal relation, one
    spatial relation, PyG HeteroConv relation aggregation, and mean/mean_max
    graph pooling.
    """

    def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, 
            output_scale: float, use_preprocess_mlp: bool = True, use_edge_weights: bool = True, add_self_loops: bool = False,
            dropout_mlp: float = 0.1, dropout_gnn: float = 0.1, dropout_head: float = 0.1,
            aggr: str = "mean", conv_type: str = "GCNConv",
            num_layers: int = 2, pooling: str = "mean_max",
            ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Choose 'mean' or 'mean_max'."
            )

        # Preprocessing MLP 
        self.use_preprocess_mlp = use_preprocess_mlp
        if self.use_preprocess_mlp:
            # print("Using preprocessing MLP for input features.")
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
            # print("Not using preprocessing MLP for input features.")
            conv1_in_channels = in_channels

        # Select convolutional layer type
        if conv_type == "GCNConv":
            ConvLayer = GCNConv
            conv_kwargs = {"add_self_loops": add_self_loops}
        elif conv_type == "GATConv":
            ConvLayer = GATConv
            conv_kwargs = {"add_self_loops": add_self_loops}
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}. Choose 'GCNConv' or 'GATConv'.")

        self.num_layers = num_layers
        self.pooling = pooling

        # Per-layer hetero conv blocks with residual + layer norm.
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_channels = conv1_in_channels if layer_idx == 0 else hidden_channels
            conv = HeteroConv(
                {
                    ("node", "temporal", "node"): ConvLayer(
                        layer_in_channels, hidden_channels, **conv_kwargs
                    ),
                    ("node", "spatial", "node"): ConvLayer(
                        layer_in_channels, hidden_channels, **conv_kwargs
                    ),
                },
                aggr=aggr,
            )
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # Projection for first residual connection from layer-0 input.
        self.input_residual_proj = nn.Linear(conv1_in_channels, hidden_channels)
        
        # Activation and dropout for GNN layers
        self.gnn_activation = nn.GELU()
        self.gnn_dropout = nn.Dropout(p=dropout_gnn)

        if pooling == "mean":
            head_in_channels = hidden_channels
        elif pooling == "mean_max":
            head_in_channels = 2 * hidden_channels
        else:
            raise ValueError(f"Unsupported pooling: {pooling}. Choose 'mean' or 'mean_max'.")

        # Final MLP for graph-level output
        # Output is bounded to [0, 10] for emotion scores
        self.head = nn.Sequential(
            nn.Linear(head_in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden_channels, out_channels),
            # nn.Sigmoid()  # Output in [0, 1], will scale to [0, output_scale]
        )
        self.output_scale = output_scale
        self.use_edge_weights = use_edge_weights

    def forward(self, data, return_graph_embedding: bool = False):
        
        # data is a HeteroData from your SpacioTemporalDataset
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        edge_weight_dict = None
        if self.use_edge_weights:
            edge_weight_dict = {k: data[k].edge_attr for k in edge_index_dict.keys() if hasattr(data[k], 'edge_attr')}
        
        # Preprocess raw input with MLP
        if self.use_preprocess_mlp:
            x_dict["node"] = self.preprocess_mlp(x_dict["node"])

        x0_node = x_dict["node"]

        for layer_idx, conv in enumerate(self.convs):
            if self.use_edge_weights and edge_weight_dict:
                layer_out = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
            else:
                layer_out = conv(x_dict, edge_index_dict)

            layer_out = {k: self.gnn_activation(v) for k, v in layer_out.items()}

            if layer_idx == 0:
                residual = self.input_residual_proj(x0_node)
            else:
                residual = x_dict["node"]

            layer_out["node"] = self.layer_norms[layer_idx](layer_out["node"] + residual)
            layer_out = {k: self.gnn_dropout(v) for k, v in layer_out.items()}
            x_dict = layer_out
        
        # we have only one node type "node"
        x_node = x_dict["node"]              # [num_nodes, hidden]
        batch = data["node"].batch           # [num_nodes] (set by PyG DataLoader)

        if self.pooling == "mean":
            graph_emb = global_mean_pool(x_node, batch)  # [num_graphs, hidden]
        elif self.pooling == "mean_max":
            mean_emb = global_mean_pool(x_node, batch)  # [num_graphs, hidden]
            max_emb = global_max_pool(x_node, batch)    # [num_graphs, hidden]
            graph_emb = torch.cat([mean_emb, max_emb], dim=1)  # [num_graphs, 2*hidden]
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}. Choose 'mean' or 'mean_max'.")

        out = self.head(graph_emb)                    # [num_graphs, out_channels]
        out = out * self.output_scale                 # Scale to [0, 10], for binary, output scale is 1.0 so no scaling

        if return_graph_embedding:
            return out, graph_emb
        return out


class SpatioTemporalHeteroGNN(SpatioTemporalHeteroGNNV1):
    """V2 spatio-temporal heterogeneous GNN architecture.

    Stage 1 keeps the v1 message-passing mechanics but consumes the v2 graph
    schema with separate temporal-forward and temporal-backward relations.
    Later stages add relation-fusion MLPs, attention pooling, and learned
    signed edge weights.
    """

    def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int,
            output_scale: float, use_preprocess_mlp: bool = True, use_edge_weights: bool = True, add_self_loops: bool = False,
            dropout_mlp: float = 0.1, dropout_gnn: float = 0.1, dropout_head: float = 0.1,
            aggr: str = "mean", conv_type: str = "GCNConv",
            num_layers: int = 2, pooling: str = "attention",
            edge_weight_mode: str = "learned_signed",
            ):
        nn.Module.__init__(self)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if pooling not in {"mean", "mean_max", "attention"}:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Choose 'mean', 'mean_max', or 'attention'."
            )
        if edge_weight_mode not in {"handcrafted", "learned_signed"}:
            raise ValueError(
                f"Unsupported edge_weight_mode='{edge_weight_mode}'. "
                "Choose 'handcrafted' or 'learned_signed'."
            )
        self.edge_weight_mode = edge_weight_mode

        self.use_preprocess_mlp = use_preprocess_mlp
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

        if conv_type == "GCNConv":
            ConvLayer = GCNConv
            conv_kwargs = {"add_self_loops": add_self_loops}
            if edge_weight_mode == "learned_signed":
                conv_kwargs = {"add_self_loops": False, "normalize": False}
        elif conv_type == "GATConv":
            ConvLayer = GATConv
            conv_kwargs = {"add_self_loops": add_self_loops}
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}. Choose 'GCNConv' or 'GATConv'.")

        self.relations = ("spatial", "temporal_forward", "temporal_backward")
        self.num_layers = num_layers
        self.pooling = pooling

        self.relation_convs = nn.ModuleList()
        self.relation_fusion_mlps = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_channels = conv1_in_channels if layer_idx == 0 else hidden_channels
            relation_convs = nn.ModuleDict(
                {
                    relation: ConvLayer(layer_in_channels, hidden_channels, **conv_kwargs)
                    for relation in self.relations
                }
            )
            self.relation_convs.append(relation_convs)
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
        self.spatial_edge_weight_mlp = nn.Sequential(
            nn.Linear(6, 6),
            nn.GELU(),
            nn.Linear(6, 4),
            nn.GELU(),
            nn.Linear(4, 2),
            nn.GELU(),
            nn.Linear(2, 1),
        )
        self.temporal_edge_weight_mlp = nn.Sequential(
            nn.Linear(7, 6),
            nn.GELU(),
            nn.Linear(6, 4),
            nn.GELU(),
            nn.Linear(4, 2),
            nn.GELU(),
            nn.Linear(2, 1),
        )

        if pooling == "mean":
            head_in_channels = hidden_channels
        elif pooling == "mean_max":
            head_in_channels = 2 * hidden_channels
        elif pooling == "attention":
            head_in_channels = hidden_channels
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(p=dropout_head),
                nn.Linear(hidden_channels, 1),
            )
        else:
            raise ValueError(f"Unsupported pooling: {pooling}. Choose 'mean', 'mean_max', or 'attention'.")

        self.head = nn.Sequential(
            nn.Linear(head_in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden_channels, out_channels),
        )
        self.output_scale = output_scale
        self.use_edge_weights = use_edge_weights

    @staticmethod
    def _edge_type(relation: str) -> tuple[str, str, str]:
        """Return the heterograph edge type tuple for one node-node relation."""
        return ("node", relation, "node")

    @staticmethod
    def _apply_relation_conv(
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply a relation convolution, passing edge weights when supported."""
        if isinstance(conv, GCNConv) and edge_weight is not None:
            return conv(x, edge_index, edge_weight=edge_weight.view(-1))
        return conv(x, edge_index)

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
        """Resolve scalar edge weights from stored edge attributes."""
        if edge_attr is None:
            return None
        if self.edge_weight_mode == "handcrafted" or edge_attr.dim() == 1 or edge_attr.shape[-1] == 1:
            return edge_attr.view(-1)

        if relation == "spatial":
            if edge_attr.shape[-1] != 6:
                raise ValueError(f"Spatial learned edge attributes must have 6 features, got {edge_attr.shape[-1]}.")
            raw_scores = self.spatial_edge_weight_mlp(edge_attr)
        elif relation in {"temporal_forward", "temporal_backward"}:
            if edge_attr.shape[-1] != 7:
                raise ValueError(f"Temporal learned edge attributes must have 7 features, got {edge_attr.shape[-1]}.")
            raw_scores = self.temporal_edge_weight_mlp(edge_attr)
        else:
            raise ValueError(f"Unsupported relation for learned edge weights: {relation}.")

        return self.normalize_signed_edge_scores(
            raw_scores=raw_scores,
            dst_index=edge_index[1],
            num_nodes=num_nodes,
        )

    def forward(self, data, return_graph_embedding: bool = False):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        edge_weight_dict = None
        if self.use_edge_weights:
            edge_weight_dict = {
                k: data[k].edge_attr
                for k in edge_index_dict.keys()
                if hasattr(data[k], "edge_attr")
            }

        if self.use_preprocess_mlp:
            x_dict["node"] = self.preprocess_mlp(x_dict["node"])

        x0_node = x_dict["node"]
        x_node = x0_node

        for layer_idx, relation_convs in enumerate(self.relation_convs):
            relation_outputs = []
            for relation in self.relations:
                edge_type = self._edge_type(relation)
                if edge_type not in edge_index_dict:
                    relation_outputs.append(
                        torch.zeros(
                            x_node.shape[0],
                            self.layer_norms[layer_idx].normalized_shape[0],
                            dtype=x_node.dtype,
                            device=x_node.device,
                        )
                    )
                    continue

                edge_weight = None
                if edge_weight_dict is not None:
                    edge_weight = self._edge_weight_from_attr(
                        relation=relation,
                        edge_attr=edge_weight_dict.get(edge_type),
                        edge_index=edge_index_dict[edge_type],
                        num_nodes=x_node.shape[0],
                    )

                relation_out = self._apply_relation_conv(
                    conv=relation_convs[relation],
                    x=x_node,
                    edge_index=edge_index_dict[edge_type],
                    edge_weight=edge_weight,
                )
                relation_outputs.append(self.gnn_activation(relation_out))

            fused = self.relation_fusion_mlps[layer_idx](torch.cat(relation_outputs, dim=1))

            if layer_idx == 0:
                residual = self.input_residual_proj(x0_node)
            else:
                residual = x_node

            x_node = self.layer_norms[layer_idx](fused + residual)
            x_node = self.gnn_dropout(x_node)

        batch = data["node"].batch

        if self.pooling == "mean":
            graph_emb = global_mean_pool(x_node, batch)
        elif self.pooling == "mean_max":
            mean_emb = global_mean_pool(x_node, batch)
            max_emb = global_max_pool(x_node, batch)
            graph_emb = torch.cat([mean_emb, max_emb], dim=1)
        elif self.pooling == "attention":
            scores = self.attention_pool(x_node)
            alpha = softmax(scores, batch)
            graph_emb = global_add_pool(alpha * x_node, batch)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}. Choose 'mean', 'mean_max', or 'attention'.")

        out = self.head(graph_emb)
        out = out * self.output_scale

        if return_graph_embedding:
            return out, graph_emb
        return out
