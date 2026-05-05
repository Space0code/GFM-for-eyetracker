"""Spatio-temporal heterogenous GNN building block for emotion tasks."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, global_max_pool, global_mean_pool

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
            num_layers: int = 2, pooling: str = "mean_max",
            ):
        nn.Module.__init__(self)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Choose 'mean' or 'mean_max'."
            )

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
        elif conv_type == "GATConv":
            ConvLayer = GATConv
            conv_kwargs = {"add_self_loops": add_self_loops}
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}. Choose 'GCNConv' or 'GATConv'.")

        self.num_layers = num_layers
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_channels = conv1_in_channels if layer_idx == 0 else hidden_channels
            conv = HeteroConv(
                {
                    ("node", "temporal_forward", "node"): ConvLayer(
                        layer_in_channels, hidden_channels, **conv_kwargs
                    ),
                    ("node", "temporal_backward", "node"): ConvLayer(
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

        self.input_residual_proj = nn.Linear(conv1_in_channels, hidden_channels)
        self.gnn_activation = nn.GELU()
        self.gnn_dropout = nn.Dropout(p=dropout_gnn)

        if pooling == "mean":
            head_in_channels = hidden_channels
        elif pooling == "mean_max":
            head_in_channels = 2 * hidden_channels
        else:
            raise ValueError(f"Unsupported pooling: {pooling}. Choose 'mean' or 'mean_max'.")

        self.head = nn.Sequential(
            nn.Linear(head_in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden_channels, out_channels),
        )
        self.output_scale = output_scale
        self.use_edge_weights = use_edge_weights
