# model.py 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HeteroConv, global_mean_pool

class SpatioTemporalHeteroGNN(nn.Module):
    def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, 
            output_scale: float, use_preprocess_mlp: bool = True, add_self_loops: bool = False,
            dropout_mlp: float = 0.1, dropout_gnn: float = 0.1, dropout_head: float = 0.1,
            aggr: str = "mean", conv_type: str = "GCNConv",
            ):
        super().__init__()

        # Preprocessing MLP 
        self.use_preprocess_mlp = use_preprocess_mlp
        if self.use_preprocess_mlp:
            # print("Using preprocessing MLP for input features.")
            self.preprocess_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(p=dropout_mlp),
                nn.Linear(hidden_channels, hidden_channels),
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

        # 1st hetero GCN layer (temporal + spatial)
        self.conv1 = HeteroConv(
            {
                ("node", "temporal", "node"): ConvLayer(conv1_in_channels, hidden_channels, **conv_kwargs),
                ("node", "spatial", "node"): ConvLayer(conv1_in_channels, hidden_channels, **conv_kwargs),
            },
            aggr=aggr,  # how to combine temporal + spatial messages
        )

        # 2nd hetero GCN layer
        self.conv2 = HeteroConv(
            {
                ("node", "temporal", "node"): ConvLayer(hidden_channels, hidden_channels, **conv_kwargs),
                ("node", "spatial", "node"): ConvLayer(hidden_channels, hidden_channels, **conv_kwargs),
            },
            aggr=aggr,
        )
        
        # Anti-oversmoothing: Layer normalization for residual connections
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        
        # Projection for residual connection from input to conv1
        self.proj_x0 = nn.Linear(conv1_in_channels, hidden_channels)
        
        # Activation and dropout for GNN layers
        self.gnn_activation = nn.GELU()
        self.gnn_dropout = nn.Dropout(p=dropout_gnn)

        # Final MLP for graph-level output
        # Output is bounded to [0, 10] for emotion scores
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden_channels, out_channels),
            # nn.Sigmoid()  # Output in [0, 1], will scale to [0, output_scale]
        )
        self.output_scale = output_scale

    def forward(self, data):
        
        # data is a HeteroData from your SpacioTemporalDataset
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # Preprocess raw input with MLP
        if self.use_preprocess_mlp:
            x_dict["node"] = self.preprocess_mlp(x_dict["node"])

        # Store input for residual connection
        x_input = x_dict.copy()
        
        # 1st layer with residual + layer norm
        # x1 = LN(GELU(conv1(x0)) + proj_x0(x0))
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.gnn_activation(v) for k, v in x_dict.items()}
        x_dict["node"] = self.ln1(x_dict["node"] + self.proj_x0(x_input["node"]))
        x_dict = {k: self.gnn_dropout(v) for k, v in x_dict.items()}
        
        # Store for next residual connection
        x_prev = x_dict

        # 2nd layer with residual + layer norm
        # x2 = LN(GELU(conv2(x1)) + x1)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: self.gnn_activation(v) for k, v in x_dict.items()}
        x_dict["node"] = self.ln2(x_dict["node"] + x_prev["node"])
        x_dict = {k: self.gnn_dropout(v) for k, v in x_dict.items()}
        
        # we have only one node type "node"
        x_node = x_dict["node"]              # [num_nodes, hidden]
        batch = data["node"].batch           # [num_nodes] (set by PyG DataLoader)

        graph_emb = global_mean_pool(x_node, batch)  # [num_graphs, hidden]
        out = self.head(graph_emb)                    # [num_graphs, out_channels] in [0, 1]
        out = out * self.output_scale                 # Scale to [0, 10], for binary, output scale is 1.0 so no scaling
        return out
