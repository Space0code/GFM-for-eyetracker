# model.py 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, global_mean_pool

class SpatioTemporalHeteroGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, output_scale: float = 10.0):
        super().__init__()

        # 1st hetero GCN layer (temporal + spatial)
        self.conv1 = HeteroConv(
            {
                ("node", "temporal", "node"): GCNConv(in_channels, hidden_channels),
                ("node", "spatial", "node"): GCNConv(in_channels, hidden_channels),
            },
            aggr="sum",  # how to combine temporal + spatial messages
        )

        # 2nd hetero GCN layer
        self.conv2 = HeteroConv(
            {
                ("node", "temporal", "node"): GCNConv(hidden_channels, hidden_channels),
                ("node", "spatial", "node"): GCNConv(hidden_channels, hidden_channels),
            },
            aggr="sum",
        )

        # Final MLP for graph-level output
        # Output is bounded to [0, 10] for emotion scores
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()  # Output in [0, 1], will scale to [0, output_scale]
        )
        self.output_scale = output_scale

    def forward(self, data):
        # data is a HeteroData from your SpacioTemporalDataset
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # 1st layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # 2nd layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # we have only one node type "node"
        x_node = x_dict["node"]              # [num_nodes, hidden]
        batch = data["node"].batch           # [num_nodes] (set by PyG DataLoader)

        graph_emb = global_mean_pool(x_node, batch)  # [num_graphs, hidden]
        out = self.head(graph_emb)                    # [num_graphs, out_channels] in [0, 1]
        out = out * self.output_scale                 # Scale to [0, 10]
        return out
