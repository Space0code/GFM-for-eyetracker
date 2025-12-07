# model.py
"""
Graph neural network model for predicting next gaze coordinates.
Uses GraphSAGE layers to avoid temporal leakage in sequential data.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class NextPointGNN(nn.Module):
    """
    Minimal GraphSAGE: aggregates only from in-neighbors.
    With edges i->i+1, node i embeds using info up to i (no future leakage).
    """
    def __init__(self, in_channels: int = 2, hidden_dim: int = 64, output_dim: int = 2, num_layers: int = 2, layer: nn.Module = SAGEConv):
        """
        Args:
            in_channels: Input feature dimension (x, y coordinates = 2)
            hidden_dim: Hidden layer dimension
            num_layers: Number of GraphSAGE layers
        """
        super().__init__()
        self.convs = nn.ModuleList()
        # first layer: input -> hidden
        self.convs.append(layer(in_channels, hidden_dim))
        # additional hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(layer(hidden_dim, hidden_dim))
        # regression head: hidden -> (x_next, y_next)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index):
        """
        Forward pass through GraphSAGE layers + regression head.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Predicted next coordinates [num_nodes, 2]
        """
        # apply GraphSAGE convolutions with ReLU activation
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # predict next coordinates
        pred = self.head(x)
        # print("     x_shape:", x.shape)
        # print("     pred_shape:", pred.shape)
        return pred

class SpatioTemporalGNN(nn.Module):
    """
    Spatio-temporal GNN model.
    Combines undirected spatial and undirected temporal edges in the graph.


    """
    
    def __init__(self, in_channels: int = 5, hidden_dim: int = 64, output_dim: int = 2, num_layers: int = 2, layer: nn.Module = GCNConv):
        pass
