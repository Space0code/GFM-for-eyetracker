"""
Binary classification GNN model for emotion recognition.

Extends the base SpatioTemporalHeteroGNN for binary classification by:
- Setting out_channels=1 for single probability output
- Removing output scaling (keeps sigmoid output in [0, 1])
"""

import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from emotions.model import SpatioTemporalHeteroGNN


class BinarySpatioTemporalGNN(SpatioTemporalHeteroGNN):
    """
    Binary classification variant of SpatioTemporalHeteroGNN.
    
    Overrides forward method to output single probability without scaling.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        use_preprocess_mlp: bool = True,
        add_self_loops: bool = False,
        dropout_mlp: float = 0.1,
        dropout_gnn: float = 0.1,
        dropout_head: float = 0.1,
        aggr: str = "mean",
        conv_type: str = "GCNConv",
    ):
        """
        Initialize binary classification GNN.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Hidden layer dimension
            use_preprocess_mlp: Whether to use preprocessing MLP
            add_self_loops: Whether to add self-loops in convolutions
            dropout_mlp: Dropout rate for preprocessing MLP
            dropout_gnn: Dropout rate for GNN layers
            dropout_head: Dropout rate for prediction head
            aggr: Aggregation method for hetero edges
            conv_type: Type of convolutional layer (GCNConv or GATConv)
        """
        # Initialize parent with out_channels=1 and output_scale=1.0 (no scaling)
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,  # Single binary output
            output_scale=1.0,  # No scaling for binary classification
            use_preprocess_mlp=use_preprocess_mlp,
            add_self_loops=add_self_loops,
            dropout_mlp=dropout_mlp,
            dropout_gnn=dropout_gnn,
            dropout_head=dropout_head,
            aggr=aggr,
            conv_type=conv_type,
        )
    
    def forward(self, data):
        """
        Forward pass for binary classification.
        
        Args:
            data: HeteroData graph batch
            
        Returns:
            Tensor of shape [batch_size, 1] with probabilities in [0, 1]
        """
        # Use parent forward, which applies sigmoid and scaling
        # Since we set output_scale=1.0, output is already in [0, 1]
        return super().forward(data)
