# baseline_models.py
"""
Baseline models for different tasks to compare against GNN-based approaches.
Includes:
- MLP
- CNN
- LightGBM

"""


from typing import List
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """
    A simple Multi-Layer Perceptron baseline model.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.2):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout

        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.convs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.convs.append(nn.ReLU())
            self.convs.append(nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dims[-1], output_dim)   
        self.net = nn.Sequential(*self.convs, self.head)
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
class CNNBaseline(nn.Module):
    """
    A simple Convolutional Neural Network baseline model.
    """
    def __init__(self, input_channels: int, output_dim: int, hidden_dims: List[int] = [64, 32], num_layers: int = 2, dropout: float = 0.2):
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        super().__init__()
        layers = []
        in_channels = input_channels
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = hidden_dim
        self.conv_net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.net = nn.Sequential(self.conv_net, nn.Flatten(), self.head)
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
    