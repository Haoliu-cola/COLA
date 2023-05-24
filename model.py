import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric
from tqdm import tqdm
from torch.optim import Adam

from torch_geometric.nn import GCNConv, GATConv, GraphConv, SGConv
from torch_geometric.datasets import Planetoid

class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.PReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, input_dim))

        super().__init__(*layers)

class GNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        layer_name="GCN",
        activation_name="relu",
        dp_rate=0.1,
        **kwargs,
    ):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer_by_name = {"GCN": GCNConv, "GAT": GATConv, "GraphConv": GraphConv, "SGC": SGConv}
        gnn_layer = gnn_layer_by_name[layer_name]
        activation_by_name = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        activation = activation_by_name[activation_name]

        layers = []
        in_channels, out_channels = input_dim, hidden_dim
        for _ in range(num_layers):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.BatchNorm1d(out_channels, momentum=0.01),
                activation,
                nn.Dropout(dp_rate),
            ]
            in_channels = hidden_dim
        layers += [gnn_layer(in_channels=out_channels, out_channels=out_dim, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self,
                x,
                edge_index,
                edge_weight=None):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x