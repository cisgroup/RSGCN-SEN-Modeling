import torch
from torch_geometric.nn import SAGEConv, Linear
import torch.nn.functional as F


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, sigmoid):
        super(GNNDecoder, self).__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.sigmoid = sigmoid

    def forward(self, node_feature, edge_index):
        z = torch.cat(
            [node_feature[edge_index[0]], node_feature[edge_index[1]]], dim=-1
        )
        z = F.relu(self.lin1(z))
        if self.sigmoid:
            z = F.sigmoid(self.lin2(z))
        else:
            z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, sigmoid):
        super(Model, self).__init__()

        self.encoder = GNNEncoder(input_channels, hidden_channels)
        self.decoder = GNNDecoder(hidden_channels, sigmoid)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z, edge_index)
