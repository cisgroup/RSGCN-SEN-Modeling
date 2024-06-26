from src.core.SpatialGCNLayer import SpatialGraphConv
import torch
from torch_geometric.nn import Linear


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = SpatialGraphConv(
            coors=2,
            in_channels=input_channels,
            out_channels=hidden_channels,
            hidden_size=hidden_channels,
            dropout=0.1,
        )
        self.conv2 = SpatialGraphConv(
            coors=2,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_size=hidden_channels,
            dropout=0.1,
        )

        self.conv3 = SpatialGraphConv(
            coors=2,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_size=hidden_channels,
            dropout=0.1,
        )

    def forward(self, x, edge_index, pos):
        x = self.conv1(x=x, pos=pos, edge_index=edge_index)
        x = self.conv2(x=x, pos=pos, edge_index=edge_index)
        x = self.conv3(x=x, pos=pos, edge_index=edge_index)
        return x


class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, sigmoid):
        super(GNNDecoder, self).__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)
        self.sigmoid = sigmoid

    def forward(self, node_feature, edge_index):
        z = torch.cat(
            [node_feature[edge_index[0]], node_feature[edge_index[1]]], dim=-1
        )
        z = torch.relu(self.lin1(z))
        z = torch.relu(self.lin2(z))
        if self.sigmoid:
            z = torch.sigmoid(self.lin3(z))
        else:
            z = self.lin3(z)
        return z.view(-1)


class SpatialGCNModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, sigmoid=True):
        super(SpatialGCNModel, self).__init__()

        self.encoder = GNNEncoder(input_channels, hidden_channels)
        self.decoder = GNNDecoder(hidden_channels, sigmoid)

    def forward(self, x, edge_index):
        z = self.encoder(x=x[:, 2:], edge_index=edge_index, pos=x[:, :2])
        return self.decoder(z, edge_index)
