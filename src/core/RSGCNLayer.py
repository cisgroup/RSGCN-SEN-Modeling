import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn as nn


class RSGCNLayer(MessagePassing):
    def __init__(self, coors, in_channels, out_channels, hidden_size, dropout=0):
        """
        coors - dimension of positional descriptors (e.g. 2 for 2D images)
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        hidden_size - number of the inner convolutions
        dropout - dropout rate after the layer
        """
        super(RSGCNLayer, self).__init__(aggr="add")
        self.dropout = dropout
        self.lin_pos = torch.nn.Linear(coors, 2 * hidden_size)
        self.lin_pos_2 = torch.nn.Linear(2 * hidden_size, 2 * hidden_size)

        self.lin_out = torch.nn.Linear(2 * hidden_size, out_channels)

        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=1,
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=1,
        )

        self.conv_lin = torch.nn.Linear(int(11 * 11 * 32), 2 * hidden_size)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.in_channels = in_channels

    def forward(self, x, pos, edge_index, node_region):
        """
        x - feature matrix of the whole graph [num_nodes, label_dim]
        pos - node position matrix [num_nodes, coors]
        edge_index - graph connectivity [2, num_edges]
        """
        # edge_index, _ = add_self_loops(
        #     edge_index, num_nodes=x.size(0)
        # )  # num_edges = num_edges + num_nodes

        return self.propagate(
            edge_index=edge_index,
            x=x,
            pos=pos,
            region=torch.reshape(node_region, (node_region.size(0), -1)),
            aggr="add",
        )  # [N, out_channels, label_dim]

    def message(self, pos_i, pos_j, x_j, region_i, region_j):
        """
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        x_j [num_edges, label_dim]
        """

        relative_pos = pos_j - pos_i  # [n_edges, hidden_size * in_channels]
        pos_feature = F.relu(
            self.lin_pos_2(F.relu(self.lin_pos(relative_pos)))
        )  # [n_edges, hidden_size * in_channels]

        # [n_edges, in_channels, ...] * [n_edges, in_channels, 1]

        region_j = torch.reshape(region_j, (region_j.size(0), 1, 50, 50))
        region_i = torch.reshape(region_i, (region_i.size(0), 1, 50, 50))
        region_spatial = region_j - region_i
        region_spatial = F.relu(self.conv(region_spatial))
        region_spatial = self.max_pool(region_spatial)
        region_spatial = F.relu(self.conv2(region_spatial))
        region_spatial = self.max_pool(region_spatial)
        region_spatial = self.flatten(region_spatial)
        regional_spatial = F.relu(self.conv_lin(region_spatial))

        n_edges = pos_feature.size(0)

        pos_feature = pos_feature.reshape(n_edges, self.in_channels, -1)
        regional_spatial = regional_spatial.reshape(n_edges, self.in_channels, -1)
        x_j = x_j.reshape(n_edges, self.in_channels, -1)

        result = pos_feature * x_j * regional_spatial
        return result.view(n_edges, -1)

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        aggr_out = F.relu(aggr_out)
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)

        return aggr_out
