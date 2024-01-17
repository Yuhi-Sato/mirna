import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv

torch.manual_seed(3047)


class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_layers):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()

        for l in range(num_layers):
            self.layers.append(GCN2Conv(num_node_features, alpha=0.1, cached=True))

        self.layers.append(GCNConv(num_node_features, 16, cached=True))
        self.layers.append(GCNConv(16, out_channels=1, cached=True))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # NOTE: edgeに重みがある場合はedge_attrを使う.
        edge_attr = data.edge_attr if data.edge_attr is not None else None
        for l in range(len(self.layers) - 2):
            x = self.layers[l].forward(
                x, x_0=data.x, edge_index=edge_index, edge_weight=edge_attr
            )
            x = F.relu(x)

        x = self.layers[-2].forward(x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.layers[-1].forward(x, edge_index=edge_index, edge_weight=edge_attr)

        x = torch.sigmoid(x)

        return x.squeeze(1).float()
