import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

torch.manual_seed(3047)


class Net(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16, cached=True)
        self.conv2 = GCNConv(16, 1, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        edge_attr = data.edge_attr if data.edge_attr is not None else None
        x = self.conv1.forward(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv2.forward(x, edge_index, edge_weight=edge_attr)

        x = torch.sigmoid(x)

        return x.squeeze(1).float()
