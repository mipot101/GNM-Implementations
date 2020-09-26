import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class twoLayerConvolutionalNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, activation=lambda x:F.softmax(x, dim=1)):
        super(twoLayerConvolutionalNetwork, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.activation = activation

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return self.activation(x)
    

class DefaultNetH(torch.nn.Module):
    def __init__(self, x_features):
        super(DefaultNetH, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=x_features, out_features=128, bias=False) 
        self.linear2 = torch.nn.Linear(in_features=128, out_features=64, bias=False)
        self.linear3 = torch.nn.Linear(in_features=64, out_features=1, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear3(x)

        return F.relu(x)


class NetR(torch.nn.Module):
    def __init__(self, modelh, h_out_features, num_classes):
        super(NetR, self).__init__()
        self.modelh = modelh
        self.linear1 = torch.nn.Linear(h_out_features, 1)
        self.linear2 = torch.nn.Linear(num_classes, 1, bias=False)
        self.linear3 = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x, y):
        # y should be one-hot encoded
        hx = self.modelh(x)
        hx = self.linear1(hx) # gamma.T @ h(x) + alpha

        y = self.linear2(y) # phi * y
        var = torch.cat((hx,y), dim = 1)
        var = self.linear3(var) # a * (gamma.T @ h(x) + alpha) + b * (phi * y)
        return torch.sigmoid(var) 