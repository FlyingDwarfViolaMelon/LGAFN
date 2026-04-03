import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GCN(Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)

        output = F.leaky_relu(output, negative_slope=0.2)
        # output = torch.tanh(output)
        return output

