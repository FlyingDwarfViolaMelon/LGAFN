import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import math


class LRGA(Module):
    def __init__(self, in_features, out_features, n_node, k=None):
        super(LRGA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_node = n_node
        
        self.k = k if k is not None else max(1, int(math.sqrt(n_node)))

        self.query = Parameter(torch.FloatTensor(in_features, out_features))
        self.key = Parameter(torch.FloatTensor(in_features, out_features))
        self.value = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.GCN_weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.gamma = nn.Parameter((torch.zeros(n_node, in_features)))

        self.E = Parameter(torch.FloatTensor(self.k, n_node))  
        self.F = Parameter(torch.FloatTensor(self.k, n_node)) 

        torch.nn.init.xavier_uniform_(self.query)
        torch.nn.init.xavier_uniform_(self.key)
        torch.nn.init.xavier_uniform_(self.value)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.GCN_weight)
        torch.nn.init.xavier_uniform_(self.E, gain=0.1)
        torch.nn.init.xavier_uniform_(self.F, gain=0.1)

    def forward(self, x, adj):
        proj_query = torch.mm(x, self.query)  # N * out_features
        proj_key = torch.mm(x, self.key)  # N * out_features
        proj_value = torch.mm(x, self.value)  # N * out_features

        # E: k x N, proj_key: N x out_features
        proj_key_E = torch.mm(self.E, proj_key)  # k * out_features

        # proj_query: N x out_features, proj_key_E^T: out_features x k
        energy = torch.mm(proj_query, torch.t(proj_key_E))  # N * k
        attention = F.softmax(energy, dim=1)  # N * k

        # F: k x N, proj_value: N x out_features
        proj_value_F = torch.mm(self.F, proj_value)  # k * out_features

        # attention: N x k, proj_value_F: k x out_features
        support = torch.mm(attention, proj_value_F)  # N * out_features

        out = torch.mm(support, self.weight)  # N * in_features
        out = self.gamma*out + torch.mm(adj, x)  # N * in_features
        output = torch.mm(out, self.GCN_weight)  # N * out_features
        output = F.leaky_relu(output, negative_slope=0.2)
        # output = torch.tanh(output)
        return output

