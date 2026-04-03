# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from module.LRGALayer import LRGALayer


class LRGALayer2(nn.Module):
    def __init__(self, n_input1, n_input2, hidden_gsa_dim, n_z, n_node):
        super(LRGALayer2, self).__init__()
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.hidden_gsa_dim = hidden_gsa_dim
        self.n_node = n_node
        self.n_z = n_z

        self.gsa_v1 = LRGALayer(self.n_input1, self.hidden_gsa_dim, self.n_z, self.n_node)
        self.gsa_v2 = LRGALayer(self.n_input2, self.hidden_gsa_dim, self.n_z, self.n_node)


    def forward(self, x1, x2, adj):
        A_pred1, r1, gsa_h1 = self.gsa_v1(x1, adj)
        A_pred2, r2, gsa_h2 = self.gsa_v2(x2, adj)
        return A_pred1, A_pred2, r1, r2, gsa_h1, gsa_h2
