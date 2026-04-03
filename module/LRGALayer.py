import torch
import torch.nn.functional as F
from torch import nn
from module.LRGA import LRGA


class LRGALayer(nn.Module):
    def __init__(self, n_input, hidden_gsa_dim, n_z, n_node):
        super(LRGALayer, self).__init__()
        self.n_node = n_node

        # LRGA encoder
        self.gsa1 = LRGA(n_input, hidden_gsa_dim[0], self.n_node)
        self.gsa2 = LRGA(hidden_gsa_dim[0], hidden_gsa_dim[1], self.n_node)
        self.gsa3 = LRGA(hidden_gsa_dim[1], hidden_gsa_dim[2], self.n_node)
        self.gsa4 = LRGA(hidden_gsa_dim[2], n_z, self.n_node)

    def forward(self, x, adj):
        r1 = self.gsa1(x, adj)
        r2 = self.gsa2(r1, adj)
        r3 = self.gsa3(r2, adj)
        r4 = self.gsa4(r3, adj)

        r = F.normalize(r4, p=2, dim=1)
        A_pred = torch.sigmoid(torch.matmul(r, r.t()))

        return A_pred, r, r4
