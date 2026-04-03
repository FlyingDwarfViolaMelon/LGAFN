# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from module.LRGALayer2 import LRGALayer2
from module.LRGA import LRGA
from module.AE2 import AE2
from module.MFFN23 import MFFN2
from module.fusion_layer import FusionLayer2


class LGAFN(nn.Module):
    def __init__(self, n_input1, n_input2, hidden_gsa_dim, encoder_dim, decoder_dim, n_z, n_clusters, n_node, v=1):
        super(LGAFN, self).__init__()

        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.hidden_gsa_dim = hidden_gsa_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_z = n_z
        self.n_clusters = n_clusters
        self.n_node = n_node

        self.gsa = LRGALayer2(n_input1=self.n_input1,
                             n_input2=self.n_input2,
                             hidden_gsa_dim=self.hidden_gsa_dim,
                             n_z=self.n_z,
                             n_node=self.n_node)

        self.AE = AE2(input_dim1=self.n_input1, input_dim2=self.n_input2,
                      encoder_dim=self.encoder_dim, decoder_dim=self.decoder_dim,
                      embedding_dim=self.n_z)

        # View 1 fusion
        self.v1_mffn1 = LRGA(self.n_input1, hidden_gsa_dim[0], self.n_node)
        self.v1_mffn2 = MFFN2(hidden_gsa_dim[0], hidden_gsa_dim[0], hidden_gsa_dim[1])
        self.v1_mffn3 = MFFN2(hidden_gsa_dim[1], hidden_gsa_dim[1], hidden_gsa_dim[2])
        self.v1_mffn4 = MFFN2(hidden_gsa_dim[2], hidden_gsa_dim[2], self.n_z)

        self.v1_fusion = MFFN2(self.n_z, self.n_z, n_clusters)

        # View 2 fusion
        self.v2_mffn1 = LRGA(self.n_input2, hidden_gsa_dim[0], self.n_node)
        self.v2_mffn2 = MFFN2(hidden_gsa_dim[0], hidden_gsa_dim[0], hidden_gsa_dim[1])
        self.v2_mffn3 = MFFN2(hidden_gsa_dim[1], hidden_gsa_dim[1], hidden_gsa_dim[2])
        self.v2_mffn4 = MFFN2(hidden_gsa_dim[2], hidden_gsa_dim[2], self.n_z)

        self.v2_fusion = MFFN2(self.n_z, self.n_z, n_clusters)

        # cluster layer z1
        self.cluster_layer_z1 = Parameter(torch.Tensor(self.n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer_z1.data)

        # cluster layer z2
        self.cluster_layer_z2 = Parameter(torch.Tensor(self.n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer_z2.data)

        # cluster layer h1
        self.cluster_layer_h1 = Parameter(torch.Tensor(self.n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer_h1.data)

        # cluster layer h2
        self.cluster_layer_h2 = Parameter(torch.Tensor(self.n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer_h2.data)

        # define the weights for fusion matrix
        self.fusion_module = FusionLayer2(num_views=2)

        self.v = v

    def forward(self, x1, x2, adj):
        A_pred1, A_pred2, r1, r2, gsa_h1, gsa_h2 = self.gsa(x1, x2, adj)

        q_z1 = 1.0 / (1.0 + torch.sum(torch.pow(r1.unsqueeze(1) - self.cluster_layer_z1, 2), 2) / self.v)
        q_z1 = q_z1.pow((self.v + 1.0) / 2.0)
        q_z1 = (q_z1.t() / torch.sum(q_z1, 1)).t()

        q_z2 = 1.0 / (1.0 + torch.sum(torch.pow(r2.unsqueeze(1) - self.cluster_layer_z2, 2), 2) / self.v)
        q_z2 = q_z2.pow((self.v + 1.0) / 2.0)
        q_z2 = (q_z2.t() / torch.sum(q_z2, 1)).t()

        x1_bar, x2_bar, h1, h2, v1_h1, v1_h2, v1_h3, v2_h1, v2_h2, v2_h3 = self.AE(x1, x2)

        q_h1 = 1.0 / (1.0 + torch.sum(torch.pow(h1.unsqueeze(1) - self.cluster_layer_h1, 2), 2) / self.v)
        q_h1 = q_h1.pow((self.v + 1.0) / 2.0)
        q_h1 = (q_h1.t() / torch.sum(q_h1, 1)).t()

        q_h2 = 1.0 / (1.0 + torch.sum(torch.pow(h2.unsqueeze(1) - self.cluster_layer_h2, 2), 2) / self.v)
        q_h2 = q_h2.pow((self.v + 1.0) / 2.0)
        q_h2 = (q_h2.t() / torch.sum(q_h2, 1)).t()

        # View 1 fusion
        v1_z1 = self.v1_mffn1(x1, adj)
        v1_z2 = self.v1_mffn2(v1_h1, v1_z1, adj)
        v1_z3 = self.v1_mffn3(v1_h2, v1_z2, adj)
        v1_z4 = self.v1_mffn4(v1_h3, v1_z3, adj)
        v1_zlast = self.v1_fusion(h1, v1_z4, adj)
        predict_v1 = F.softmax(v1_zlast, dim=1)

        # View 2 fusion
        v2_z1 = self.v2_mffn1(x2, adj)
        v2_z2 = self.v2_mffn2(v2_h1, v2_z1, adj)
        v2_z3 = self.v2_mffn3(v2_h2, v2_z2, adj)
        v2_z4 = self.v2_mffn4(v2_h3, v2_z3, adj)
        v2_zlast = self.v2_fusion(h2, v2_z4, adj)
        predict_v2 = F.softmax(v2_zlast, dim=1)

        # combine the fusion matrix v1_zlast, v2_zlast
        z_last = self.fusion_module(v1_zlast, v2_zlast)
        predict = F.softmax(z_last, dim=1)

        # return x1_bar, x2_bar, A_pred1, A_pred2, predict, q_v1, q_v2, z_last
        return x1_bar, x2_bar, A_pred1, A_pred2, predict, predict_v1, predict_v2, q_z1, q_z2, q_h1, q_h2, z_last
