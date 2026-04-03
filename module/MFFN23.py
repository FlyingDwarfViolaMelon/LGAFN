# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn import Linear
from module.GCN import GCN


class MFFN2(Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        """
        :param input_dim: the dimension of input features
        :param output_dim: the dimension of output features
        """
        super(MFFN2, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.mlp = Linear(self.input_dim1 + self.input_dim2, 2)

        
        self.interact1 = Linear(input_dim2, input_dim1)  
        self.interact2 = Linear(input_dim1, input_dim2)  

        self.gcn = GCN(self.input_dim1, output_dim)

    def forward(self, input_features1, input_features2, adj):
        """
        :param input_features1: input features 1
        :param input_features2: input features 2
        :param adj: the Symmetric graph matrix
        :return: gcn_output_features
        """
        
        interact_12 = F.relu(self.interact1(input_features2))  
        interact_21 = F.relu(self.interact2(input_features1))  

        
        enhanced_features1 = input_features1 + 0.1 * interact_12  
        enhanced_features2 = input_features2 + 0.1 * interact_21

        
        cat_features = torch.cat((enhanced_features1, enhanced_features2), 1)

        # linear
        mlp_features = self.mlp(cat_features)

        # activate
        activate_features = F.leaky_relu(mlp_features, negative_slope=0.2)

        # softmax
        softmax_features = F.softmax(activate_features, dim=1)

        # normalization
        normalize_features = F.normalize(softmax_features)

        # slice and transpose

        M_i_1 = normalize_features[:, 0].reshape(normalize_features.shape[0], 1)
        M_i_2 = normalize_features[:, 1].reshape(normalize_features.shape[0], 1)

        ones1 = torch.ones(1, input_features1.shape[1]).cuda()
        ones2 = torch.ones(1, input_features2.shape[1]).cuda()
        # ones1 = torch.ones(1, input_features1.shape[1])
        # ones2 = torch.ones(1, input_features2.shape[1])

        # calculate the weight matrix
        w_1 = torch.mm(M_i_1, ones1)
        w_2 = torch.mm(M_i_2, ones2)

        # fuse features
        fusion_features = w_1 * enhanced_features1 + w_2 * enhanced_features2
        # fusion_features = w_1 * input_features1 + w_2 * input_features2
        # fusion_features = input_features1 + input_features2

        # gcn
        gcn_output_features = self.gcn(fusion_features, adj)
        return gcn_output_features
