# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch import nn
from module.AE import AE

class AE2(Module):
    def __init__(self,
                 input_dim1,
                 input_dim2,
                 encoder_dim,
                 decoder_dim,
                 embedding_dim,
                 ):
        """
        param input_dim: the dimension of input data
        param embedding_dim: the dimension of embedding features
        param encoder_dim: the dimension of the encoder
        param decoder_dim: the dimension of the decoder
        """
        super(AE2, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embedding_dim = embedding_dim

        self.AE1 = AE(self.input_dim1, self.encoder_dim, self.decoder_dim, self.embedding_dim)
        self.AE2 = AE(self.input_dim2, self.encoder_dim, self.decoder_dim, self.embedding_dim)


    def forward(self, x1, x2):
        """

        :param x:
        :return:
        - x_bar: the reconstructed features
        - enc_h1: the 1st layers features of encoder
        - enc_h2: the 2nd layers features of encoder
        - enc_h3: the 3rd layers features of encoder
        - z: the embedding
        """
        x1_bar, enc1_h1, enc1_h2, enc1_h3, h1_embed = self.AE1(x1)
        x2_bar, enc2_h1, enc2_h2, enc2_h3, h2_embed = self.AE2(x2)

        return x1_bar, x2_bar, h1_embed, h2_embed, enc1_h1, enc1_h2, enc1_h3, enc2_h1, enc2_h2, enc2_h3
