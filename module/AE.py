# -*- coding: utf-8 -*-
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch import nn

class AE(Module):
    def __init__(self,
                 input_dim,
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
        super(AE, self).__init__()
        self.enc_1 = Linear(input_dim, encoder_dim[0])
        self.enc_2 = Linear(encoder_dim[0], encoder_dim[1])
        self.enc_3 = Linear(encoder_dim[1], encoder_dim[2])
        self.z_layer = Linear(encoder_dim[2], embedding_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.dec_1 = Linear(embedding_dim, decoder_dim[0])
        self.dec_2 = Linear(decoder_dim[0], decoder_dim[1])
        self.dec_3 = Linear(decoder_dim[1], decoder_dim[2])
        self.x_bar_layer = Linear(decoder_dim[2], input_dim)

    def forward(self, x):
        """

        :param x:
        :return:
        - x_bar: the reconstructed features
        - enc_h1: the 1st layers features of encoder
        - enc_h2: the 2nd layers features of encoder
        - enc_h3: the 3rd layers features of encoder
        - z: the embedding
        """
        enc_h1 = self.act(self.enc_1(x))
        enc_h2 = self.act(self.enc_2(enc_h1))
        enc_h3 = self.act(self.enc_3(enc_h2))
        h_embed = self.z_layer(enc_h3)

        dec_h1 = self.act(self.dec_1(h_embed))
        dec_h2 = self.act(self.dec_2(dec_h1))
        dec_h3 = self.act(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, h_embed
