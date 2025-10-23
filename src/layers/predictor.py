"""
===============================================================================
File Name   : predictor.py
Author      : Alex Bui
Created     : 2024-11-13
Description : Include prediction heads
===============================================================================
"""
import sys
sys.path.append('..')
from torch import nn

from torch.nn import Linear, Sequential, Dropout
import torch.nn.functional as F
from common.utils import reset
from common.mapping_utils import get_activation_layer

class Predictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.1, activation='silu', num_layers=2):
        # output_size should be 1 of regression and 2 for classification
        super().__init__()
        
        if num_layers == 2:
            self.predictor = Sequential(
                Linear(input_size, hidden_size),
                get_activation_layer(activation)(),
                Dropout(dropout),
                Linear(hidden_size, output_size)
            )
        else:
            layers = [Linear(input_size, hidden_size), get_activation_layer(activation)()]
            next_size = hidden_size // 2
            for _ in range(num_layers-2):
                layers.append(Linear(hidden_size, next_size))
                hidden_size = next_size
            layers.extend([Dropout(dropout), Linear(hidden_size, output_size)])
            self.predictor = Sequential(*layers)

    def reset_parameters(self):
        reset(self.predictor)

    def get_embedding(self, x):
        # 1) first Linear
        x = self.predictor[0](x)
        # 2) activation —> capture it
        act_out = self.predictor[1](x)
        return act_out

    def forward(self, input):
        return self.predictor(input)
    
class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(input_size*2, hidden_size), 
								nn.BatchNorm1d(hidden_size), 
								nn.ELU(), 
                                nn.Dropout(p=dropout))
        self.z_pi = nn.Linear(hidden_size, output_size)
        self.z_sigma = nn.Linear(hidden_size, output_size)
        self.z_mu = nn.Linear(hidden_size, output_size)

    def forward(self, h):
        """
        input: h (B x E x D)
        output: pi, sigma, mu
        """
        h = self.MLP(h)
        pi = F.softmax(self.z_pi(h), -1)
        sigma = F.elu(self.z_sigma(h)) + 1.1
        mu = F.elu(self.z_mu(h)) + 1.
        return pi, mu, sigma
    
          
