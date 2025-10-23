import sys
sys.path.append('..')

import torch
from torch.nn import Module, Sequential, Linear, Dropout

from models.graph_models import GraphModel
from layers.conv import ConvBlock

class NDGraphModel(Module):
    def __init__(self, conv_fn: ConvBlock, embed_size: int, hidden_size: int, out_channels: int,
                 num_layers: int, dropout: float = 0.0, use_edge_feat=True, activation='silu',
                 **kargs):
        super().__init__()
        self.use_edge_feat = use_edge_feat
        self.graph_model1 = GraphModel(conv_fn, embed_size, hidden_size, out_channels, num_layers, dropout,
                                       use_edge_feat, activation, **kargs)
        self.graph_model2 = GraphModel(conv_fn, embed_size, hidden_size, out_channels, num_layers, dropout,
                                       use_edge_feat, activation, **kargs)
        self.predictor = Sequential(
            Linear(hidden_size*2, hidden_size),
            Dropout(dropout),
            Linear(hidden_size, out_channels)
        )
    
    def forward(self, g1, g2):
        output1 = self.graph_model1(g1, g1.ndata['feat'], edge_attr=g1.edata['attr'] if self.use_edge_feat else None)
        output2 = self.graph_model2(g2, g2.ndata['feat'], edge_attr=g2.edata['attr'] if self.use_edge_feat else None)
        g_h = torch.cat([output1['g_h'], output2['g_h']], dim=1)
        out = self.predictor(g_h)
        return {'logits': out}