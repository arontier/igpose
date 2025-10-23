"""Using for merged ab-ig graph
"""
import sys
sys.path.append('..')
from torch import nn

# from layers.predictor import Predictor
from torch.nn import Linear, Sequential, Dropout
from common import utils
from common.mapping_utils import get_pooling_operator, get_activation_layer

class AbEpitope(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, out_channels: int, num_layers: int,
                 dropout: float = 0.0, activation='silu', **kargs):
        super().__init__()
        
        self.hidden_size, self.output_size = hidden_size, out_channels
        # self.predictor = Predictor(embed_size, out_channels,
        #                            hidden_size,
        #                            dropout=dropout,
        #                            activation=activation)
        self.predictor = Sequential(
            Linear(embed_size, hidden_size),
            get_activation_layer(activation)(),
            Dropout(0.65),
            Linear(hidden_size, 250),
            get_activation_layer(activation)(),
            Dropout(0.6),
            Linear(250, 50),
            get_activation_layer(activation)(),
            Dropout(0.5),
            Linear(50, self.output_size)
        )
        self.init_pooling_layer(embed_size)
        self.reset_parameters()

    def reset_parameters(self):
        utils.reset(self.predictor)
        utils.reset(self.pool)

    def init_pooling_layer(self, hidden_size, **kargs):
        pooling_name = kargs['pooling'] if 'pooling' in kargs else ''
        if pooling_name in ['attention', 'set', 'setdecoder'] or pooling_name.startswith('weighted'):
            pooling_args = [hidden_size]
        else:
            pooling_args = []
        self.pool = get_pooling_operator(pooling_name)(*pooling_args)   

    def forward(self, g, x, edge_attr=None):
        global_h = self.pool(g, x)
        out = self.predictor(global_h)
        outputs = {'logits': out}
        return outputs