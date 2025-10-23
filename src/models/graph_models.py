"""Using for merged ab-ig graph
"""
import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn import Linear
from layers.conv import ConvBlock, GINEConvBlock
from layers.predictor import Predictor, MixtureDensityNetwork

from common import utils
from common.mapping_utils import get_activator, get_pooling_operator, get_activation_layer

class GraphModel(nn.Module):
    def __init__(self, conv_fn: ConvBlock, embed_size: int, hidden_size: int, out_channels: int,
                 num_layers: int, dropout: float = 0.0, use_edge_feat=True, activation='silu', **kargs):
        super().__init__()
        
        self.use_edge_feat = use_edge_feat
        self.hidden_size, self.output_size = hidden_size, out_channels
        self.lin1 = nn.Sequential(
            Linear(embed_size, hidden_size),
            get_activation_layer(activation)(),
        )
        self.input_block = conv_fn(hidden_size, hidden_size, use_edge_feat, dropout, **kargs)
        self.edge_linear = None
        if conv_fn == GINEConvBlock and use_edge_feat:
            self.edge_linear = Linear(kargs['edge_dim'], hidden_size)

        self.atom_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = conv_fn(hidden_size, hidden_size, dropout=dropout, **kargs)
            self.atom_convs.append(conv)
        
        self.activator = get_activator(activation)
        self.mdn_cls = str(kargs['mdn_cls']) if 'mdn_cls' in kargs else 'False'
        self.init_pooling_layer(hidden_size, **kargs)
        self.addon_functions(hidden_size, **kargs)
        self.node_predictor = Predictor(hidden_size, 3, hidden_size//2, activation=activation) if 'node_pred' in kargs and kargs['node_pred'] else None
        self.mdn = MixtureDensityNetwork(hidden_size, hidden_size, 10, dropout) if 'mdn' in kargs and kargs['mdn'] else None
        if self.mdn_cls == 'False':
            hidden_size = hidden_size if not kargs['pooling'].startswith('weightedcomponent') else hidden_size * 3
            self.predictor = Predictor(hidden_size, out_channels, hidden_size//2, activation=activation, num_layers=kargs['num_pred_layer'] if 'num_pred_layer' in kargs else 2)
        elif self.mdn_cls == 'True':
            self.predictor = nn.Linear(1, out_channels)
        else:
            self.predictor = None
        self.reset_parameters()

    def addon_functions(self, *args, **kargs):
        pass

    def init_pooling_layer(self, hidden_size, **kargs):
        pooling_name = kargs['pooling'] if 'pooling' in kargs else ''
        if pooling_name == 'afp':
            pooling_args = [hidden_size, hidden_size, hidden_size, 2]
        elif pooling_name in ['attention', 'set', 'setdecoder'] \
            or pooling_name.startswith('weighted') \
            or pooling_name.startswith('mdn'):
            pooling_args = [hidden_size]
        else:
            pooling_args = []
        self.pool = get_pooling_operator(pooling_name)(*pooling_args)   

    def reset_parameters(self):
        utils.reset(self.lin1)
        self.input_block.reset_parameters()
        if not self.edge_linear is None:
            self.edge_linear.reset_parameters()
        for conv in self.atom_convs:
            utils.reset(conv)
        if not self.predictor is None:
            self.predictor.reset_parameters()        
        if not self.mdn is None:
            utils.reset(self.mdn)
        if not self.node_predictor is None:
            utils.reset(self.node_predictor)
        utils.reset(self.pool)

    def convolution_step(self, g, x, edge_attr):
        """
            g: a batch of input graphs
            gs: a batch of supernoded graphs derived from g
            x: input features of nodes
            edge_attr: input features of edges
        """
        # Atom Embedding
        h = self.activator(self.lin1(x))
        if not self.edge_linear is None:
            edge_attr = self.edge_linear(edge_attr)

        if self.use_edge_feat:
            h = self.input_block(g, h, edge_attr)
        else:
            h = self.input_block(g, h)
        for conv in self.atom_convs:
            h = conv(g, h)  
        global_h = self.pool(g, h)
        global_h = self.activator(global_h)
        return global_h, h

    def get_global_embedding(self, g, x, edge_attr=None):
        global_h, _ = self.convolution_step(g, x, edge_attr)
        if self.predictor is None:
            return global_h
        return self.predictor.get_embedding(global_h)
    
    def forward(self, g, x, edge_attr=None):
        global_h, h = self.convolution_step(g, x, edge_attr)
        outputs = {'g_h': global_h, 'h_repr': h}
        out = self.predictor(global_h) if not self.predictor is None else global_h

        outputs.update({'logits': out})

        if not self.node_predictor is None:
            out_node = self.node_predictor(h)
            outputs['node_logits'] = out_node

        if not self.mdn is None:
            with g.local_scope():
                g.ndata['h_repr'] = h
                g.apply_edges(lambda edges: utils.mdn_message(edges, self.mdn))
                pi, mu, sigma, dist = g.edata['pi'], g.edata['mu'], g.edata['sigma'], g.edata['dist']
                outputs.update({'pi': pi, 'mu': mu, 'sigma': sigma, 'dist': dist})
        return outputs