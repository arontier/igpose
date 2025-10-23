"""Using for merged ab-ig graph
"""
import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.graph_models import GraphModel
from layers.pooling_pyg import SumPoolingPyg, WeightAndSumPyG, AttentiveFPPoolingPyG

def mdn_message(x_j, x_i, mdn):
    edge_h = torch.cat([x_j, x_i], dim=-1)
    dist = torch.norm(x_j - x_i, p=2, dim=1)
    pi, mu, sigma = mdn(edge_h)
    return {'pi': pi, 'mu': mu, 'sigma': sigma, 'dist': dist}

class GraphModelPyG(GraphModel):

    def addon_functions(self, hidden_size, **kargs):
        if 'virtual_channels' in kargs:
            self.virtual_channels = kargs['virtual_channels'] 
            self.virtual_node_feat = nn.Parameter(data=torch.randn(size=(1, hidden_size, self.virtual_channels)), requires_grad=True)

    def init_pooling_layer(self, hidden_size, **kargs):
        pooling_name = kargs['pooling'] if 'pooling' in kargs else ''
        if pooling_name == 'sum':
            self.pool = SumPoolingPyg() 
        elif pooling_name == 'afp':
            self.pool = AttentiveFPPoolingPyG(hidden_size, hidden_size, hidden_size)
        else:
            self.pool = WeightAndSumPyG(hidden_size)

    def convolution_step(self, g):
        # Atom Embedding
        x, edge_index, coord = g.x, g.edge_index, g.pos
        edge_attr = g.edge_attr if self.use_edge_feat else None
        
        input_h = self.activator(self.lin1(x))

        kargs = {}
        if hasattr(self, 'virtual_channels'):
            virtual_node_feat = self.virtual_node_feat.repeat(g.batch_size, 1, 1)
            virtual_node_loc = global_mean_pool(coord, g.batch).view(g.batch_size, 3, 1).repeat(1, 1, self.virtual_channels)
            kargs.update({
                'virtual_coord': virtual_node_loc,
                'virtual_node_feat': virtual_node_feat,
                'data_batch': g.batch
            })
        out = self.input_block(input_h, edge_index, edge_attr=edge_attr, coord=coord, **kargs)
        coord = out['coord'] if 'coord' in out else None 
        for conv in self.atom_convs:
            if 'virtual_coord' in out:
                kargs = {
                    'virtual_coord': out['virtual_coord'],
                    'virtual_node_feat': out['virtual_node_feat'],
                    'data_batch': g.batch
                }
            out = conv(out['h'], edge_index, coord=coord, **kargs)
            coord = out['coord'] if 'coord' in out else None 
        global_h = self.pool(g, out['h'])
        return self.activator(global_h), out['h']

    def forward(self, g):
        global_h, h = self.convolution_step(g)
        outputs = {'g_h': global_h, 'h_repr': h}
        if not self.predictor is None:
            out = self.predictor(global_h)
            outputs.update({'logits': out})

        if not self.node_predictor is None:
            out_node = self.node_predictor(h)
            outputs['node_logits'] = out_node

        if not self.mdn is None:
            src, dst = g.edge_index
            x_j, x_i = h[src], h[dst]
            mdn_outputs = mdn_message(x_j, x_i, self.mdn)
            outputs.update(mdn_outputs)
        return outputs