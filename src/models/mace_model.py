import sys
sys.path.append('..')
import torch
import torch.nn as nn
from mace_layer import MACE_layer

from layers.predictor import Predictor, MixtureDensityNetwork
from models.graph_models_pyg import mdn_message
from layers.pooling_pyg import SumPoolingPyg, WeightAndSumPyG, WeightAndMeanPyG, AttentiveFPPoolingPyG, WeightAndSumInterfacePyG

class MaceModel(nn.Module):
    def __init__(self, embed_size, hidden_size, out_channels, num_layers=2, dropout=0.1, num_neighbors=10, **kargs):
        super().__init__()
        self.input_linear = nn.Linear(embed_size, hidden_size)
        self.node_attr_linear = nn.Linear(embed_size, hidden_size)
        self.convs = nn.ModuleList()
        hidden_scalar = hidden_size // 4
        hidden_irreps = f"{hidden_scalar}x0e + {hidden_scalar}x1o"
        for i in range(num_layers):
            if i == 0:
                node_feat_irreps = f"{hidden_size}x0e + 1x1e"
            else:
                node_feat_irreps = hidden_irreps
            layer = MACE_layer(
                max_ell=2,
                correlation=3,
                n_dims_in=hidden_size,
                hidden_irreps=hidden_irreps,
                node_feats_irreps=node_feat_irreps,
                edge_feats_irreps=f"{kargs['edge_dim']}x0e",
                avg_num_neighbors=num_neighbors,
                use_sc=True,
            )
            self.convs.append(layer)

        self.predictor = Predictor(hidden_size, out_channels, hidden_size//2, dropout=dropout, activation='silu')
        self.mdn = MixtureDensityNetwork(hidden_size, hidden_size, 10, dropout) if 'mdn' in kargs and kargs['mdn'] else None
        self.node_predictor = Predictor(hidden_size, 3, hidden_size//2, activation='silu') 
        self.init_pooling_layer(hidden_size, **kargs)

    def init_pooling_layer(self, hidden_size, **kargs):
        pooling_name = kargs['pooling'] if 'pooling' in kargs else ''
        if pooling_name == 'sum':
            self.pool = SumPoolingPyg() 
        elif pooling_name == 'afp':
            self.pool = AttentiveFPPoolingPyG(hidden_size, hidden_size, hidden_size)
        elif pooling_name == 'weightedmean':
            self.pool = WeightAndMeanPyG(hidden_size)
        elif pooling_name == 'weightedinterface':
            self.pool = WeightAndSumInterfacePyG(hidden_size)
        else:
            self.pool = WeightAndSumPyG(hidden_size)

    def forward(self, g):
        node_attrs, edge_index, coord, edge_feats = g.x, g.edge_index, g.pos, g.edge_attr
        edge_index = edge_index.to(torch.int64)
        src, dst = edge_index
        vectors = coord[src] - coord[dst]
        node_feats = self.input_linear(node_attrs)
        node_feats = torch.cat([node_feats, coord], dim=-1)
        node_attrs = self.node_attr_linear(node_attrs)
        for i, conv in enumerate(self.convs):
            node_feats = conv(vectors, node_feats, node_attrs, edge_feats, edge_index)
        
        global_h = self.pool(g, node_feats)
        outputs = {'g_h': global_h, 'h_repr': node_feats}
        out_node = self.node_predictor(node_feats)
        outputs['node_logits'] = out_node
        x_j, x_i = node_feats[src], node_feats[dst]
        mdn_outputs = mdn_message(x_j, x_i, self.mdn)
        outputs.update(mdn_outputs)
        out = self.predictor(global_h)
        outputs.update({'logits': out})
        return outputs