import sys
sys.path.append('..')
from torch import nn
from torch_geometric.nn.models.attentive_fp import GATEConv
from torch_geometric.nn.conv import GATConv
from layers.fastegnn import FastEGNN
from layers.conv import ConvBlock
from layers.egnn_pyg import EGNNConv

class ConvBlockPyG(ConvBlock):
    def forward(self, h, edge_index, edge_attr=None, **kargs):
        h_prime = self.conv(h, edge_index, edge_attr)
        h_prime = h_prime.squeeze(1)
        h = self.aggregate(h_prime, h)
        return {'h': h}

class GATConvBlockPyG(ConvBlockPyG):       
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        if use_edge_feat:
            self.conv = GATEConv(input_size, hidden_size, dropout=dropout, edge_dim=kargs['edge_dim'])
        else:
            self.conv = GATConv(input_size, hidden_size, heads=1, dropout=dropout, residual=False)

class EGNNConvBlockPyG(ConvBlockPyG):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        if use_edge_feat:
            self.conv = EGNNConv(input_size, hidden_size, hidden_size, edge_feat_size=kargs['edge_dim'])
        else:
            self.conv = EGNNConv(input_size, hidden_size, hidden_size)

    def forward(self, h, edge_index, edge_attr=None, coord=None, **kargs):
        h_prime, new_coord = self.conv(h, edge_index, edge_attr, coord)
        h_prime = h_prime.squeeze(1)
        h = self.aggregate(h_prime, h)
        return {'h': h, 'coord': new_coord}
    
class FastEGNNConvBlockPyG(ConvBlockPyG):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, residual=False, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        edge_size = 0 if not use_edge_feat else kargs['edge_dim']
        self.residual = residual
        self.conv = FastEGNN(input_size, hidden_size, edge_size, hidden_size, 3, act_fn=nn.SiLU(),
                             residual=residual, attention=False, normalize=False, tanh=False)

    def forward(self, h, edge_index, edge_attr=None, coord=None, virtual_node_feat=None, virtual_coord=None, data_batch=None):
        h_prime, real_node_loc, v_h_prime, virtual_node_loc = self.conv(h, edge_index, coord, virtual_coord, virtual_node_feat, data_batch, edge_attr)
        if not self.residual:
            new_h = self.aggregate(h_prime, h)
            # apply gru to virtual nodes
            v_shape = v_h_prime.shape
            v_h_prime = v_h_prime.permute(0, 2, 1).reshape(-1, v_h_prime.shape[1])
            virtual_node_feat = virtual_node_feat.permute(0, 2, 1).reshape(-1, virtual_node_feat.shape[1])
            new_v_h = self.aggregate(v_h_prime, virtual_node_feat)
            new_v_h = new_v_h.view(v_shape[0], v_shape[2], v_shape[1]).permute(0, 2, 1) # B, H, C
        else:
            new_h, new_v_h = h_prime, v_h_prime
        return {'h': new_h, 'coord': real_node_loc, 'virtual_coord': virtual_node_loc, 'virtual_node_feat': new_v_h}