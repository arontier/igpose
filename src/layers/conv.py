import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import GRUCell, Linear

import dgl
from dgl import function as fn
from dgl.nn.pytorch.conv import GATConv, EdgeGATConv, GINConv, GINEConv, GATv2Conv, HGTConv, EGNNConv
from dgl.nn.functional import edge_softmax

from layers.egnn import EGNNNormConv, EGNNFroConv
from layers.etgnn import ETGNNConv
from common import utils as ut

def copy_v(in_key, out_key):
    def copy_v_to(edges):
        return {out_key: edges.dst[in_key]}  
    return copy_v_to

class GATEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, **kargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lin1 = Linear(in_channels + kargs['edge_dim'], out_channels, bias=False)
        self.lin2 = Linear(in_channels, out_channels, bias=False)

        # Attention weights
        self.att_l = nn.Parameter(torch.FloatTensor(size=(1, out_channels)))
        self.att_r = nn.Parameter(torch.FloatTensor(size=(1, in_channels)))

        self.bias = nn.Parameter(torch.FloatTensor(size=(out_channels,)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.bias)

    def forward(self, g, x, edge_attr):
        with g.local_scope():
            # Prepare node and edge features
            g.ndata['x'] = x
            g.ndata['x_transformed'] = self.lin2(x)

            # graph is symmetric => copy_u can work for both
            # compute score for destination
            ar = (x * self.att_r).sum(-1).unsqueeze(-1)
            g.ndata['ar'] = ar
            g.apply_edges(copy_v('ar', 'er')) 

            # compute score for incoming edges
            g.apply_edges(fn.copy_u('x', 'x_e'))
            f_srt = self.lin1(torch.cat([g.edata['x_e'], edge_attr], dim=-1))
            el = (f_srt * self.att_l).sum(-1).unsqueeze(-1)

            # compute attention score
            alpha = F.leaky_relu(g.edata['er'] + el)

            # compute attention score
            g.edata['alpha'] = self.dropout(edge_softmax(g, alpha))

            # Message passing using DGL's native functions
            g.update_all(fn.u_mul_e('x_transformed', 'alpha', 'm'), fn.sum('m', 'h'))

            # Final output with bias
            out = g.ndata['h'] + self.bias
            return out

class GATEv2Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, **kargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lin1 = Linear(in_channels + kargs['edge_dim'], out_channels, bias=False)
        self.lin2 = Linear(in_channels, out_channels, bias=False)

        # Attention weights
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, out_channels+in_channels)))
        self.bias = nn.Parameter(torch.FloatTensor(size=(out_channels,)))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.bias)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, edge_attr: torch.Tensor):
        with g.local_scope():
            # Prepare node and edge features
            g.ndata['x'] = x
            g.ndata['x_transformed'] = self.lin2(x) # v

            # compute score for incoming edges
            g.apply_edges(fn.copy_u('x', 'x_l'))
            g.apply_edges(copy_v('x', 'x_r'))
            f_srt = self.lin1(torch.cat([g.edata['x_l'], edge_attr], dim=-1))
            f_dst = g.edata['x_r']

            e = F.leaky_relu(torch.cat([f_srt, f_dst], dim=-1))

            # compute attention score
            alpha = (e * self.attn).sum(dim=-1)
            g.edata['alpha'] = self.dropout(edge_softmax(g, alpha))

            # Message passing using DGL's native functions
            g.update_all(fn.u_mul_e('x_transformed', 'alpha', 'm'), fn.sum('m', 'h'))

            # Final output with bias
            out = g.ndata['h'] + self.bias
            return out

class ConvBlock(nn.Module):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, activation=F.relu, **kargs):
        super().__init__()
        self.use_edge_feat = use_edge_feat
        self.dropout = dropout
        self.conv = None
        self.aggregation = None
        if 'aggregation_fn' in kargs and kargs['aggregation_fn'] == 'gru':
            self.gru_input_size = input_size if 'aggregation_mode' in kargs and kargs['aggregation_mode'] == 'single' else hidden_size + input_size
            # x0: edge_dim (~hidden_dim) + input_dim, h0: input_size
            self.aggregation = GRUCell(self.gru_input_size, input_size) 
        self.layer_norm = nn.LayerNorm(hidden_size) if 'layer_norm' in kargs and kargs['layer_norm'] else None
        self.activation = activation

    def reset_parameters(self):
        ut.reset(self.conv)
        if not self.aggregation is None:
            ut.reset(self.aggregation)
    
    def execute_convolution(self, g, h, **kargs):
        if self.use_edge_feat:
            h_prime = self.conv(g, h, kargs['edge_attr'])
        else:
            h_prime = self.conv(g, h)
        return h_prime

    def aggregate(self, h_prime, h):
        if not self.aggregation is None:
            h_prime = F.dropout(F.elu(h_prime), p=self.dropout, training=self.training)
            if self.gru_input_size != h_prime.shape[-1]:
                x_g_input = torch.cat([h, h_prime], dim=-1)
            else:
                x_g_input = h_prime
            h = self.activation(self.aggregation(x_g_input, h))
        else:
            h = h_prime
        return h

    def forward(self, g, h, edge_attr=None, **kargs):
        h_prime = self.execute_convolution(g, h, edge_attr=edge_attr, **kargs)
        h_prime = h_prime.squeeze(1)
        h = self.aggregate(h_prime, h)
        if not self.layer_norm is None:
            h = self.layer_norm(h)
        return h

class GATConvBlock(ConvBlock):       
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        edge_dim = kargs['edge_dim']
        edge_fn = kargs['edge_fn']
        if use_edge_feat:
            if edge_fn == 'egat':
                self.conv = EdgeGATConv(input_size, edge_dim, hidden_size, num_heads=1, feat_drop=dropout, attn_drop=dropout, residual=False)   
            elif edge_fn.startswith('gat'):
                edge_types = None if 'edge_types' not in kargs else kargs['edge_types']
                if edge_fn[-2:] == 'v2':
                    self.conv = GATEv2Conv(input_size, hidden_size, dropout=dropout, edge_dim=kargs['edge_dim'], edge_types=edge_types)
                else:
                    self.conv = GATEConv(input_size, hidden_size, dropout=dropout, edge_dim=kargs['edge_dim'], edge_types=edge_types)
        else:
            if edge_fn[-2:] == 'v2':
                self.conv = GATv2Conv(input_size, hidden_size, num_heads=1, feat_drop=dropout, attn_drop=dropout, residual=False)
            else:
                self.conv = GATConv(input_size, hidden_size, num_heads=1, feat_drop=dropout, attn_drop=dropout, residual=False)

class GINEConvBlock(ConvBlock):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        apply_fn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        if use_edge_feat:
            self.conv = GINEConv(apply_fn, learn_eps=False)
        else:
            self.conv = GINConv(apply_fn, learn_eps=False)

    def reset_parameters(self):
        ut.reset(self.conv.apply_func)
        if not self.aggregation is None:
            self.aggregation.reset_parameters()

class HGTConvBlock(ConvBlock):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        if use_edge_feat:
            self.conv = EdgeGATConv(input_size, kargs['edge_dim'], hidden_size, num_heads=1, feat_drop=dropout, attn_drop=dropout, residual=False)   
        else:
            self.conv = HGTConv(input_size, hidden_size, num_heads=1, num_ntypes=1, num_etypes=1, dropout=dropout, use_norm=True)

    def reset_parameters(self):
        ut.reset(self.conv)
        if not self.aggregation is None:
            self.aggregation.reset_parameters()
    
    def execute_convolution(self, g, h, **kargs):
        if self.use_edge_feat:
            h_prime = self.conv(g, h, kargs['edge_attr'])
        else:
            ntype = torch.zeros((g.num_nodes(),), dtype=torch.int32, device=g.device)
            etype = torch.zeros((g.num_edges(),), dtype=torch.int32, device=g.device)
            h_prime = self.conv(g, h, ntype=ntype, etype=etype)
        return h_prime
    
class EGNNConvBlock(ConvBlock):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        self.conv = EGNNConv(input_size, hidden_size, hidden_size, edge_feat_size=kargs['edge_dim'] if use_edge_feat else 0)

    def reset_parameters(self):
        ut.reset(self.conv)
        if not self.aggregation is None:
            self.aggregation.reset_parameters()

    def execute_convolution(self, g, h, **kargs):
        coord = g.ndata.pop('coord')
        if self.use_edge_feat:
            h_prime, x = self.conv(g, h, coord, kargs['edge_attr'])
        else:
            h_prime, x = self.conv(g, h, coord)
        g.ndata['coord'] = x
        return h_prime
    
class EGNNNormConvBlock(EGNNConvBlock):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        self.conv = EGNNNormConv(input_size, hidden_size, hidden_size, edge_feat_size=kargs['edge_dim'] if use_edge_feat else 0)

class EGNNFroConvBlock(EGNNConvBlock):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        self.conv = EGNNFroConv(input_size, hidden_size, hidden_size, edge_feat_size=kargs['edge_dim'] if use_edge_feat else 0)

class ETGNNConvBlock(EGNNConvBlock):
    def __init__(self, input_size, hidden_size, use_edge_feat=False, dropout=0.1, **kargs):
        super().__init__(input_size, hidden_size, use_edge_feat, dropout, **kargs)
        self.conv = ETGNNConv(input_size, hidden_size, hidden_size, edge_feat_size=kargs['edge_dim'] if use_edge_feat else 0)