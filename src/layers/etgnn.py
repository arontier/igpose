"""Torch Module for E(n) Equivariant Graph Convolutional Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class EGNNv2Conv(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(ETGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(2*in_size + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )        

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["radial"]], dim=-1
            )

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat

            # coordinate feature
            graph.ndata["x"] = coord_feat
            
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat

            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = (
                graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
            )
            # normalize coordinate difference
            graph.edata["x_diff"] = graph.edata["x_diff"] / (
                graph.edata["radial"].sqrt() + 1e-30
            )

            graph.apply_edges(self.message)
            # edge feature
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))            
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            
            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            # sum instead of concat
            h = node_feat + self.node_mlp(h_neigh)
            x = coord_feat + x_neigh

            return h, x

class ParameterizedTrick(nn.Module):
    def __init__(self, bias=0., beta=1.):
        self.bias = bias
        self.beta = beta
    
    def forward(self, input):
        random_noise = torch.rand(input.size()).to(input.device)
        if self.bias > 0. and self.bias < 0.5:
            r = 1 - self.bias - self.bias
            random_noise = r * random_noise + self.bias
        gate_inputs = torch.log(random_noise) - torch.log(1 - random_noise)
        gate_inputs = (gate_inputs + input) / self.beta
        return gate_inputs

class ETGNNConv(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(ETGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(2*in_size + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )        

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
            ParameterizedTrick(),
            nn.Sigmoid()
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size+in_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["radial"]], dim=-1
            )

        msg_h = self.edge_mlp(f)
        score_h = self.attention(msg_h)
        msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h, 'score_h': score_h}

    def corr_aggregate(self, nodes):
        return {
            'x_neigh': nodes.mailbox['m'].sum(1) / (nodes.data['score'] + 1e-8)
        }

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat

            # coordinate feature
            graph.ndata["x"] = coord_feat
            
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat

            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = (
                graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
            )
            # normalize coordinate difference
            graph.edata["x_diff"] = graph.edata["x_diff"] / (
                graph.edata["radial"].sqrt() + 1e-30
            )

            graph.apply_edges(self.message)
            # edge feature
            graph.edata['msg_h_t'] = graph.edata['msg_h'] * graph.edata['score_h']
            graph.update_all(fn.copy_e("msg_h_t", "m"), fn.sum("m", "h_neigh"))   
            graph.edata['msg_x_t'] = graph.edata['msg_x'] * graph.edata['score_h']         
            graph.update_all(fn.copy_e("score_h", "m_score"), fn.sum("m_score", "score"))
            graph.update_all(fn.copy_e("msg_x_t", "m"), self.corr_aggregate)
            
            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            return h, x