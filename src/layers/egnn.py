"""Torch Module for E(n) Equivariant Graph Convolutional Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch.nn as nn

import dgl
from dgl import function as fn

class EGNNNormConv(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNNormConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
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
            graph.ndata["h"] = node_feat
            # coordinate feature
            if torch.isnan(graph.ndata["h"]).any():
                print('h in egnn', graph.ndata["h"])
            graph.ndata["x"] = coord_feat
            if torch.isnan(graph.ndata["x"]).any():
                print('coord', graph.ndata["x"])
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = (
                graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
            )

            radial_norm = (graph.edata["radial"] + 1e-6).sqrt()
            graph.edata["x_diff"] = graph.edata["x_diff"] / (radial_norm + 1e-6)

            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]
            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh
            # standard normalize coordinates
            graph.ndata['x'] = x
            x_mean = dgl.mean_nodes(graph, 'x') # must have shape B x 3
            x_mean_broadcast = dgl.broadcast_nodes(graph, x_mean)
            x_center = x - x_mean_broadcast
            graph.ndata['x_center_square'] = x_center.square()
            x_var = dgl.mean_nodes(graph, 'x_center_square')
            x_std_broadcast = dgl.broadcast_nodes(graph, (x_var + 1e-6).sqrt())
            x_norm = x_center / (x_std_broadcast + 1e-6)

            return h, x_norm

class EGNNFroConv(EGNNNormConv):
    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat
            # coordinate feature
            graph.ndata["x"] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            radial_norm = (graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1) + 1e-6).sqrt()

            graph.edata["radial"] = radial_norm
            graph.edata["x_diff"] = graph.edata["x_diff"] / (radial_norm + 1e-6)
    
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]
            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            return h, x