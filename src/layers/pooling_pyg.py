import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GATConv

class SumPoolingPyg(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, g, feat):
        return global_add_pool(feat, g.batch)

class WeightAndSumPyG(nn.Module):
    """Compute importance weights for nodes and perform a weighted sum in PyG."""
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        # Compute weights for each node
        weights = self.atom_weighting(x)  # Shape: (N, 1)
        
        # Element-wise multiplication of features by weights
        weighted_x = x * weights  # Shape: (N, in_feats)
        
        # Aggregate weighted node features graph-wise
        h_g_sum = global_add_pool(weighted_x, g.batch)  # Shape: (B, in_feats)
        
        return h_g_sum
    
class WeightAndMeanPyG(WeightAndSumPyG):
    """Compute importance weights for nodes and perform a weighted mean in PyG."""
    def forward(self, g, x):
        # Compute weights for each node
        weights = self.atom_weighting(x)  # Shape: (N, 1)
        
        # Element-wise multiplication of features by weights
        weighted_x = x * weights  # Shape: (N, in_feats)
        
        # Aggregate weighted node features graph-wise
        h_g_sum = global_add_pool(weighted_x, g.batch)  # Shape: (B, in_feats)
        w_g_sum = global_add_pool(weights, g.batch)
        
        return h_g_sum / w_g_sum
    
class AttentiveFPPoolingPyG(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, timesteps=2, activation=F.silu, dropout=0.1):
        super().__init__()
        self.gat = GATConv(in_dim, hidden_dim,
                        dropout=dropout, add_self_loops=False,
                        negative_slope=0.01)
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.timesteps = timesteps
        self.activation = activation
    
    def forward(self, g, x):
        out = global_add_pool(x, g.batch) # B x D
        for _ in range(self.timesteps):
            h = F.elu(self.gat((x, out), g.edge_index)) # B x D
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.activation(self.gru(h, out))

        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin(out)
    
class WeightAndSumInterfacePyG(WeightAndSumPyG):
    """Compute importance weights for nodes and perform a weighted sum in PyG."""
    def get_interface_nodes(self, g):
        mask = g.etype == 1
        src, dst = g.edge_index
        src, dst = src[mask], dst[mask]
        masked_nodes = torch.unique(torch.cat([src, dst], dim=-1))
        return masked_nodes
    
    def calculate_weights(self, g, feats):
        masked_nodes = self.get_interface_nodes(g)    
        weights = self.atom_weighting(feats)
        mask = torch.zeros_like(weights, dtype=torch.bool, device=weights.device)
        mask[masked_nodes] = True
        weights = weights.masked_fill(mask, 0)
        return weights        

    def forward(self, g, x):
        # Compute weights for each node
        weights = self.calculate_weights(g, x)
        
        # Element-wise multiplication of features by weights
        weighted_x = x * weights  # Shape: (N, in_feats)
        
        # Aggregate weighted node features graph-wise
        h_g_sum = global_add_pool(weighted_x, g.batch)  # Shape: (B, in_feats)
        
        return h_g_sum
