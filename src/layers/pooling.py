import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SumPooling, SetTransformerEncoder, SetTransformerDecoder, WeightAndSum
from common.utils import get_interface_nodes, get_cdr_interface, mdn_message, mdn_score

class Parametization(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        if self.training:
            random_noise = torch.rand(x.size()).to(x.device)
            gate_inputs = torch.log(random_noise) - torch.log(1 - random_noise)
            gate_inputs = (gate_inputs + x) / self.beta
        else:
            gate_inputs = x
        return gate_inputs


class AttentionPooling(nn.Module):
    def __init__(self, in_feats):
        super(AttentionPooling, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Linear(in_feats, 1)

    def forward(self, g, feats, get_attention=False):
        with g.local_scope():
            g.ndata['gate'] = self.atom_weighting(feats)
            gate = dgl.softmax_nodes(g, "gate")
            g.ndata.pop("gate")
            g.ndata["r"] = feats * gate
            h_g_sum = dgl.sum_nodes(g, "r")
            g.ndata.pop("r")

        if get_attention:
            return h_g_sum, gate
        else:
            return h_g_sum
        
class AttentiveFPGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.pool = SumPooling()
        self.att_l = nn.Linear(in_dim, 1, bias=False)
        self.att_r = nn.Linear(in_dim, 1, bias=False)
        self.node_linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, src_feats, dst_feats):
        left_att = self.att_l(src_feats).squeeze(-1) # N x 1
        batch_num_nodes = g.batch_num_nodes()
        right_att = self.att_r(dst_feats).squeeze(-1) # B x 1
        right_att = right_att.repeat_interleave(batch_num_nodes, dim=0) # N x 1
        with g.local_scope():
            g.ndata['alpha'] = self.dropout(F.leaky_relu(left_att + right_att))
            node_scores = dgl.softmax_nodes(g, 'alpha').unsqueeze(1)
            h_neighbors = self.node_linear(src_feats) # N x D
            msg = h_neighbors * node_scores.expand(h_neighbors.shape[0], h_neighbors.shape[1])
        
        sn_outputs = self.pool(g, msg)
        return sn_outputs

class AttentiveFPPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, timesteps=2, activation=F.silu, dropout=0.1):
        super().__init__()
        self.pool = SumPooling()
        self.gat = AttentiveFPGAT(in_dim, hidden_dim, dropout)
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.timesteps = timesteps
        self.activation = activation
    
    def forward(self, g, x):
        out = self.pool(g, x) # B x D
        for _ in range(self.timesteps):
            h = F.elu(self.gat(g, x, out)) # B x D
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.activation(self.gru(h, out))

        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin(out)

class WeightAndMean(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, feats):
        with g.local_scope():
            g.ndata["h"] = feats
            g.ndata["w"] = self.atom_weighting(feats)
            # Compute the weighted sum of node features
            h_sum = dgl.sum_nodes(g, "h", "w")
            # Compute the sum of weights
            w_sum = dgl.sum_nodes(g, "w")
            # Compute weighted mean: divide weighted sum by sum of weights
            h_mean = h_sum / w_sum
        return h_mean

class WeightAndMeanV2(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, feats):
        with g.local_scope():
            g.ndata["h"] = feats
            g.ndata["w"] = self.atom_weighting(feats)
            h_sum = dgl.sum_nodes(g, "h", "w")
            num_nodes = g.batch_num_nodes().to(torch.float32).unsqueeze(-1)
            h_mean = h_sum / num_nodes
        return h_mean 

class WeightAndSumParamatization(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            Parametization(),
            nn.Sigmoid()
        )

    def forward(self, g, feats):
        with g.local_scope():
            g.ndata["h"] = feats
            g.ndata["w"] = self.atom_weighting(feats)
            h_sum = dgl.sum_nodes(g, "h", "w")
        return h_sum
    
class SetPooling(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.pool_encoder = SetTransformerEncoder(d_model=in_feats, n_heads=4,
                                                  d_head=in_feats//4, d_ff=in_feats,
                                                  n_layers=2, block_type='isab',
                                                  m=32, dropouth=0.1, dropouta=0.1)
        self.pool_decoder = SetTransformerDecoder(in_feats, 4, in_feats//4, in_feats, 1, 1,
                                               dropouth=0.1, dropouta=0.1)

    def forward(self, g, feats):
        with g.local_scope():
            h_enc = self.pool_encoder(g, feats)
            return self.pool_decoder(g, h_enc)

class SetDecoderPooling(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.pool_decoder = SetTransformerDecoder(in_feats, 4, in_feats//4, in_feats, 1, 1,
                                               dropouth=0.1, dropouta=0.1)

    def forward(self, g, feats):
        with g.local_scope():
            return self.pool_decoder(g, feats)

# only sum nodes not in the binding interface, it is reverse sampling pooling  
class WeightedSumInterface(WeightAndSum):    
    def init_mask(self, g, feats):
        """Return a mask, where true for nodes in binding interface
        """
        masked_nodes = get_interface_nodes(g)
        # print(masked_nodes)
        weights = self.atom_weighting(feats)
        mask = torch.zeros_like(weights, dtype=torch.bool, device=weights.device)
        mask[masked_nodes] = True
        return mask, weights
    
    def calculate_weights(self, g, feats):
        mask, weights = self.init_mask(g, feats)
        weights = weights.masked_fill(mask, 0)
        return weights        

    def forward(self, g, feats):
        """Etype must be provided
        """
        with g.local_scope():
            weights = self.calculate_weights(g, feats)
            g.ndata["h"] = feats
            g.ndata['w'] = weights
            h_sum = dgl.sum_nodes(g, "h", "w")
        return h_sum

# CR stands for correct
class WeightedSumInterfaceCR(WeightedSumInterface):    
    def calculate_weights(self, g, feats):
        mask, weights = self.init_mask(g, feats)
        # reverse the mask to select nodes in interface
        weights = weights.masked_fill(~mask, 0)
        return weights        

class WeightedSumCDR(WeightedSumInterface):
    def init_mask(self, g, feats):
        weights = self.atom_weighting(feats)
        mask = (g.ndata['cdr'] == 1).unsqueeze(-1)
        return mask, weights

class WeightedSumCDRCR(WeightedSumCDR):
    def calculate_weights(self, g, feats):
        # cdr feature must be set in g.ndata
        mask, weights = self.init_mask(g, feats)
        weights = weights.masked_fill(~mask, 0)
        return weights    

class WeightedSumCDRInterface(WeightedSumInterface):
    def init_mask(self, g, feats):
        weights = self.atom_weighting(feats)
        # cdr feature must be set in g.ndata
        final_nodes = get_cdr_interface(g)
        mask = torch.zeros_like(weights, dtype=torch.bool, device=weights.device)
        mask[final_nodes] = True
        return mask, weights

class WeightedSumCDRInterfaceCR(WeightedSumCDRInterface):
    def calculate_weights(self, g, feats):
        mask, weights = self.init_mask(g, feats)
        weights = weights.masked_fill(~mask, 0)
        return weights

# only mean nodes in the binding interface        
class WeightedMeanInterface(WeightedSumInterface):
    def forward(self, g, feats):
        """Etype must be provided
        """
        with g.local_scope():
            weights = self.calculate_weights(g, feats)
            g.ndata["h"] = feats
            g.ndata['w'] = weights
            h_sum = dgl.sum_nodes(g, "h", "w")
            w_sum = dgl.sum_nodes(g, "w")
        return h_sum / (w_sum+1e-8)

class WeightedMeanCDR(WeightedSumCDR):
    def forward(self, g, feats):
        """Etype must be provided
        """
        with g.local_scope():
            weights = self.calculate_weights(g, feats)
            g.ndata["h"] = feats
            g.ndata['w'] = weights
            h_sum = dgl.sum_nodes(g, "h", "w")
            w_sum = dgl.sum_nodes(g, "w")
        return h_sum / (w_sum+1e-8)
    
class WeightedSumAb(WeightedSumInterface):    
    def init_mask(self, g, feats):
        weights = self.atom_weighting(feats)
        # cdr feature must be set in g.ndata
        ag_nodes = (g.ndata['ntype'] == 2).nonzero().flatten()
        mask = torch.zeros_like(weights, dtype=torch.bool, device=weights.device)
        # keep weights of nodes in ab
        mask[ag_nodes] = True
        return mask, weights
    
class WeightedSumAg(WeightedSumInterface):    
    def init_mask(self, g, feats):
        weights = self.atom_weighting(feats)
        # cdr feature must be set in g.ndata
        ab_nodes = (g.ndata['ntype'] != 2).nonzero().flatten()
        mask = torch.zeros_like(weights, dtype=torch.bool, device=weights.device)
        # keep weights of nodes in ag
        mask[ab_nodes] = True
        return mask, weights

class WeightedComponent(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.in_feats = in_feats
        self.weighting = nn.Linear(in_feats, 3)

    def exclude_mask(self, mask1, mask2):
        return (mask1 != mask2) & mask1

    def get_binding_nodes(self, g):
        return get_interface_nodes(g)

    def init_masks(self, g):
        binding_nodes = self.get_binding_nodes(g)
        mask3 = torch.zeros(g.num_nodes(), dtype=torch.bool).to(g.device)
        mask3[binding_nodes] = True

        # mask 1, nodes only in ab and not in interfaces
        mask1 = (g.ndata['ntype'] != 2) & (~mask3)
        # mask 2, nodes only in ag and not in interfaces
        mask2 = (g.ndata['ntype'] == 2) & (~mask3)
        # output 3 types of mask for ab, ag, inter
        return mask1, mask2, mask3

    def apply_mask(self, w, mask):
        # filter out weights of nodes corresponding to mi = True
        w = w.masked_fill(mask, 0)
        return w
    
    def forward(self, g, feats):
        ab_mask, ag_mask, inter_mask = self.init_masks(g)
        W = self.weighting(feats)
        M = torch.stack([ab_mask, ag_mask, inter_mask], dim=1).to(W.dtype)
        W = W * M
        weighted_feats = feats.unsqueeze(2) * W.unsqueeze(1)
        weighted_feats.reshape(feats.shape[0], -1)
        with g.local_scope():
            g.ndata['h'] = weighted_feats.reshape(g.num_nodes(), -1)
            h_sum = dgl.sum_nodes(g, "h")
        return h_sum
    
class WeightedComponentCDR(WeightedComponent):
    def get_binding_nodes(self, g):
        return get_cdr_interface(g)

class MDNPooling(nn.Module):
    def __init__(self, hidden_size, output_size=10, dropout=0.1):
        super().__init__()
        self.is_cdr = False
        self.aggressive = False
        self.MLP = nn.Sequential(nn.Linear(hidden_size*2, hidden_size), 
								nn.BatchNorm1d(hidden_size), 
								nn.ELU(), 
                                nn.Dropout(p=dropout))
        self.z_pi = nn.Linear(hidden_size, output_size)
        self.z_sigma = nn.Linear(hidden_size, output_size)
        self.z_mu = nn.Linear(hidden_size, output_size)
        

    def calculate_components(self, h):
        h = self.MLP(h)
        pi = F.softmax(self.z_pi(h), -1)
        sigma = F.elu(self.z_sigma(h)) + 1.1
        mu = F.elu(self.z_mu(h)) + 1.
        return pi, mu, sigma
    
    def forward(self, g, x):
        with g.local_scope():
            g.ndata['h_repr'] = x
            g.apply_edges(lambda edges: mdn_message(edges, lambda x: self.calculate_components(x)))
            pi, mu, sigma, dist = g.edata['pi'], g.edata['mu'], g.edata['sigma'], g.edata['dist']
            graph_scores = mdn_score(g, pi, mu, sigma, dist, is_cdr=self.is_cdr, aggressive=self.aggressive)
        return graph_scores.view(g.batch_size, 1)

class MDNCDRPooling(MDNPooling):
    def __init__(self, hidden_size, output_size=10, dropout=0.1):
        super().__init__(hidden_size, output_size, dropout)
        self.is_cdr = True

class MDNCDRPoolingAggressive(MDNPooling):
    def __init__(self, hidden_size, output_size=10, dropout=0.1):
        super().__init__(hidden_size, output_size, dropout)
        self.is_cdr = True
        self.aggressive = True