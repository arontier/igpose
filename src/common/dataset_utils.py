import sys, math, os, re
sys.path.append('..')
from typing import Iterator, Sequence, Optional
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
from torch_geometric.data import Data, Batch
import dgl
import numpy as np
from common.utils import standard_norm, get_relative_pos

class TruncatedSVDTorch:
    def __init__(self, n_components: int):
        """
        Initialize the TruncatedSVDTorch with a specified number of components.
        Parameters:
            n_components (int): Number of singular values and vectors to compute.
        """
        self.n_components = n_components
        self.components_ = None

    def fit(self, X):
        """
        Fit the model with X using truncated SVD.
        Parameters:
            X (torch.Tensor): The data matrix to fit (shape [n_samples, n_features]).
        """
        # Perform SVD
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        # Keep only the first n_components singular values/vectors
        self.components_ = Vh[:self.n_components, :]

        return self

    def transform(self, X):
        """
        Project the data onto the reduced singular vectors.
        Parameters:
            X (torch.Tensor): The data matrix to transform (shape [n_samples, n_features]).
        Returns:
            torch.Tensor: Transformed data with reduced dimensionality.
        """
        if self.components_ is None:
            raise ValueError("The model has not been fitted yet.")
        
        # Project the data
        X_transformed = X @ self.components_.T
        return X_transformed

    def fit_transform(self, X):
        """
        Fit the model with X and then transform X.
        Parameters:
            X (torch.Tensor): The data matrix to fit and transform (shape [n_samples, n_features]).
        Returns:
            torch.Tensor: Transformed data with reduced dimensionality.
        """
        return self.fit(X).transform(X)

class DistributedWeightedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, must be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.replacement = replacement
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Compute number of samples and total size
        if num_samples is None:
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:
                num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            else:
                num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.num_samples = num_samples
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))
        
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = torch.cat([indices, indices[:padding_size]])
            else:
                indices = torch.cat([indices, indices.repeat((padding_size + len(indices) - 1) // len(indices))[:padding_size]])
        else:
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        rank_weights = self.weights[indices]
        sampled_indices = torch.multinomial(rank_weights, self.num_samples, self.replacement)
        yield from iter(indices[sampled_indices].tolist())

def process_graph_edges(g, edge_dim=10, is_edge_onehot=True, add_reverse_edge=True):
    # set node features
    g.ndata['ntype'] = g.ndata['ntype'].to(torch.uint8)
    g.ndata['coord'] = standard_norm(g.ndata['coord'].float())
    # set edge embeddings
    g.edata['rel_pos'] = get_relative_pos(g)
    distance_attr = g.edata['distance'] # min_distance, ca_distance, center_of_mass distance
    all_distance_attr = gaussian_edge_embedding(distance_attr, edge_dim)
    if is_edge_onehot:
        g.edata['etype'] = g.edata['etype'].to(torch.long)
        g.edata['attr'] = torch.cat([all_distance_attr.view(g.num_edges(), -1), F.one_hot(g.edata['etype'], 3).to(torch.float32)], dim=-1)
    else:
        g.edata['etype'] = g.edata['etype'].to(torch.uint8)
        g.edata['attr'] = all_distance_attr.view(g.num_edges(), -1)

    if add_reverse_edge:
        g = dgl.add_reverse_edges(g, copy_edata=True)
    
    return g

def clean_graph(g, keep_resid=False):
    ntype = g.ndata['ntype']
    nfeat = g.ndata.pop('feat') if 'feat' in g.ndata else None
    efeat = g.edata.pop('attr') if 'attr' in g.edata else None
    distance = g.edata.pop('distance')
    coord = g.ndata.pop('coord')
    etype = g.edata['etype']
    res_id = g.ndata['resid']
    keys = list(g.ndata.keys())
    for k in keys:
        g.ndata.pop(k)
    keys = list(g.edata.keys())
    for k in keys:
        g.edata.pop(k)
    g.ndata['ntype'] = ntype
    g.ndata['coord'] = coord
    if not nfeat is None:
        g.ndata['feat'] = nfeat

    if keep_resid:
        g.ndata['resid'] = res_id.to(torch.int32)

    g.edata['etype'] = etype
    if not efeat is None:
        g.edata['attr'] = efeat
    g.edata['distance'] = distance
    return g   

def get_real_path(graph_path):
    if os.path.islink(graph_path):
        graph_path = os.path.realpath(graph_path)
    return graph_path

def extract_graph(g, etype):
    if isinstance(etype, int):
        edge_mask = g.edata['etype'] == etype
    else:
        edge_mask = torch.isin(g.edata['etype'], torch.LongTensor(etype).to(g.device))
    src, dst = g.edges()
    src, dst = src[edge_mask], dst[edge_mask]
    unique_nodes = torch.unique(torch.cat([src, dst]))
    return dgl.node_subgraph(g, unique_nodes)

def collate_multi(batch):
    # prepare data
    orig_labels, labels, batch_g = [], [], []
    for ab, ig, lb, orig_lb in batch:
        batch_g.extend([ab, ig])
        labels.append(lb)
        orig_labels.append(orig_lb)

    batch = dgl.batch(batch_g)
    if isinstance(labels[0], torch.Tensor):
        lb_type = labels[0].dtype
    elif isinstance(labels[0], int) or isinstance(labels[0], bool):
        lb_type = torch.int64
    else:
        lb_type = torch.float32
    return batch, torch.tensor(labels, dtype=lb_type), torch.tensor(orig_labels, dtype=torch.float32)

def collate_simple(batch):
    # prepare data
    orig_labels, labels, batch_g, batch_g2, batch_g3 = [], [], [], [], []
    for all_g, lb, orig_lb in batch:
        if all_g is None: continue
        if isinstance(all_g, tuple):
            if not all_g[0] is None:
                batch_g.append(all_g[0])
                batch_g2.append(all_g[1])
                labels.append(lb)
                orig_labels.append(orig_lb)
                if len(all_g) == 3: # sep case
                    batch_g3.append(all_g[2])
        else:
            batch_g.append(all_g)
            labels.append(lb)
            orig_labels.append(orig_lb)

    if isinstance(labels[0], torch.Tensor):
        lb_type = labels[0].dtype
    elif isinstance(labels[0], int) or isinstance(labels[0], bool):
        lb_type = torch.int64
    else:
        lb_type = torch.float32
    batch = dgl.batch(batch_g)
    if batch_g2:
        batch = (batch, dgl.batch(batch_g2))
    if batch_g3:
        batch = batch + (dgl.batch(batch_g3),)
    return batch, torch.tensor(labels, dtype=lb_type), torch.tensor(orig_labels, dtype=torch.float32)

def collate_pyg(batch):
    graphs, labels, olabels, graphs2 = [], [], [], []
    for g, lb, olb in batch:
        if isinstance(g, tuple):
            graphs.append(g[0])
            graphs2.append(g[1])
        else:
            graphs.append(g)
        labels.append(lb)
        olabels.append(olb)
    batch = Batch.from_data_list(graphs)
    if graphs2:
        batch_g2 = Batch.from_data_list(graphs2)
        batch = (batch, batch_g2)
    return batch, torch.LongTensor(labels), torch.FloatTensor(olabels)

def is_empty(graph):
    if graph is None or graph.edge_index is None or graph.edge_index.numel() == 0:
        return True
    return False

def collate_ppa(batch):
    ab_list, ig_list, ag_list = [], [], []
    labels, olabels = [], []
    for all_g, lb, olb in batch:
        if len(all_g) == 3:
            ab, ig, ag = all_g
            if is_empty(ab) or is_empty(ig) or is_empty(ag):
                continue
            ab_list.append(ab)
            ig_list.append(ig)
            ag_list.append(ag)
        else:
            ab, ag = all_g
            if is_empty(ab) or is_empty(ag):
                continue
            ab_list.append(ab)
            ag_list.append(ag)
        labels.append(lb)
        olabels.append(olb)
    labels = torch.LongTensor(labels)
    olabels = torch.FloatTensor(olabels)
    if not ig_list:
        return (Batch.from_data_list(ab_list), Batch.from_data_list(ag_list)), labels, olabels
    return (Batch.from_data_list(ab_list), Batch.from_data_list(ig_list), Batch.from_data_list(ag_list)), labels, olabels

def reduce_edge_dimension(graphs, d=20):
    merge_features = torch.cat([g.edata['attr'] for g in graphs], dim=0)
    dim = merge_features.shape[-1]
    if d != dim:
        svd = TruncatedSVDTorch(n_components=d)
        merge_features.to(torch.device('cuda:0'))
        compressed_features = svd.fit_transform(merge_features)
        return compressed_features.cpu()
    return merge_features.numpy()

def map_edge_features(graphs, features):
    if isinstance(features, torch.Tensor):
        mean, std = torch.mean(features, dim=0), torch.std(features, dim=0)
    else:
        mean, std = np.mean(features, axis=0), np.std(features, axis=0)
    features = (features - mean) / std
    start = 0
    for g in graphs:
        num_edges = g.num_edges()
        end = start + num_edges
        edge_features = features[start:end, :]
        start = end
        if not isinstance(edge_features, torch.Tensor):
            edge_features = torch.from_numpy(edge_features)
        g.edata['attr'] = edge_features

def sample_node_graph(g, ratio=0.8, mode='degree'):
    """Sample node graph for contrastive learning
    """
    num_sel_nodes = int(g.num_nodes() * ratio)
    if mode == 'simple':
        sampled_nodes = torch.randperm(g.num_nodes())[:num_sel_nodes]
    else:
        in_degrees = g.in_degrees()
        sorted_nodes = torch.argsort(in_degrees, descending=True)
        proba = in_degrees.float() / in_degrees.sum().float()
        sampled_nodes = sorted_nodes[torch.multinomial(proba, num_sel_nodes, replacement=False)]

    sg = dgl.node_subgraph(g, sampled_nodes, store_ids=False)
    in_degrees = sg.in_degrees()
    valid_nodes = (in_degrees > 0).nonzero(as_tuple=True)[0]
    return dgl.node_subgraph(g, valid_nodes, store_ids=False)

def compute_class_weights(labels, type='balanced', class_ratio=''):
    """
    Compute the weights for each class based on the inverse frequency or balanced of the class.
    """
    class_counts = torch.bincount(labels)
    if type == 'balanced':
        class_weights = len(labels) / (len(class_counts) * class_counts.float())
    else:
        class_weights = 1.0 / class_counts.float()

    if class_ratio:
        class_ratio = list(map(float, class_ratio.split(',')))
        class_weights = torch.tensor(class_ratio) * class_weights

    sample_weights = class_weights[labels]
    return sample_weights

def create_weighted_sampler(labels, type='balanced', class_ratio=''):
    """
    Create a WeightedRandomSampler to handle the imbalance using oversampling.
    """
    sample_weights = compute_class_weights(labels, type, class_ratio)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  
    )
    return sampler

# using for haddock decoys, & datasets which merge all sequences to [heavy, light, ag]
def map_embeddings(embeds, g, is_syn=True, id=None, is_merge=False, ab_chain='', ag_chain=''):
    h_indices = g.ndata['ntype'] == 0
    l_indices = g.ndata['ntype'] == 1
    ag_indices = g.ndata['ntype'] == 2
    embeddings = torch.zeros((g.num_nodes(), embeds[-1].shape[1]), dtype=torch.float32)
    l_idx, ag_idx = 0, 0
    if is_merge: # only having h_indices & ag_indices (not clear which one is heavy, light)
        len_ab = len(ab_chain)
        len_ag = len(ag_chain)
        ab_embed = torch.cat(embeds[:len_ab], dim=0)
        ag_embed = torch.cat(embeds[-len_ag:], dim=0)
        embeds = [ab_embed, ag_embed]
        
    if h_indices.sum() != 0:
        h_embeddings = embeds[0][g.ndata['chain_offset'][h_indices]]
        embeddings[h_indices.nonzero().flatten(), :] = h_embeddings
        ag_idx += 1
        l_idx += 1
    elif is_syn: # decoy dataset
        ag_idx += 1
        l_idx += 1
    
    if l_indices.sum() != 0:
        l_embeddings = embeds[l_idx][g.ndata['chain_offset'][l_indices]]
        embeddings[l_indices.nonzero().flatten(), :] = l_embeddings
        ag_idx += 1
    
    remain_embeds = embeds[ag_idx:]
    if len(remain_embeds) > 1:
        ag_embeds = torch.cat(remain_embeds, dim=0)
    else:
        ag_embeds = remain_embeds[0]
    ag_embeddings = ag_embeds[g.ndata['chain_offset'][ag_indices]]
    embeddings[ag_indices.nonzero().flatten(), :] = ag_embeddings
    return embeddings

# using for cross dock-like datasets where ag & ab embeddings are separated
def map_separate_embeddings(ab_embeds, ag_embeds, g):
    embeddings = torch.zeros((g.num_nodes(), ag_embeds[0].shape[1]), dtype=torch.float32)
    h_indices = g.ndata['ntype'] == 0
    l_indices = g.ndata['ntype'] == 1
    if len(ab_embeds) == 2:
        embeddings[h_indices.nonzero().flatten(), :] = ab_embeds[0][g.ndata['chain_offset'][h_indices]]
        embeddings[l_indices.nonzero().flatten(), :] = ab_embeds[1][g.ndata['chain_offset'][l_indices]]
    elif len(ab_embeds) == 1:
        embeddings[h_indices.nonzero().flatten(), :] = ab_embeds[0][g.ndata['chain_offset'][h_indices]] if torch.any(h_indices) else ab_embeds[0][g.ndata['chain_offset'][l_indices]]
    else:
        raise ValueError('Ab embeddings are empty')

    if len(ag_embeds) > 1:
        ag_embeds = torch.cat(ag_embeds)
    else:
        ag_embeds = ag_embeds[0]
    ag_indices = g.ndata['ntype'] == 2
    embeddings[ag_indices.nonzero().flatten(), :] = ag_embeds[g.ndata['chain_offset'][ag_indices]]
    return embeddings

def gaussian_edge_embedding(distance, edge_dim=10, min_scale=0.25, max_scale=8.):
    scales = torch.logspace(np.log10(min_scale), np.log10(max_scale), edge_dim, dtype=torch.float32, device=distance.device)
    return torch.exp(-distance.unsqueeze(-1) / scales)

def convert_dgl_to_pyg(g):
    x = g.ndata['feat'] if 'feat' in g.ndata else None
    pos = g.ndata['coord'] if 'coord' in g.ndata else None
    # Extract edge indices and edge features
    edge_index = torch.stack(g.edges(), dim=0)  # [2, num_edges]
    edge_attr = g.edata['attr'] if 'attr' in g.edata else None
    # Create PyTorch Geometric Data object
    pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, ntype=g.ndata['ntype'], etype=g.edata['etype'])
    return pyg_graph

# extract chain id for casp
def extract_casp_chains(query_id):
    pattern = r"H\d+TS\d+_\d+_(.*?)_H\d+TS\d+_\d+_(.*?)_clean"
    match = re.match(pattern, query_id)
    return match.groups()

def extract_af3_chains(query_id):
    x = query_id.split('_')
    return f'{x[3]}_{x[4]}'

def rotate_graph(coords, rotation_range=(-torch.pi, torch.pi)):
    """
    Augment a set of 3-D coordinates by applying a random rotation only.

    Parameters:
        coords (torch.Tensor): Input tensor of shape (B, 3) where B is the number of points.
        rotation_range (tuple): Tuple (min, max) specifying the range of rotation angles in radians.
    
    Returns:
        torch.Tensor: Rotated coordinates of shape (B, 3).
    """
    # Generate random rotation angles as tensors
    rx = torch.empty(1).uniform_(*rotation_range)
    ry = torch.empty(1).uniform_(*rotation_range)
    rz = torch.empty(1).uniform_(*rotation_range)
    
    # Compute cosine and sine using the tensor inputs
    cos_rx = torch.cos(rx)
    sin_rx = torch.sin(rx)
    cos_ry = torch.cos(ry)
    sin_ry = torch.sin(ry)
    cos_rz = torch.cos(rz)
    sin_rz = torch.sin(rz)
    
    # Extract scalar values from the one-element tensors
    cos_rx_val = cos_rx.item()
    sin_rx_val = sin_rx.item()
    cos_ry_val = cos_ry.item()
    sin_ry_val = sin_ry.item()
    cos_rz_val = cos_rz.item()
    sin_rz_val = sin_rz.item()

    # Construct rotation matrix around x-axis
    Rx = torch.tensor([
        [1, 0, 0],
        [0, cos_rx_val, -sin_rx_val],
        [0, sin_rx_val,  cos_rx_val]
    ], dtype=coords.dtype, device=coords.device)

    # Construct rotation matrix around y-axis
    Ry = torch.tensor([
        [ cos_ry_val, 0, sin_ry_val],
        [ 0,          1, 0         ],
        [-sin_ry_val, 0, cos_ry_val]
    ], dtype=coords.dtype, device=coords.device)

    # Construct rotation matrix around z-axis
    Rz = torch.tensor([
        [cos_rz_val, -sin_rz_val, 0],
        [sin_rz_val,  cos_rz_val, 0],
        [0,          0,          1]
    ], dtype=coords.dtype, device=coords.device)

    # Combine the rotations: apply Rx, then Ry, then Rz
    R = Rz @ Ry @ Rx

    # Rotate the coordinates
    rotated_coords = torch.matmul(coords, R.T)
    return rotated_coords

def get_cdr_onehot(g, h_cdr, l_cdr):
    cdr_onehot = torch.zeros((g.num_nodes(),))
    # try:
    all_res_ids = g.ndata['resid'].to(torch.int32)
    h_index = (g.ndata['ntype'] == 0).nonzero().flatten()
    l_index = (g.ndata['ntype'] == 1).nonzero().flatten()
    
    if len(h_index):
        h_cdr_res_ids = set(map(int, h_cdr.split(',')))
        for i in h_index:
            if not all_res_ids[i].item() in h_cdr_res_ids:
                continue
            cdr_onehot[i] = 1
    
    if len(l_index):
        l_cdr_res_ids = set(map(int, l_cdr.split(',')))
        for i in l_index:
            if not all_res_ids[i].item() in l_cdr_res_ids:
                continue
            cdr_onehot[i] = 1
    # except Exception as e:
    return cdr_onehot
    
def prune_edges(g, edge_distance_threshold):
    distance = g.edata['distance'][:,0] #take the minimum distance
    keep_eids = torch.nonzero(distance <= edge_distance_threshold, as_tuple=False).squeeze()
    if keep_eids.dim() == 0:
        keep_eids = keep_eids.unsqueeze(0)
    
    pruned_g = dgl.edge_subgraph(g, keep_eids.to(torch.int32), relabel_nodes=True)
    if '_ID' in g.ndata:
        original_nid = g.ndata['_ID']
        pruned_g.ndata['_ID'] = original_nid[pruned_g.ndata['_ID']]
    return pruned_g