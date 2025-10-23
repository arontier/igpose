import os, sys
sys.path.append('..')
import torch
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
import numpy as np
from common import utils
from common.dataset_utils import *
from common.sampling import sample_khop, extract_cdr_seeds
from common.mapping_utils import EMBEDDING_FOLDER_MAP

OUTLIER_FOLDERS = ['ab_chai1_fullgraph', 'nb_chai1_fullgraph', 'tcr_pmhc_chai1_fullgraph']

class AntibodyGraphDatasetFolder(DGLDataset):
    def __init__(self, root_dir, metadata, all_embeds, edge_components=10,
                 label_mean=None, label_std=None, pre_fetch=False, is_binary=False,
                 use_ef=True, edge_dim=10, is_node_onehot=False, is_di_angle=False,
                 is_edge_onehot=False, khop=3, embed_path='', **kargs):
        self.edge_components = edge_components
        self.root_dir = root_dir
        self.all_embeds = all_embeds
        self.embed_path = embed_path
        self.metadata = metadata
        self.label_mean = label_mean
        self.label_std = label_std
        self.pre_fetch = pre_fetch
        self.is_binary = is_binary
        self.use_ef = use_ef
        self.edge_dim = edge_dim
        self.is_node_onehot = is_node_onehot
        self.is_di_angle = is_di_angle
        self.is_edge_onehot =  is_edge_onehot
        self.has_lightchain = True
        self.khop = khop
        self.add_reverse_edge = True
        self.sep_embed = kargs['sep_embed'] if 'sep_embed' in kargs else False
        self.sample_binding_site = kargs['sample_binding_site'] if 'sample_binding_site' in kargs else False
        self.mask_embed = kargs['mask_embed'] if 'mask_embed' in kargs else False
        self.augmentation = kargs['augmentation'] if 'augmentation' in kargs else False
        self.node_threshold = kargs['node_threshold'] if 'node_threshold' in kargs else -1
        self.rc_node_threshold = kargs['rc_node_threshold'] if 'rc_node_threshold' in kargs else self.node_threshold
        self.sampling_method = kargs['sampling_method'] if 'sampling_method' in kargs else 'k_iter'
        self.cdr = kargs['cdr'] if 'cdr' in kargs else False
        self.keep_resid = kargs['keep_resid'] if 'keep_resid' in kargs else False
        self.store_nids = kargs['store_nids'] if 'store_nids' in kargs else False # store original node_ids after sampling
        self.cdr_onehot = kargs['cdr_onehot'] if 'cdr_onehot' in kargs else False
        self.cdr_pooling = True if self.cdr and 'cdr' in kargs['pooling'] else False
        self.edge_distance_threshold = kargs['edge_distance_threshold'] if 'edge_distance_threshold' in kargs else 0
        self.label_norm = kargs['label_norm'] if 'label_norm' in kargs else 'stdlog10'
        if self.cdr:
            self.metadata.fillna('', inplace=True)
        # print('node threshold', self.node_threshold, self.rc_node_threshold)
        super().__init__(name="Synthetic Antibody Dataset")
    
    def sample_bs(self, g):
        # ignore intra edges of antigen
        mask = g.edata['etype'] != 2
        src, dst = g.edges()
        src, dst = src[mask], dst[mask]
        unique_nodes = torch.unique(torch.cat([src, dst]))
        return dgl.node_subgraph(g, unique_nodes, store_ids=False)

    def load_labels(self):
        orig_labels, labels = [], []
        for i, row in self.metadata.iterrows():
            lb = row['DockQ']
            orig_labels.append(lb)
            if not self.is_binary:                
                lb_val = -np.log10(lb+1e-8) if self.label_norm == 'stdlog10' else lb
            else:
                lb_val = lb >= 0.8
            labels.append(lb_val)            
        
        if not self.is_binary:
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            if self.label_norm.startswith('std'):
                if self.label_mean is None:
                    self.label_mean = label_tensor.mean(dim=0, keepdim=True)
                if self.label_std is None:
                    self.label_std = label_tensor.std(dim=0, keepdim=True)
                self.labels = (label_tensor - self.label_mean) / (self.label_std + 1e-8)
            else:
                self.labels = label_tensor
        else:
            self.labels = torch.tensor(labels, dtype=torch.int64)
        self.orig_labels = torch.tensor(orig_labels, dtype=torch.float32)

    def map_embeds(self, query_id):
        if not 'ab' in self.all_embeds:
            return self.all_embeds[f"{query_id}_embed.pt"]
        else:
            # separate embeds
            ab_id, ag_id = utils.extract_file_id(query_id)
            ab_embeds = self.all_embeds['ab'][ab_id]
            ag_embeds = self.all_embeds['ag'][ag_id]
            return ab_embeds + ag_embeds            

    def load_decoy_node_features(self, g, folder, query_id):
        if self.embed_path and not self.all_embeds:
            embeds = torch.load(get_real_path(f"{self.embed_path}/{EMBEDDING_FOLDER_MAP.get(folder, folder)}/{query_id}_embed.pt"))
        else:
            embeds = self.map_embeds(query_id)
        esm_node_embeds = map_embeddings(embeds, g, True, id=query_id)
        return esm_node_embeds
    
    def load_casp_node_features(self, g, folder, query_id):
        ab_chain, ag_chain = extract_casp_chains(query_id)
        embeds = torch.load(get_real_path(f"{self.embed_path}/{EMBEDDING_FOLDER_MAP.get(folder, folder)}/{query_id}_embed.pt"))
        esm_node_embeds = map_embeddings(embeds, g, False, is_merge=False, ab_chain=ab_chain, ag_chain=ag_chain)
        return esm_node_embeds

    def load_af3_node_features(self, g, folder, query_id):
        chain_id = extract_af3_chains(query_id)
        embeds = torch.load(get_real_path(f"{self.embed_path}/{EMBEDDING_FOLDER_MAP.get(folder, folder)}/{query_id}_embed.pt"))
        esm_node_embeds = map_embeddings(embeds, g, False, is_merge=False, ab_chain=chain_id, ag_chain=chain_id)
        return esm_node_embeds

    def load_merge_node_features(self, g, folder, query_id):
        embeds = torch.load(get_real_path(f"{self.embed_path}/{EMBEDDING_FOLDER_MAP.get(folder, folder)}/{query_id}_embed.pt"))
        esm_node_embeds = map_embeddings(embeds, g, False)
        return esm_node_embeds

    def load_crossdock_node_features(self, g, folder, query_id):
        # separate embeds
        ab_id, ag_id = utils.extract_file_id(query_id)
        ab_embeds = torch.load(get_real_path(f"{self.embed_path}/{EMBEDDING_FOLDER_MAP.get(folder, folder)}/ab_{ab_id.lower()}_embed.pt"))
        ag_embeds = torch.load(get_real_path(f"{self.embed_path}/{EMBEDDING_FOLDER_MAP.get(folder, folder)}/ag_{ag_id.lower()}_embed.pt"))
        esm_node_embeds = map_separate_embeddings(ab_embeds, ag_embeds, g)
        return esm_node_embeds

    def generate_node_features(self, g, folder, query_id, dataset='haddock'):
        if self.mask_embed:
            return g
        try:
            if dataset.startswith('had'):
                esm_node_embeds = self.load_decoy_node_features(g, folder, query_id)
            elif dataset.startswith('cas'):
                esm_node_embeds = self.load_casp_node_features(g, folder, query_id)
            elif dataset.startswith('af3'):
                esm_node_embeds = self.load_af3_node_features(g, folder, query_id)
            else:
                if self.sep_embed:
                    esm_node_embeds = self.load_crossdock_node_features(g, folder, query_id)
                else:
                    esm_node_embeds = self.load_merge_node_features(g, folder, query_id)
            feats = [esm_node_embeds]
            if not self.is_node_onehot and not self.is_di_angle:
                g.ndata['feat'] = feats[0]
            elif self.is_node_onehot:
                feats.append(F.one_hot(g.ndata['ntype'].to(torch.int64), 3).to(torch.float32))
            elif self.is_di_angle:
                feats.append(g.ndata['angle'])
            if self.is_node_onehot or self.is_di_angle:
                g.ndata['feat'] = torch.cat(feats, dim=1)       
        except: 
            print(folder, query_id)

    def set_graph(self, g):
        process_graph_edges(g, edge_dim=self.edge_dim,
                            is_edge_onehot=self.is_edge_onehot,
                            add_reverse_edge=self.add_reverse_edge)
        if not self.use_ef:
            g.edata.pop('attr')
        return g

    def load_graph_from_file(self, folder, subfolder, filename):
        if subfolder:
            graph_path = get_real_path(os.path.join(self.root_dir, f"{folder}/{subfolder}/{filename}_abg.npz"))
        else:
            graph_path = get_real_path(os.path.join(self.root_dir, f"{EMBEDDING_FOLDER_MAP.get(folder, folder)}/{filename}_abg.npz"))
        g = utils.decompress_decoy_graph(graph_path)
        return g

    def load_graph(self, idx):
        row = self.metadata.iloc[idx]
        subfolder = '' if not row['folder'].startswith('had') else row['query_id']
        # step 1: load graph
        g = self.load_graph_from_file(row['folder'], subfolder, row['file'])
        if self.augmentation:
            g.ndata['coord'] = rotate_graph(g.ndata['coord'])
        
        # step 2: load embedding
        self.generate_node_features(g, row['folder'], row['query_id'], dataset=row['folder'])
        g = self.set_graph(g)
        
        if self.cdr_onehot:
            cdr_onehot = get_cdr_onehot(g, str(row['H_CDR']), str(row['L_CDR']))
            feat = g.ndata.pop('feat')
            g.ndata['feat'] = torch.cat([feat, cdr_onehot.unsqueeze(1)], dim=-1)
            
        if self.mask_embed:
            # dynamically generate embedding at binding site
            if self.cdr:
                g.h_cdr = str(row['H_CDR'])
                g.l_cdr = str(row['L_CDR'])
            return g

        # step 3: clean redundant info    
        g = clean_graph(g, keep_resid=True)
        
        # step 4: load seeds & perform k-hop sampling
        if self.cdr or self.cdr_pooling:
            seeds = extract_cdr_seeds(g, str(row['H_CDR']), str(row['L_CDR'])) if self.cdr else None
            cdr_marking = torch.zeros((g.num_nodes(), ), dtype=torch.int32)
            if seeds is None:
                seeds = utils.get_interface_nodes(g)
            cdr_marking[seeds] = 1
            g.ndata['cdr'] = cdr_marking
        else:
            seeds = None

        if self.khop != -1:
            nt = self.node_threshold if not row['folder'] in OUTLIER_FOLDERS else self.rc_node_threshold
            g = sample_khop(g, self.khop, store_ids=self.store_nids, node_threshold=nt, seeds=seeds, mode=self.sampling_method)

        if not self.keep_resid:
            g.ndata.pop('resid')
        if not 'feat' in g.ndata:
            return None
        
        # pruning edge
        if self.edge_distance_threshold:
            g = prune_edges(g, self.edge_distance_threshold)

        g.edata.pop('distance')
        return g

    def process(self):
        self.load_labels()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        g = self.load_graph(idx)
        return g, self.labels[idx], self.orig_labels[idx]

# using to validate few-shot datasets
class AntibodyGraphDatasetFolder2(AntibodyGraphDatasetFolder):
    def __init__(self, root_dir, metadata, all_embeds, edge_components=10,
                 label_mean=None, label_std=None, pre_fetch=False, is_binary=False,
                 use_ef=True, edge_dim=10, is_node_onehot=False, is_di_angle=False,
                 is_edge_onehot=False, khop=0, **kargs):
        super().__init__(root_dir, metadata, all_embeds, edge_components,
                 label_mean, label_std, pre_fetch,
                 is_binary, use_ef, edge_dim, is_node_onehot, is_di_angle, is_edge_onehot, khop, **kargs)
        self.sample_binding_site = kargs['sample_binding_site']
        self.graph_fusion = kargs['graph_fusion']
        self.sep_embed = kargs['sep_embed'] if 'sep_embed' in kargs else False

    def post_process(self, g, idx):
        g = clean_graph(g, keep_resid=True)
        if self.cdr or self.cdr_pooling:
            row = self.metadata.iloc[idx]
            seeds = extract_cdr_seeds(g, str(row['H_CDR']), str(row['L_CDR'])) if self.cdr else None
            cdr_marking = torch.zeros((g.num_nodes(), ), dtype=torch.int32)
            if seeds is None:
                seeds = utils.get_interface_nodes(g)
            cdr_marking[seeds] = 1
            g.ndata['cdr'] = cdr_marking
        else:
            seeds = None
        if self.khop != -1:
            g = sample_khop(g, self.khop, node_threshold=self.node_threshold, seeds=seeds, mode=self.sampling_method)

        if self.sample_binding_site:
            g = self.sample_bs(g)

        g.ndata.pop('resid')

        if self.graph_fusion:
            ab = extract_graph(g, 0)
            ig = extract_graph(g, 1)
            ag = extract_graph(g, 2)
            if ab.num_nodes() == 0 or ig.num_nodes() == 0 or ag.num_nodes() == 0:
                return (None, None, None), self.labels[idx], self.orig_labels[idx]
            return (ab, ig, ag), self.labels[idx], self.orig_labels[idx]
        return g, self.labels[idx], self.orig_labels[idx]

    def load_graph_from_file(self, query_id):
        g = utils.decompress_decoy_graph(get_real_path(os.path.join(self.root_dir, f"{query_id}_abg.npz")))
        return g
    
    def generate_node_features(self, g, query_id):
        query_id = query_id.replace("_abg.pt", "")
        if not self.sep_embed:
            embeds = torch.load(f"{self.embed_path}/{query_id}_embed.pt")
            esm_node_embeds = map_embeddings(embeds, g, is_syn=False)
        else:
            # separate embeds
            ab_id, ag_id = utils.extract_file_id(query_id)
            ab_embeds = torch.load(f"{self.embed_path}/embed/ab_{ab_id.lower()}_embed.pt")
            ag_embeds = torch.load(f"{self.embed_path}/embed/ag_{ag_id.lower()}_embed.pt")
            esm_node_embeds = map_separate_embeddings(ab_embeds, ag_embeds, g)
            
        g.ndata['feat'] = esm_node_embeds

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        g = self.load_graph_from_file(row['query_id'])
        self.generate_node_features(g, row['query_id'])
        g = self.set_graph(g)
        return self.post_process(g, idx)
    
class AntibodyGraphDatasetFolderCasp(AntibodyGraphDatasetFolder2):
    def generate_node_features(self, g, query_id):
        ab_chain, ag_chain = extract_casp_chains(query_id)
        embeds = torch.load(get_real_path(f"{self.embed_path}/{query_id}_embed.pt"))
        feat = map_embeddings(embeds, g, False, is_merge=False, ab_chain=ab_chain, ag_chain=ag_chain)
        g.ndata['feat'] = feat

    # def __getitem__(self, idx):
    #     row = self.metadata.iloc[idx]
    #     g = self.load_graph_from_file(row['query_id'])
    #     self.generate_node_features(g, row)
    #     g = self.set_graph(g)
    #     return self.post_process(g, idx)

class AntibodyGraphDatasetFolderAF3(AntibodyGraphDatasetFolder2):
    def generate_node_features(self, g, query_id):
        chain_id = extract_af3_chains(query_id)
        embeds = torch.load(get_real_path(f"{self.embed_path}/{query_id}_embed.pt"))
        g.ndata['feat'] = map_embeddings(embeds, g, False, is_merge=False, ab_chain=chain_id, ag_chain=chain_id)

"""For nondock, return only one graph for ab & ag without binding site
"""
class AntibodyGraphDatasetND(AntibodyGraphDatasetFolder):
    def __getitem__(self, idx):
        g, lb, olb = super().__getitem__(idx)
        g = extract_graph(g, [0, 2])
        return g, lb, olb
    
class AntibodyGraphDatasetNDSep(AntibodyGraphDatasetFolder):
    def __getitem__(self, idx):
        g, lb, olb = super().__getitem__(idx)
        ab = extract_graph(g, 0)
        ag = extract_graph(g, 2)
        return (ab, ag), lb, olb

class AntibodyGraphDatasetFolderCaspND(AntibodyGraphDatasetFolderCasp):
    def __getitem__(self, idx):
        g, lb, olb = super().__getitem__(idx)
        g = extract_graph(g, [0, 2])
        return g, lb, olb

class AntibodyGraphDatasetFolderCaspNDSep(AntibodyGraphDatasetFolderCasp):
    def __getitem__(self, idx):
        g, lb, olb = super().__getitem__(idx)
        ab = extract_graph(g, 0)
        ag = extract_graph(g, 2)
        return (ab, ag), lb, olb

# use for decoy & pyg
class AntibodyGraphDatasetFolderPyG(AntibodyGraphDatasetFolder):
    def __getitem__(self, idx):
        g = self.load_graph(idx)            
        pyg = convert_dgl_to_pyg(g)
        return pyg, self.labels[idx], self.orig_labels[idx]
    
class AntibodyGraphDatasetFolderCaspPyG(AntibodyGraphDatasetFolderCasp):
    def __getitem__(self, idx):
        g, l, ol = super().__getitem__(idx)
        pyg = convert_dgl_to_pyg(g)
        return pyg, l, ol