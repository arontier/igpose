"""
Decompose graph generation and embedding generation
Embedding generation is gpu bounded while graph generation is cpu bounded
Improve the edge distance computation via torch.cdist
"""
import os, sys
sys.path.append('..')
import math
import warnings
warnings.filterwarnings("ignore")
import MDAnalysis as mda
import torch.multiprocessing as mp
import multiprocessing
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import dgl
from esm import pretrained
import itertools
from common.compress_decoy_graph import compress_and_store

ALL_ELEMENTS = ['A', 'C', 'OA', 'N', 'SA', 'HD', 'NA'] ###PDBQT 
amino_acids = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
}

amino_acids_dict = { 'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19 }

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

model_map = {
    'esm2_t33_650M_UR50D': pretrained.esm2_t33_650M_UR50D, # 1280 dims
    'esm2_t30_150M_UR50D': pretrained.esm2_t30_150M_UR50D, # 640 dims
    'esm2_t12_35M_UR50D': pretrained.esm2_t12_35M_UR50D, # 480 dims
    'esm2_t6_8M_UR50D': pretrained.esm2_t6_8M_UR50D # 320 dims
}

class TextDataset(Dataset):
    def __init__(self, sentences):
        """
        Args:
            sentences (list of str): List of text sentences.
        """
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return sentence

def extract_graph(g, etype):
    edge_mask = g.edata['etype'] == etype
    src, dst = g.edges()
    src, dst = src[edge_mask], dst[edge_mask]
    unique_nodes = torch.unique(torch.cat([src, dst]))
    return dgl.node_subgraph(g, unique_nodes)

def get_pretrained_model(name):
    model, alphabet = model_map[name]()
    batch_processor = alphabet.get_batch_converter()
    return model, batch_processor

def cumsum(arr):
    cumsum_array = [0]
    total = 0
    for num in arr:
        total += num
        cumsum_array.append(total)
    return cumsum_array

# def get_chain_sequence(u, chain_id):
#     if not chain_id:
#         return ''
#     sequence = [three_to_one.get(res.resname, 'X') for res in u.residues if res.resname in amino_acids and res.segid.strip() == chain_id]
#     return ''.join(sequence)

def get_sequence(u):
    return ''.join([three_to_one.get(res.resname, 'X') for res in u.residues])

def generate_sequences(gpu_id, parent_path, files, heavy_chains, lchain_chains, ag_chains, args):
    """Generate residue sequences to save times for generating embeddings & graphs
    """
    all_sequences = []
    seq_map = []
    if not isinstance(heavy_chains, str):
        pack = zip(files, heavy_chains, lchain_chains, ag_chains)
    else:
        pack = files
    for data in tqdm(pack):
        if isinstance(data, str):
            pdb_id, h, l, ag = data, heavy_chains, lchain_chains, ag_chains
        else:
            pdb_id, h, l, ag = data
        if "is_text" in args and args.is_text:
            h_sequence, antigen_sequence = torch.load(os.path.join(parent_path, f'{pdb_id}_seq.pt'))
            l_sequence = ''
        else:
            u = mda.Universe(os.path.join(parent_path, f'{pdb_id}{args.file_type}'))
            h_sequence = get_chain_sequence(u, h)
            l_sequence = get_chain_sequence(u, l)
            antigen_sequence = get_chain_sequence(u, ag)
        sequences = [seq for seq in (h_sequence, l_sequence, antigen_sequence) if seq]
        seq_map.append(len(sequences))
        all_sequences.extend(sequences)

    if hasattr(args, 'tag'):
        prefix = f"{args.output_prefix}/{args.tag}"
    else:
        prefix = f"{args.output_prefix}"
    output_path = f"{prefix}/sequence_{gpu_id}.pt"
    torch.save((all_sequences, seq_map), output_path)
   
def load_and_merge_sequences(folder_path):
    # Get list of files in the folder with the pattern "embed_*.pt"
    files = [f for f in os.listdir(folder_path) if f.startswith("sequence") and f.endswith(".pt")]
    print(files)
    if len(files) > 1:
        # Sort the files based on the numeric part after "embed_"
        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        # Load each file and append its content to a list
        all_sequences, lengths = [], []
        for file in files:
            seq, length = torch.load(os.path.join(folder_path, file))
            all_sequences.extend(seq)
            lengths.extend(length)
    else:
        all_sequences, lengths = torch.load(os.path.join(folder_path, files[0]))
    return all_sequences, lengths

def execute_embedding_model(model, batch, device, repr_layer, to_cpu=True):
    batch_tokens, lengths = batch
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    embeddings = results["representations"][repr_layer][:,1:-1,:]
    out_embeddings = []
    for i in range(len(lengths)):
        embed = embeddings[i,:lengths[i],:]
        if to_cpu:
            embed = embed.cpu()
        out_embeddings.append(embed)
    return out_embeddings

def collate_seq(batch_converter, sequences):
    data = []
    lengths = []
    for seq in sequences:
        data.append(("protein", seq))
        lengths.append(len(seq))
    return batch_converter(data)[-1], lengths

def generate_batch_residue_embeddings(model, batch_converter, all_sequences, offsets, device, batch_size, repr_layer, to_cpu=True):
    """
    Return: 
        - pdb_embeddings: a list of residue embeddings, where each element is an array [heavy_chain embedding, light_chain embedding (optional), antigen_embedding]
    """
    dataset = TextDataset(all_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_seq(batch_converter, x))

    all_embeddings = []
    for batch in tqdm(dataloader):
        all_embeddings.extend(execute_embedding_model(model, batch, device, repr_layer, to_cpu))
    end, start = 0, 0
    pdb_embeddings = []
    for l in offsets:
        end += l
        temp = all_embeddings[start:end]
        pdb_embeddings.append(temp)
        start = end
    return pdb_embeddings

def generate_residue_embeddings(gpu_id, files, all_sequences, offsets, args, return_embeds=False):
    """Generate embeddings for all sequences of all graph
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    model, batch_converter = get_pretrained_model(args.model_name)
    model = model.to(device)
    pdb_embeddings = generate_batch_residue_embeddings(model, batch_converter, all_sequences, offsets, device, args.batch_size, args.repr_layer)
    if not return_embeds:
        if hasattr(args, 'tag'):
            prefix = f"{args.output_prefix}/{args.tag}"
        else:
            prefix = f"{args.output_prefix}"
        
        for pdb_id, embeddings in zip(files, pdb_embeddings):
            output_path = f"{prefix}/{pdb_id}_embed.pt"
            torch.save(embeddings, output_path)
        return
    return pdb_embeddings

def get_embeddings(model, batch_converter, sequence):
    """for a single graph
    """
    model.eval()
    with torch.no_grad():
        data = [("protein", sequence)]
        _, _, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        embeddings = results["representations"][33][0, 1:-1].cpu().numpy()
    return embeddings

def get_batch_embeddings(model, batch_converter, sequences, device=None):
    data = [("protein", sequence) for sequence in sequences]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    model.eval()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    embeddings = results["representations"][33][:,1:-1,:]
    out_embeddings = [embeddings[i,:len(sequences[i]),:].cpu() for i in range(len(sequences))]
    return out_embeddings

def load_and_merge_embeddings(folder_path):
    # Get list of files in the folder with the pattern "embed_*.pt"
    files = [f for f in os.listdir(folder_path) if f.startswith("embed_") and f.endswith(".pt")]
    
    # Sort the files based on the numeric part after "embed_"
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Load each file and append its content to a list
    embeddings = []
    for file in files:
        embeddings.extend(torch.load(os.path.join(folder_path, file)))
        
    return embeddings

def get_single_chain_residues(u, chain_id, start, end=99999):
    if not chain_id:
        return []
    return [res for res in u.residues \
                if res.resname in amino_acids \
                    and res.segid.strip() == chain_id \
                    and res.resid >= start \
                    and res.resid <= end]

def get_chain_residues(u, chain_id, start, end=99999):
    if not chain_id:
        return []
    
    all_chains = chain_id.split(',')
    if not len(all_chains):
        return []
    
    if len(all_chains) == 1:
        return get_single_chain_residues(u, all_chains[0], start, end)
    
    all_chains = set(all_chains)
    outputs = [res for res in u.residues \
                if res.resname in amino_acids \
                    and res.segid.strip() in all_chains \
                    and res.resid >= start \
                    and res.resid <= end]
    return outputs

def get_chain_sequence(u, chain_id, start, end=99999):
    if not chain_id:
        return ''
    return ''.join([three_to_one.get(res.resname, 'X') \
                        for res in u.residues 
                            if res.resname in amino_acids \
                                and res.segid.strip() == chain_id \
                                and res.resid >= start \
                                and res.resid <= end])

def get_sequence_from_residue_list(residue_list):
    return ''.join([three_to_one.get(res.resname, 'X') for res in residue_list if res.resname in amino_acids])

def get_id_from_residue_list(residue_list):
    return [res.resid for res in residue_list]

def get_center_mass(res):
    return res.atoms.center_of_mass()

def check_connect(u, i, j):
    print(u, i, j)
    if abs(i-j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i+1].get_connections("bonds"))
        nb3 = len(u.residues[i:i+2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0

def get_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi*0.01, psi*0.01, omega*0.01, chi1*0.01]
    except:
        return [0, 0, 0, 0]
    
# def get_self_dist(res):
# 	try:
# 		xx = res.atoms
# 		dists = distances.self_distance_array(xx.positions)
# 		ca = xx.select_atoms("name CA")
# 		c = xx.select_atoms("name C")
# 		n = xx.select_atoms("name N")
# 		o = xx.select_atoms("name O")
# 		return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
# 	except:
# 		return [0, 0, 0, 0, 0]

def get_ca_pos(res):
    return res.atoms.select_atoms("name CA").positions[0]

def compute_batch_distance(X, Y, threshold=3.5):
    distances =  torch.cdist(X, Y, p=2)    
    src, dst = torch.where(distances <= threshold)
    edge_distances = distances[src, dst]
    return (src, dst), edge_distances.tolist()

def construct_fully_connected_graph(residues1, residues2=None, all_atoms=True):
    num_nodes1 = len(residues1)
    if residues2 is None:
        # Create source and destination indices for a fully connected graph
        src = torch.arange(num_nodes1).repeat(num_nodes1)
        dst = torch.arange(num_nodes1).repeat_interleave(num_nodes1)
        mask = src < dst
        src = src[mask]
        dst = dst[mask]
        # Create the graph using the edge indices
        graph = dgl.graph((src, dst), num_nodes=num_nodes1)
        all_residues = residues1
    else:
        # construct bipartite graph
        num_nodes2 = len(residues2)
        src = torch.arange(num_nodes1).repeat_interleave(num_nodes2)
        dst = torch.arange(num_nodes2).repeat(num_nodes1)
        graph = dgl.graph((src, dst+num_nodes1))
        all_residues = residues1 + residues2
        num_nodes1 += num_nodes2

    c_alpha_positions = torch.zeros((num_nodes1, 3), dtype=torch.float32)

    if all_atoms:
        max_num_atoms = max(len(r.atoms) for r in all_residues) 
        center_masses = torch.zeros((num_nodes1, 3), dtype=torch.float32)
        atom_lengths = []
        atom_positions = torch.zeros((num_nodes1, max_num_atoms * 3), dtype=torch.float32) 
    
    valid_c_alpha_masks = []
    for i, r in enumerate(all_residues):
        position = torch.from_numpy(r.atoms.positions).flatten()
        if all_atoms:
            atom_lengths.append(len(r.atoms))
            if len(r.atoms) < max_num_atoms:
                atom_positions[i,:] = F.pad(position, (0, (max_num_atoms-len(r.atoms)) * 3), mode='constant', value=0.)
            else:
                atom_positions[i,:] = position
            center_masses[i,:] = torch.from_numpy(get_center_mass(r))

        try:
            ca_pos = get_ca_pos(r)
            valid_c_alpha_masks.append(1)
        except:
            ca_pos = np.array([0, 0, 0])
            valid_c_alpha_masks.append(0)
        c_alpha_positions[i,:] = torch.from_numpy(ca_pos)

    if all_atoms:
        graph.ndata['atom_pos'] = atom_positions
        graph.ndata['cm_pos'] = center_masses
        graph.ndata['offset'] = torch.tensor(atom_lengths, dtype=torch.int32)
    
    graph.ndata['ca_pos'] = c_alpha_positions
    graph.ndata['valid_calpha'] = torch.tensor(valid_c_alpha_masks)
    return graph

def distance_message(edges, threshold, all_atoms):
    # distance between atoms    
    if all_atoms:
        num_edges, num_atoms = edges.src['atom_pos'].shape[0], edges.src['atom_pos'].shape[-1] // 3
        src = edges.src['atom_pos'].view(num_edges, num_atoms, 3)
        dst = edges.dst['atom_pos'].view(num_edges, num_atoms, 3)
        distances = torch.cdist(src, dst, p=2)
        mask1 = torch.arange(num_atoms).unsqueeze(0).expand(num_edges, num_atoms) < edges.src['offset'].unsqueeze(1)
        mask2 = torch.arange(num_atoms).unsqueeze(0).expand(num_edges, num_atoms) < edges.dst['offset'].unsqueeze(1)
        mask = mask1.unsqueeze(2) & mask2.unsqueeze(1)
        distances[~mask] = float('inf')
        distances, _ = torch.min(distances.view(distances.shape[0], -1), dim=-1)

        # distance between center of mask
        src, dst = edges.src['cm_pos'], edges.dst['cm_pos']
        cm_distances = torch.cdist(src.unsqueeze(1), dst.unsqueeze(1), p=2).squeeze(1)

    # distance between ca
    src, dst = edges.src['ca_pos'], edges.dst['ca_pos']
    ca_distances = torch.cdist(src.unsqueeze(1), dst.unsqueeze(1), p=2).squeeze(1)

    if all_atoms:
        all_distances = torch.cat([distances.unsqueeze(1), ca_distances, cm_distances], dim=1)
    else:
        all_distances = ca_distances
        distances = ca_distances.squeeze()
    return {'distance': all_distances, 'mask': distances <= threshold}

def filter_edges_by_distances(g, threshold, all_atoms):
    g.apply_edges(lambda x: distance_message(x, threshold, all_atoms))
    src, dst = g.edges()
    return (src[g.edata['mask']], dst[g.edata['mask']]), g.edata['distance'][g.edata['mask']]


def create_graph(residues_list, embeddings_list, threshold=10., intra_inter='intra', all_atoms=True):
    # threshold is 3.5 for inter and 10. for intra
    all_residues = list(itertools.chain(*residues_list))
    all_embeddings = torch.cat(embeddings_list, dim=0) if embeddings_list and not embeddings_list[0] is None else None

    # Edge 생성 단계
    edge_indices, distances = [], []
    unique_nodes = set([])
    if intra_inter == 'intra':
        # 같은 그룹 내의 잔기 쌍에 대해 엣지 생성 (거리 <= 3.5Å)
        fg = construct_fully_connected_graph(all_residues, all_atoms=all_atoms)
        edges, distances = filter_edges_by_distances(fg, threshold, all_atoms)
        for s, d in zip(*edges):
            s, d = s.item(), d.item()
            if not all_atoms and (not fg.ndata['valid_calpha'][s].item() or not fg.ndata['valid_calpha'][d].item()):
                continue
            unique_nodes.update([all_residues[s], all_residues[d]])
            edge_indices.append((all_residues[s], all_residues[d]))
            
    elif intra_inter == 'inter':
        # Bipartite edges between antibody and antigen residues, d <= 15Å
        fg = construct_fully_connected_graph(*residues_list, all_atoms=all_atoms)
        edges, distances = filter_edges_by_distances(fg, threshold, all_atoms)
        for s, d in zip(*edges):
            s, d = s.item(), d.item()
            if not all_atoms and (not fg.ndata['valid_calpha'][s].item() or not fg.ndata['valid_calpha'][d].item()):
                continue
            unique_nodes.update([all_residues[s], all_residues[d]])
            edge_indices.append((all_residues[s], all_residues[d]))

    mapping_ids = None
    if unique_nodes and len(unique_nodes) != len(all_residues) and not all_embeddings is None:
        # some residues are not included in graphs
        residue_map = {r: i for i, r in enumerate(all_residues)}
        mapping_ids = torch.tensor([residue_map[n] for n in unique_nodes], dtype=torch.int32)
        all_embeddings = all_embeddings[mapping_ids]
    elif not unique_nodes:
        warnings.warn("graph is empty: " + intra_inter) 

    return edge_indices, unique_nodes, all_embeddings, distances
  
def merge_graphs(list_edge_indices, list_unique_nodes, list_node_embeddings, distances, edge_types):
    embedding_map = {}
    all_residues = []
    idx = 0
    for unique_nodes, embeddings in zip(list_unique_nodes, list_node_embeddings):
        for i, node in enumerate(unique_nodes):
            if not node in embedding_map:
                i_embed = embeddings[i, :] if not embeddings is None else None
                embedding_map[node] = (idx, i_embed)
                all_residues.append(node)
                idx += 1

    all_edges_ids = []
    for edge_indices, distance, type in zip(list_edge_indices, distances, edge_types):
        for (src, dst), d in zip(edge_indices, distance):
            if not src in embedding_map or not dst in embedding_map:
                continue
            s_idx, d_idx = embedding_map[src][0], embedding_map[dst][0]
            all_edges_ids.append((s_idx, d_idx, d.tolist(), type))

    final_embeddings = None
    if not list_node_embeddings[0] is None:
        final_embeddings = torch.cat([embedding_map[n][1].unsqueeze(0) for n in all_residues], dim=0)
    all_edges_ids.sort()
    return all_edges_ids, all_residues, final_embeddings

def make_residue_map(residues):
    return {r:i for i, r in enumerate(residues)}

def get_all_chain_residues(parent_path, pdb_id, heavy_id, light_id, ag_id, file_type):
    u = mda.Universe(os.path.join(parent_path, f'{pdb_id}{file_type}'))
    h_residues = get_single_chain_residues(u, heavy_id, 0)
    l_residues = get_single_chain_residues(u, light_id, 0)
    antigen_residues = get_single_chain_residues(u, ag_id, 0)
    return h_residues, l_residues,  antigen_residues

def generate_graph(pdb_id, h_residues, l_residues, antigen_residues, args, return_graph=False):
    """Generate graphs
    """
    if hasattr(args, 'tag'):
        save_dir = f"{args.output_prefix}/{args.tag}"
    else:
        save_dir = f"{args.output_prefix}"

    name = pdb_id if not hasattr(args, 'suffix') else pdb_id.replace(args.suffix, '')
    name = name.replace('.polarH', '').replace('.noAltlocs', '')
    
    if not return_graph:
        output_path = f"{save_dir}/{name}_abg.npz"

        if os.path.exists(output_path):
            return
    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_embeddings = None
    if args.load_embeddings:
        p = pdb_id
        if '/' in pdb_id:
            p = p.split('/')[0]
        embed_path = os.path.join(save_dir, f"{p}_embed.pt")
        all_embeddings = torch.load(embed_path)

    antibody_embeddings = all_embeddings[:-1] if not all_embeddings is None else []
    antigen_embeddings = all_embeddings[-1] if not all_embeddings is None else None

    if args.merge_path:
        existing_g = torch.load(f"{args.merge_path}/{pdb_id}_abg.pt")
        ab_g = extract_graph(existing_g, 0)
        ab_chain_offset = ab_g.ndata['chain_offset']
        ab_chain_types = ab_g.ndata['ntype']
        ab_distances = ab_g.edata['distance']
        abg_src, abg_dst = ab_g.edges()
        ab_edges, ab_nodes, ab_node_embeds = [], set([]), None
        for s, d in zip(abg_src, abg_dst):
            st, dt = ab_chain_types[s.item()], ab_chain_types[d.item()]
            s_residues = h_residues if st == 0 else l_residues
            d_residues = h_residues if dt == 0 else l_residues
            s, d = ab_chain_offset[s.item()].item(), ab_chain_offset[d.item()].item()
            ab_edges.append((s_residues[s], d_residues[d]))
            ab_nodes.update([s_residues[s], d_residues[d]])

        ig_g = extract_graph(existing_g, 1)
        ig_chain_offset = ig_g.ndata['chain_offset']
        ig_chain_types = ig_g.ndata['ntype']
        ig_distances = ig_g.edata['distance']
        igg_src, igg_dst = ig_g.edges()
        ig_edges, ig_nodes, ig_node_embeds = [], set([]), None
        for s, d in zip(igg_src, igg_dst):
            st, dt = ig_chain_types[s.item()], ig_chain_types[d.item()]
            if st == 0:
                s_residues = h_residues
            elif st == 1:
                s_residues = l_residues
            else:
                s_residues = antigen_residues
            if dt == 0:
                d_residues = h_residues
            elif dt == 1:
                d_residues = l_residues
            else:
                d_residues = antigen_residues
            s, d = ig_chain_offset[s.item()].item(), ig_chain_offset[d.item()].item()
            ig_edges.append((s_residues[s], d_residues[d]))
            ig_nodes.update([s_residues[s], d_residues[d]])
    else:
        antibody_residues = h_residues + l_residues
        ab_edges, ab_nodes, ab_node_embeds, ab_distances = create_graph(
            residues_list=[antibody_residues],
            embeddings_list=antibody_embeddings,
            intra_inter='intra',
            threshold=args.intra_cutoff,
            all_atoms=args.all_atoms
        )

        antibody_embeddings = torch.cat(antibody_embeddings, dim=0) if antibody_embeddings and not antibody_embeddings[0] is None else None
        ig_edges, ig_nodes, ig_node_embeds, ig_distances = create_graph(
            residues_list=[antibody_residues, antigen_residues],
            embeddings_list=[antibody_embeddings, antigen_embeddings],
            intra_inter='inter',
            threshold=args.inter_cutoff,
            all_atoms=args.all_atoms
        )

    # generate intra edges for antigen at binding site
    if not args.load_antigen:
        # edge type: 0 - intra edge, 1 - inter edge
        all_edges_ids, all_residues, node_embeds = merge_graphs([ab_edges, ig_edges], [ab_nodes, ig_nodes], [ab_node_embeds, ig_node_embeds], [ab_distances, ig_distances], [0, 1])
        all_residue_set = make_residue_map(all_residues)
        filter_ag_residues = [r for r in antigen_residues if r in all_residue_set]
        if len(filter_ag_residues) > 1:
            ag_edges, _, _, ag_distances = create_graph(
                residues_list=[filter_ag_residues],
                embeddings_list=None,
                intra_inter='intra',
                threshold=args.intra_cutoff,
                all_atoms=args.all_atoms
            )
            if ag_edges:
                all_edges_ids += [(all_residue_set[s], all_residue_set[t], d, 2) for (s, t), d in zip(ag_edges, ag_distances)]
    else:
        ag_edges, ag_nodes, ag_node_embeds, ag_distances = create_graph(
            residues_list=[antigen_residues],
            embeddings_list=[antigen_embeddings],
            intra_inter='intra',
            threshold=args.intra_cutoff,
            all_atoms=args.all_atoms
        )
        all_edges_ids, all_residues, node_embeds = merge_graphs([ab_edges, ag_edges, ig_edges],
                                                                [ab_nodes, ag_nodes, ig_nodes],
                                                                [ab_node_embeds, ag_node_embeds, ig_node_embeds],
                                                                [ab_distances, ag_distances, ig_distances],
                                                                [0, 2, 1])

    src, dst, distances, types = zip(*all_edges_ids)
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)))  

    # get center mass for residue, node_type: 0-H, 1-L, 2-AG
    h_residues, l_residues, antigen_residues = make_residue_map(h_residues), make_residue_map(l_residues), make_residue_map(antigen_residues)
    coor_feats, res_ids, node_type, chain_offsets = [], [], [], []
    for k in all_residues:
        # angles.append(get_dihediral_angles(k))
        coords = get_center_mass(k) if args.all_atoms else get_ca_pos(k)
        coor_feats.append(torch.from_numpy(coords).unsqueeze(0))
        res_ids.append(k.resid)   
        if k in h_residues:
            # hchain
            chain_offsets.append(h_residues[k])
            node_type.append(0)
            del h_residues[k]
        elif k in l_residues:
            # lchain
            node_type.append(1)
            chain_offsets.append(l_residues[k])
            del l_residues[k]
        elif k in antigen_residues:
            # agchain
            chain_offsets.append(antigen_residues[k])
            node_type.append(2)
            del antigen_residues[k]
        else:
            raise ValueError(f"Please check graph data, mismatch in residues: {pdb_id}")
    g.ndata['resid'] = torch.tensor(res_ids, dtype=torch.int32)
    g.ndata['label'] = torch.tensor([amino_acids_dict[r.resname] for r in all_residues], dtype=torch.int32)
    g.ndata['coord'] = torch.cat(coor_feats, dim=0)
    g.ndata['ntype'] = torch.tensor(node_type, dtype=torch.uint8)
    g.ndata['chain_offset'] = torch.tensor(chain_offsets, dtype=torch.int32)
    # g.ndata['angle'] = torch.tensor(angles, dtype=torch.float32)
    if not node_embeds is None:
        g.ndata['feat'] = node_embeds
    g.edata['etype'] = torch.tensor(types, dtype=torch.uint8)
    g.edata['distance'] = torch.tensor(distances)
    if args.all_atoms:
        g.edata['distance'] = g.edata['distance'].view((g.num_edges(), 3))
    if not return_graph:
        compress_and_store(g, output_path)
        return
    return g

def generate_graphs(gpu_id, parent_path, files, heavy_chains, lchain_chains, ag_chains, args):
    desc = f"Processing PDB files on CPU {gpu_id}" if gpu_id >= 0 else "Processing PDB files on CPU"
    with tqdm(total=len(files), desc=desc) as pbar:
        
        if not isinstance(heavy_chains, str):
            pack = zip(files, heavy_chains, lchain_chains, ag_chains)
        else:
            pack = zip(files)
        for i, data in enumerate(pack):
            pdb_id = data[0]
            h, l, ag = data[1:4] if len(data) > 1 else (heavy_chains, lchain_chains, ag_chains)
            try:
                h_residues, l_residues, ag_residues = get_all_chain_residues(parent_path, pdb_id, h, l, ag, args.file_type)
                generate_graph(pdb_id, h_residues, l_residues, ag_residues, args)
            except Exception as e:
                print("ERROR", e, pdb_id)
            pbar.update(1)
    return True

def main_multi(args: Namespace):
    metadata = pd.read_csv(os.path.join('../data', f'metadata_{args.tag}_filtered.csv'))
    start = args.start
    end = args.end
    if args.end == -1:
        end = len(metadata)
    if end - start != len(metadata):
        ft_metadata = metadata.iloc[start:end]
    else:
        ft_metadata = metadata
    
    files = ft_metadata['pdb']
    heavy_chains = ft_metadata['Hchain']
    lchain_chains = ft_metadata['Lchain']
    ag_chains = ft_metadata['antigen_chain']

    if args.mode == 'seq':
        process = generate_sequences
        master = multiprocessing.Process
        num_gpus = 4
        max_workers = num_gpus if num_gpus > 0 else 1
    elif args.mode == 'graph':
        process = generate_graphs
        master = multiprocessing.Process
        # num_gpus = os.cpu_count()
        num_gpus =  1
        max_workers = num_gpus if num_gpus > 0 else 1
    else:
        process = generate_residue_embeddings
        master = mp.Process
        num_gpus = torch.cuda.device_count()
        max_workers = num_gpus if num_gpus > 0 else 1
        all_sequences, all_lengths = load_and_merge_sequences(f"{args.output_prefix}/{args.tag}")
        offsets = cumsum(all_lengths)

    n = len(ft_metadata)
    task_per_worker = math.ceil(n / max_workers)
    ranges = list(range(0, n, task_per_worker)) + [n]
    tasks = [(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)]
    
    processes = []
    for gpu_id, (s, e) in enumerate(tasks):
        if args.mode == 'graph':
            fn_args = [args.pdb, files[s:e], heavy_chains[s:e], lchain_chains[s:e], ag_chains[s:e], args]
        elif args.mode == 'seq':
            fn_args = [args.pdb, files[s:e], heavy_chains[s:e], lchain_chains[s:e], ag_chains[s:e], args]
        else:
            fn_args = [files[s:e], all_sequences[offsets[s]:offsets[e]], all_lengths[s:e], args]
        p = master(target=process, args=[gpu_id] + fn_args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def make_common_argument(parser):
    parser.add_argument("-m", "--mode", default="graph", help="graph|embed embed ~ generate residue embeddings, graph ~ make graph")
    parser.add_argument("-mn", "--model_name", default="esm2_t6_8M_UR50D", help="https://github.com/facebookresearch/esm")
    parser.add_argument("-r", "--repr_layer", help="representation layer", default=6, type=int)
    parser.add_argument("-f", "--pdb", help="path to pdb folder", default="/Arontier_1/Projects/AbAg_decoy/dataset/structures")
    parser.add_argument("-mf", "--merge_path", help="path to existing ab ig file to merge to save time (syn only)", default="")
    parser.add_argument("-o", "--output_prefix", default="../data/abg_syn")
    parser.add_argument("-s", "--start", help="Start index", default=0, type=int)
    parser.add_argument("-e", "--end", help="End index", default=-1, type=int)
    parser.add_argument("-b", "--bins", help="number of bins for edge embedding", default=10, type=int)
    parser.add_argument("-ca", "--intra_cutoff", default=3.5, type=float)
    parser.add_argument("-ci", "--inter_cutoff", default=10., type=float)
    parser.add_argument("-bs", "--batch_size", default=32, type=int)
    parser.add_argument("-ag", "--load_antigen", action="store_true", default=False, help="make full intra graph for antigen")
    parser.add_argument("-le", "--load_embeddings", action="store_true", default=False)
    parser.add_argument("-ft", "--file_type", type=str, default='.pdbqt', help=".pdbqt or .pdb")
    parser.add_argument("-at", "--all_atoms", action="store_true", default=False)

def make_sabdab_argument(parser):
    parser.add_argument("-t", "--tag", help="peptide|protein", default='protein')

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Make sure to use 'spawn' for multiprocessing with CUDA

    parser = ArgumentParser()
    make_common_argument(parser)
    make_sabdab_argument(parser)
    args = parser.parse_args()

    os.makedirs(f"{args.output_prefix}/protein", exist_ok=True)
    os.makedirs(f"{args.output_prefix}/peptide", exist_ok=True)
    main_multi(args)
