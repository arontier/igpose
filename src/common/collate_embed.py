import sys
sys.path.append('..')
from functools import partial
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import dgl
from esm import pretrained
import numpy as np
from common.sampling import sample_khop

embedding_model = None
batch_converter = None
AMINO_ACIDS = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

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

def clean_graph(g):
    ntype = g.ndata['ntype']
    nfeat = g.ndata.pop('feat') if 'feat' in g.ndata else None
    efeat = g.edata.pop('attr') if 'attr' in g.edata else None
    coord = g.ndata.pop('coord')
    etype = g.edata['etype']
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

    g.edata['etype'] = etype
    if not efeat is None:
        g.edata['attr'] = efeat
    return g   

def init_embedding_model(model_name, device):
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    return model, batch_converter

def get_chain_sequences(g, sg):
    h_index = (g.ndata['ntype'] == 0).nonzero().flatten()
    l_index = (g.ndata['ntype'] == 1).nonzero().flatten()
    a_index = (g.ndata['ntype'] == 2).nonzero().flatten()
    sequences = np.array([AMINO_ACIDS[a] for a in g.ndata['label']])
    masks = np.array(['<mask>'] * g.num_nodes())
    selected_nodes = sg.ndata['_ID']
    masks[selected_nodes] = sequences[selected_nodes]
    # sequences corresponding to each chains in original graphs
    h_sequences = masks[h_index]
    l_sequences = masks[l_index]
    a_orders = np.argsort(g.ndata['chain_offset'][a_index]) # only antigen nodes are shuffled after merging
    a_sequences = masks[a_index][a_orders]
    return h_sequences, l_sequences, a_sequences

# truncate <mask> at start and end of sequences
def truncate_sequences(sequence):
    # Find the first valid amino acid token
    first_valid = next((i for i, token in enumerate(sequence) if token != '<mask>'), None)
    # Find the last valid amino acid token by iterating in reverse
    last_valid = next((i for i, token in enumerate(reversed(sequence)) if token != '<mask>'), None)
    if first_valid is not None and last_valid is not None:
        # Convert last_valid from reverse index to actual index
        last_valid = len(sequence) - last_valid - 1
        # Slice the list from the first valid token to the last valid token (inclusive)
        filtered_sequence = sequence[first_valid:last_valid + 1]
    else:
        # No valid amino acid found, return an empty list or handle as needed
        filtered_sequence = []
    return filtered_sequence, last_valid

def generate_batch_sequences(batch_g, batch_sg):
    all_sequences, valid_indices = [], []
    tracking = [0]
    offset = 0
    for i, (g, sg) in enumerate(zip(batch_g, batch_sg)):
        try:
            h_sequences, l_sequences, a_sequences = get_chain_sequences(g, sg)
            h_sequences, h_e = truncate_sequences(h_sequences)
            c_offset = 0
            if not h_e is None:
                all_sequences.append(''.join(h_sequences))
                c_offset += 1
            l_sequences, l_e = truncate_sequences(l_sequences)
            if not l_e is None:
                all_sequences.append(''.join(l_sequences))
                c_offset += 1
            a_sequences, a_e = truncate_sequences(a_sequences)
            if not a_e is None:
                all_sequences.append(''.join(a_sequences))
                c_offset += 1
            offset += c_offset
            valid_indices.append(i)
            tracking.append(offset)
        except Exception as e:
            print('error at', i, e)
    return all_sequences, tracking, valid_indices

def collate_seq(batch_converter, sequences):
    data = []
    lengths = []
    for seq in sequences:
        data.append(("protein", seq))
        lengths.append(len(seq))
    batch_tokens = batch_converter(data)[-1]
    masks = torch.where((batch_tokens == 0) | (batch_tokens == 1) | (batch_tokens == 2) | (batch_tokens == 32), 0, 1)
    return batch_tokens, masks

def realign_subgraph_embeddings(sg, embeddings):
    all_orders = []
    offsets = sg.ndata['chain_offset']
    h_index = (sg.ndata['ntype'] == 0).nonzero().flatten()
    if len(h_index):
        all_orders.append(offsets[h_index])
    l_index = (sg.ndata['ntype'] == 1).nonzero().flatten()
    if len(l_index):
        all_orders.append(offsets[l_index])
    a_index = (sg.ndata['ntype'] == 2).nonzero().flatten()
    if len(a_index):
        all_orders.append(offsets[a_index])
    aligned_embeddings = []
    for order, embedding in zip(all_orders, embeddings):
        _, index = torch.sort(order)
        aligned_embeddings.append(embedding[index])

    return torch.cat(aligned_embeddings, dim=0)

def generate_dynamic_embeddings(model, batch_converter, batch_g, batch_sg, repr_layers, device=None):
    all_sequences, tracking, valid_indices = generate_batch_sequences(batch_g, batch_sg)
    dataset = TextDataset(all_sequences)
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=lambda x: collate_seq(batch_converter, x))

        all_embeddings = []
        for batch, masks in dataloader:
            batch = batch.to(device)
            results = model(batch, repr_layers=[repr_layers], return_contacts=False)
            reprs = results['representations'][repr_layers]
            l = len(batch)
            for i in range(l):
                valid_masks = masks[i,:].bool()
                rep = reprs[i, valid_masks].detach().cpu()
                all_embeddings.append(rep)
    
    batch_new_g = []
    for i in range(len(tracking)-1):
        sg = batch_sg[valid_indices[i]]
        graph_reps = [all_embeddings[j] for j in range(tracking[i], tracking[i+1])]
        graph_rep_tensor = realign_subgraph_embeddings(sg, graph_reps)
        sg.ndata['feat'] = graph_rep_tensor # should be N x D
        clean_graph(sg)
        batch_new_g.append(sg)
    return batch_new_g, valid_indices

# using for khop & dynamic embeddings
def collate_embed(batch, model_name, repr_layer, khop, dynamic_khop=False, node_threshold=600, device=None): 
    global embedding_model, batch_converter
    if embedding_model is None or batch_converter is None:
        embedding_model, batch_converter = init_embedding_model(model_name, device)
    orig_labels, labels, batch_g, batch_sg = [], [], [], []
    for all_g, lb, orig_lb in batch:
        if all_g is None: continue
        batch_g.append(all_g)
        rand_khop = np.random.randint(2, khop+1) if dynamic_khop else khop
        batch_sg.append(sample_khop(all_g, rand_khop, store_ids=True, node_threshold=node_threshold))
        labels.append(lb)
        orig_labels.append(orig_lb)

    if isinstance(labels[0], torch.Tensor):
        lb_type = labels[0].dtype
    elif isinstance(labels[0], int) or isinstance(labels[0], bool):
        lb_type = torch.int64
    else:
        lb_type = torch.float32
    # process embedding here
    batch_sg, valid_indices = generate_dynamic_embeddings(embedding_model, batch_converter, batch_g, batch_sg, repr_layer, device)
    if len(batch_sg) != len(labels):
        labels = [labels[i] for i in valid_indices]
        orig_labels = [orig_labels[i] for i in valid_indices]

    batch_sg = dgl.batch(batch_sg)
    return batch_sg, torch.tensor(labels, dtype=lb_type), torch.tensor(orig_labels, dtype=torch.float32)

def get_collate_embed(model_name, repr_layer, khop, **kargs):
    return partial(collate_embed, model_name=model_name, repr_layer=repr_layer, khop=khop, **kargs)