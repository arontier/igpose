import torch
import dgl

def cdr_to_tensor(cdr):
    return torch.tensor([int(x) for x in cdr], dtype=torch.int32)

def find_cdr_node_ids(all_res_ids, chain_node_index, cdr_res_ids):
    """
    Find node indices of residues in CDR regions
    input:
        all_res_ids: get from g.ndata['resid']
        chain_node_index: (g.ndata['ntype'] == node_type).nonzero().flatten() [0,1,2]
        cdr_res_ids: get from cdr metadata
    return corresponding node indices of residues in cdr_res_ids
    """
    chain_resids = all_res_ids[chain_node_index]
    cdr_chain_indices = torch.searchsorted(chain_resids, cdr_res_ids).to(torch.int32)
    cdr_found_indices = cdr_chain_indices[(cdr_chain_indices < chain_resids.size(0)) & (chain_resids[cdr_chain_indices] == cdr_res_ids)]
    return chain_node_index[cdr_found_indices].to(torch.int32)

def extract_cdr_seeds(g, h_cdr, l_cdr):
    try:
        all_res_ids = g.ndata['resid'].to(torch.int32)
        h_index = (g.ndata['ntype'] == 0).nonzero().flatten()
        l_index = (g.ndata['ntype'] == 1).nonzero().flatten()
        
        all_nodes = []
        if len(h_index):
            h_cdr_res_ids = cdr_to_tensor(h_cdr.split(',') if isinstance(h_cdr, str) else h_cdr)
            all_nodes.append(find_cdr_node_ids(all_res_ids, h_index, h_cdr_res_ids))
        
        if len(l_index):
            l_cdr_res_ids = cdr_to_tensor(l_cdr.split(',') if isinstance(l_cdr, str) else l_cdr)
            all_nodes.append(find_cdr_node_ids(all_res_ids, l_index, l_cdr_res_ids))
        if len(all_nodes) == 1:
            return all_nodes[0]
        elif len(all_nodes) == 2:
            return torch.cat(all_nodes, dim=0)
        return None
    except:
        return None
    
def iterative_sampling(g, k, node_threshold, seeds=None, ignore_k=False):
    if len(seeds) > node_threshold:
        # sort by in_degree
        seed_degrees = g.in_degrees(seeds)
        sorted_indices = torch.argsort(seed_degrees, descending=True)
        sorted_seeds = seeds[sorted_indices]
        final_selected_nodes = set()
        for node in sorted_seeds:
            node_to_consider = set([node.item()])
            successors = g.successors(node.item()).tolist()
            node_to_consider.update(successors)
            new_node_count = len(node_to_consider.difference(final_selected_nodes))
            if not final_selected_nodes or len(final_selected_nodes) + new_node_count <= node_threshold:
                final_selected_nodes.update(node_to_consider)
            elif len(final_selected_nodes) < node_threshold:
                final_selected_nodes.update(node_to_consider)
                break
            else:
                break
            if len(final_selected_nodes) >= node_threshold:
                break
        current_selected_nodes = torch.tensor(list(final_selected_nodes), dtype=seeds.dtype, device=seeds.device)
    else:
        current_selected_nodes = seeds
        bfs_generator = dgl.bfs_nodes_generator(g, seeds)
        for i, layer_nodes in enumerate(bfs_generator):
            if i == 0:
                continue
            if not ignore_k and i > k: # only upto k layers
                break
            unique_layer_nodes = torch.unique(layer_nodes)
            new_nodes_in_layer = unique_layer_nodes[~torch.isin(unique_layer_nodes, current_selected_nodes)]
            if len(new_nodes_in_layer) == 0:
                continue
            potential_node_count = len(current_selected_nodes) + len(new_nodes_in_layer)
            if potential_node_count <= node_threshold:
                current_selected_nodes = torch.cat([current_selected_nodes, new_nodes_in_layer])
                if potential_node_count  == node_threshold:
                    break
            else:
                break
    return current_selected_nodes

def sample_k_iterations(g, k=3, node_threshold=-1, seeds=None):
    if node_threshold == -1 or g.num_nodes() < node_threshold: # k-hop sampling
        bfs_generator = dgl.bfs_nodes_generator(g, seeds)
        all_nodes = []
        # sampling khop
        for i, nodes in enumerate(bfs_generator):
            if i > k:
                break
            all_nodes.append(nodes)
        current_selected_nodes = torch.unique(torch.cat(all_nodes))
    else: # k-hop sampling with node threshold
        current_selected_nodes = iterative_sampling(g, k, node_threshold, seeds, ignore_k=False)
    
    return current_selected_nodes

def sample_khop(g, k=3, store_ids=False, node_threshold=-1, seeds=None, mode='k_iter', print_out=False):
    # g should be bidirected
    if seeds is None:
        src, dst = g.edges()
        emask = g.edata['etype'] == 1
        src, dst = src[emask], dst[emask]
        seeds = torch.unique(torch.cat([src, dst]))

    if seeds is None or not len(seeds):
        return g

    if mode == 'iter_balance':
        current_selected_nodes = iterative_sampling(g, k, node_threshold, seeds, ignore_k=True)
    else:
        current_selected_nodes = sample_k_iterations(g, k, node_threshold, seeds)
    g = dgl.node_subgraph(g, current_selected_nodes, store_ids=store_ids)
    return g