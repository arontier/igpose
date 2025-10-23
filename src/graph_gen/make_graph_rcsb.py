"""
Decompose graph generation and embedding generation
Embedding generation is gpu bounded while graph generation is cpu bounded
Improve the edge distance computation via torch.cdist
"""
import os, random, re, sys
sys.path.append('..')
from pathlib import Path
import math
import warnings
warnings.filterwarnings("ignore")
import MDAnalysis as mda
import torch.multiprocessing as mp
import multiprocessing
import pandas as pd
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import numpy as np
import torch
from graph_gen.make_graph import make_common_argument, get_chain_sequence, get_chain_residues, generate_graph, generate_residue_embeddings, load_and_merge_sequences, cumsum
from common.utils import extract_file_id

def clean_pdb_name(name):
    return re.sub(r'\.polarH|\.noAltlocs|_polarH', '', name)

def get_chain_sequences(u, chain_id, start=0, end=99999):
    if not chain_id:
        return ['']
    all_chains = chain_id.split(',')
    if len(all_chains) == 0:
        return ['']

    return [get_chain_sequence(u, c, start, end) for c in all_chains]

def generate_sequences(gpu_id, parent_path, files, metadata, args):
    """Generate residue sequences to save times for generating embeddings & graphs
    """
    all_sequences = []
    seq_map = []
    ag_chain = args.antigen_chain if hasattr(args, 'antigen_chain') else ''
    
    for file in tqdm(files):
        pdb_id = Path(file).stem
        try:
            u = mda.Universe(os.path.join(parent_path, f'{pdb_id}{args.file_type}'))
        except Exception as e:
            print(e, f'{pdb_id}{args.file_type}')
    
        if not metadata is None:
            try:
                cleaned_file = clean_pdb_name(file)
                row = metadata[metadata['file'] == cleaned_file]
                heavy_chain, light_chain = row['H_pdb_chain'].values[0], row['L_pdb_chain'].values[0]
                if 'Antigen_chain' in row:
                    ag_chain = row['Antigen_chain'].values[0]
            except:
                print("error in generating seq", file)
                heavy_chain, light_chain = args.heavy_chain, args.light_chain
        else:
            heavy_chain, light_chain = args.heavy_chain, args.light_chain
        
        heavy_sequences = get_chain_sequences(u, heavy_chain, 0, args.heavy_index) if heavy_chain else []
        light_sequences = get_chain_sequences(u, light_chain, args.light_index) if light_chain else []
        antibody_sequences = heavy_sequences + light_sequences
        antigen_sequences = get_chain_sequences(u, ag_chain)
        sequences = [seq for seq in antibody_sequences+antigen_sequences if seq]
        seq_map.append(len(sequences))
        all_sequences.extend(sequences)

    prefix = f"{args.output_prefix}"
    output_path = f"{prefix}/sequence_{gpu_id}.pt"
    torch.save((all_sequences, seq_map), output_path)

def get_all_chain_residues(parent_path, pdb_id, heavy_id, light_id, ag_id, heavy_index, light_index, ag_index):
    u = mda.Universe(os.path.join(parent_path, f'{pdb_id}'))
    h_residues = get_chain_residues(u, heavy_id, 0, heavy_index)
    l_residues = get_chain_residues(u, light_id, light_index)
    antigen_residues = get_chain_residues(u, ag_id, ag_index)
    return h_residues, l_residues, antigen_residues

def generate_graphs(gpu_id, parent_path, files, metadata, args):
    desc = f"Processing PDB files on CPU {gpu_id}" if gpu_id >= 0 else "Processing PDB files on CPU"
    with tqdm(total=len(files), desc=desc) as pbar:
        for i, pdb_id in enumerate(files):
            try:
                if not metadata is None:
                    row = metadata[metadata['file'] == clean_pdb_name(pdb_id)]
                    heavy_chain, light_chain = row['H_pdb_chain'].values[0], row['L_pdb_chain'].values[0]
                    if 'Antigen_chain' in row:
                        ag_chain = row['Antigen_chain'].values[0]
                    else:
                        ag_chain = args.antigen_chain
                else:
                    heavy_chain, light_chain = args.heavy_chain, args.light_chain
                    ag_chain = args.antigen_chain
                h_residues, l_residues, antigen_residues = get_all_chain_residues(parent_path, pdb_id,
                                                                                heavy_chain, light_chain, ag_chain,
                                                                                args.heavy_index, args.light_index, args.antigen_index)
                generate_graph(pdb_id, h_residues, l_residues, antigen_residues, args=args)
            except Exception as e:
                print("ERROR: ", e, pdb_id)
            pbar.update(1)
    return True

# use this function to extract sequences of ab/ag chains in cross-dock datasets
def unify_sequences(gpu_id, files, all_sequences, all_lengths, args):
    ab_map, ag_map = {}, {}
    offset = cumsum(all_lengths)
    for i, f in enumerate(files):
        s = all_sequences[offset[i]:offset[i+1]]
        ab_id, ag_id = extract_file_id(f)
        if len(s) == 2:
            h_seq, l_seq = s[0], ""
            ag_seq = [s[1]]
        else:
            h_seq = s[0]
            l_seq = s[1] if args.light_chain else ""
            ag_seq = s[1:] if not args.light_chain else s[2:]
        
        if not ab_id in ab_map or not ab_map[ab_id]['seq']:
            seq = [seq for seq in [h_seq, l_seq] if seq]
            ab_map[ab_id] = {'seq': seq, 'len': len(seq)}
        if not ag_id in ag_map or not ag_map[ag_id]['seq']:
            ag_map[ag_id] = {'seq': ag_seq, 'len': len(ag_seq)}
    torch.save(ab_map, os.path.join(args.output_prefix, 'unify_sequence_ab.pt'))
    torch.save(ag_map, os.path.join(args.output_prefix, 'unify_sequence_ag.pt'))

# generate sequence files for ab/ag chains
def load_chain_sequences(path, prefix):
    ab_map = torch.load(path)
    files, all_sequences, all_lengths = [], [], []
    for k, v in ab_map.items():
        seq = v['seq']
        if isinstance(seq, list): # multiple chains
            all_lengths.append(v['len'])
            all_sequences += seq
        else: # only one chain
            all_sequences += [seq]
            all_lengths.append(1)
        files.append(f'{prefix}_{k}'.lower())
    return files, all_sequences, all_lengths

# def generate_chain_embeddings(gpu_id, all_sequences, offsets, args):
#     torch.cuda.set_device(gpu_id)
#     device = torch.device(f"cuda:{gpu_id}")
#     model, batch_converter = get_pretrained_model(args.model_name)
#     model = model.to(device)
#     pdb_embeddings = generate_batch_residue_embeddings(model, batch_converter, all_sequences, offsets, device, args.batch_size, args.repr_layer)

def main_multi(args: Namespace):
    start = args.start
    end = args.end
    if args.mode == 'embed_uni':
        files = None
    else:
        if args.pdb2 and '.pt' in args.pdb2:
            all_files = torch.load(args.pdb2)
        else:
            all_files = sorted([f for f in os.listdir(args.pdb) if f.endswith(f'{args.file_type}')])
            if args.mode.endswith('seq'):
                torch.save(all_files, f'{args.output_prefix}/all_files.pt')
        if args.end == -1:
            end = len(all_files)
        if end - start != len(all_files):
            files = all_files[start:end]
        else:
            files = all_files

    if args.mode == 'graph':
        random.shuffle(files)
    task_id = int(os.environ.get("SLURM_PROCID", 0))
    num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    if num_tasks > 1:
        total_files = len(files)
        files_per_task = math.ceil(total_files / num_tasks)
        task_start = task_id * files_per_task
        task_end = min((task_id + 1) * files_per_task, total_files)
        files_per_task = math.ceil(total_files / num_tasks)
        files = files[task_start:task_end]
    
    metadata = pd.read_csv(args.metapath).fillna('') if args.metapath else None # use this to load heavy chain & light chain info

    # if file order is changed before generating sequences or embeddings => mismatch in embeddings & graphs
    # graph generation is not affected by file order.
    if args.mode == 'seq':
        # extract sequences from pdbs
        process = generate_sequences
        master = multiprocessing.Process
        max_workers = 8
    elif args.mode == 'graph':
        if files:
            random.shuffle(files)
        # generate graphs for pdbqt
        process = generate_graphs
        master = multiprocessing.Process
        num_gpus = int(os.environ.get("SLURM_CPUS_ON_NODE", "-1")) 
        max_workers = num_gpus if num_gpus > 0 else 16
    elif args.mode == 'uni_seq':
        # extract sequences from pdbs and store for unique chains
        all_sequences, all_lengths = load_and_merge_sequences(f"{args.output_prefix}")
        process = unify_sequences
        master = multiprocessing.Process
        max_workers = 1
    else:
        # generate embedding for either pdb sequences or for chain sequences
        mp.set_start_method('spawn', force=True)  # Make sure to use 'spawn' for multiprocessing with CUDA
        master = mp.Process
        num_gpus = torch.cuda.device_count()
        max_workers = num_gpus if num_gpus > 0 else 1
        if args.mode.startswith('embed_uni'):
            # match with uni_seq
            files, all_sequences, all_lengths = load_chain_sequences(f"{args.output_prefix}/unify_sequence_{args.mode[-2:]}.pt", args.mode[-2:])
        else:
            # match with seq
            all_sequences, all_lengths = load_and_merge_sequences(f"{args.output_prefix}")
        process = generate_residue_embeddings
        offsets = cumsum(all_lengths)

    n = len(files) 
    task_per_worker = math.ceil(n / max_workers)
    ranges = list(range(0, n, task_per_worker)) + [n]
    tasks = [(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)]
    print(tasks)
    
    processes = []
    for gpu_id, (s, e) in enumerate(tasks):
        if args.mode == 'graph':
            fn_args = [args.pdb, files[s:e], metadata, args]
        elif args.mode == 'seq':
            fn_args = [args.pdb, files[s:e], metadata, args]
        elif args.mode == 'uni_seq':
            fn_args = [files, all_sequences, all_lengths, args]
        else:
            if args.mode.startswith('embed_uni'):
                fn_args = [files[s:e], all_sequences[offsets[s]:offsets[e]], all_lengths[s:e], args]
            else:
                fn_args = [files[s:e], all_sequences[offsets[s]:offsets[e]], all_lengths[s:e], args]
        p = master(target=process, args=[gpu_id] + fn_args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    make_common_argument(parser)
    parser.add_argument('-mp', '--metapath', default='', help='metadata contains info for heavy & light chains')
    parser.add_argument('-f2', '--pdb2', help='path to all_files.pt files which stores all files in sorted order')
    parser.add_argument('-hn', '--heavy_chain', default='A', help='A list of chains separate by ,')
    parser.add_argument('-hi', '--heavy_index', type=int, default=999)
    parser.add_argument('-ln', '--light_chain', default='', help='A list of chains separate by ,')
    parser.add_argument('-li', '--light_index', type=int, default=0)
    parser.add_argument('-agn', '--antigen_chain', default='B', help='A list of chains separate by ,')
    parser.add_argument('-agi', '--antigen_index', type=int, default=0)
    parser.add_argument('-se', '--sep_embed', action="store_true")
    parser.add_argument('-me', '--mask_embed', action="store_true", help="mask out unimportant nodes")

    args = parser.parse_args()
    print(args)

    os.makedirs(f"{args.output_prefix}", exist_ok=True)
    main_multi(args)
