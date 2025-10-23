import os
from pathlib import Path
from tqdm import tqdm
from itertools import chain
import math
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
from argparse import ArgumentParser, Namespace
import torch
import pandas as pd
import MDAnalysis as mda

from make_graph import get_single_chain_residues, generate_graph, get_pretrained_model, generate_batch_residue_embeddings, make_common_argument, get_chain_sequence

def load_and_merge_sequences(folder_path):
    # Get list of files in the folder with the pattern "embed_*.pt"
    files = [f for f in os.listdir(folder_path) if f.startswith("sequence") and f.endswith(".pt")]
    # Sort the files based on the numeric part after "embed_"
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(files)
    # Load each file and append its content to a list
    all_sequences = {}
    for file in files:
        seqs = torch.load(os.path.join(folder_path, file))
        for k, v in seqs.items():
            all_sequences['_'.join(k.split('_')[:-1])] = v
    return all_sequences

def generate_query_embeds(gpu_id, files, sequences, args):
    """Generate embeddings for all sequences of query ids
    """
    files = [f.split('/')[0] for f in files]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    model, batch_converter = get_pretrained_model(args.model_name)
    model = model.to(device)
    all_sequences = list(chain.from_iterable([sequences[f] for f in files]))
    offsets = [args.num_chains] * len(all_sequences)
    pdb_embeddings = generate_batch_residue_embeddings(model, batch_converter, all_sequences, offsets, device, args.batch_size, args.repr_layer)
    prefix = f"{args.output_prefix}"
    
    for pdb_id, embeddings in zip(files, pdb_embeddings):
        output_path = f"{prefix}/{pdb_id}_embed.pt"
        torch.save(embeddings, output_path)

def get_all_chain_residues(parent_path, pdb_id, file_type):
    u = mda.Universe(os.path.join(parent_path, f'{pdb_id}{file_type}'))
    h_residues = get_single_chain_residues(u, 'A', 0, 999)
    l_residues = get_single_chain_residues(u, 'A', 1001)
    antigen_residues = get_single_chain_residues(u, 'B', 0)
    return h_residues, l_residues, antigen_residues

def generate_sequences(gpu_id, parent_path, files, args):
    """Generate residue sequences to save times for generating embeddings & graphs
    """
    results = {}
    for pdb_id in tqdm(files):
        real_path = os.path.join(parent_path, f'{pdb_id}{args.file_type}')
        u = mda.Universe(real_path)
        heavy_sequences = get_chain_sequence(u, 'A', 0, 999)
        light_sequences = get_chain_sequence(u, 'A', 1001)
        antigen_sequences = get_chain_sequence(u, 'B', 0)
        if not heavy_sequences or not light_sequences or not antigen_sequences:
            print(pdb_id)
        if args.num_chains == 3:
            # some pdbqt indices are wrong and not start 0
            sequences = [heavy_sequences, light_sequences, antigen_sequences]
        else:
            sequences = [seq for seq in [heavy_sequences, light_sequences, antigen_sequences] if seq]
        pdb_id = Path(pdb_id).stem
        key = pdb_id.replace(args.suffix, '') if args.suffix else pdb_id
        results[key] = sequences

    output_path = f"{args.output_prefix}/sequence_{gpu_id}.pt"
    torch.save(results, output_path)

def generate_graphs(node_id, files, args):
    print(f"Processing files at {node_id}" )
    for f in tqdm(files):
        try:
            h_residues, l_residues, antigen_residues = get_all_chain_residues(args.pdb, f, args.file_type)
            generate_graph(f, h_residues, l_residues, antigen_residues, args)
        except:
            print("Error while processing:", f)

def main_multi(files, args: Namespace):
    if args.mode == 'seq':
        process = generate_sequences
        master = multiprocessing.Process
        max_workers = 4

    elif args.mode == 'graph':
        print("Generate graphs")
        process = generate_graphs
        master = multiprocessing.Process
        num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", "-1")) 
        max_workers = num_cpus if num_cpus > 0 else 16

    elif args.mode == 'embed':
        os.makedirs(args.output_prefix, exist_ok=True)
        print("Generate embeds")
        master = mp.Process
        process = generate_query_embeds
        num_gpus = torch.cuda.device_count()
        max_workers = num_gpus if num_gpus > 0 else 1
        all_sequences = load_and_merge_sequences(args.pdb)
        
    n = len(files)
    task_per_worker = math.ceil(n / max_workers)
    ranges = list(range(0, n, task_per_worker)) + [n]
    tasks = [(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)]
        
    processes = []
    for gpu_id, (s, e) in enumerate(tasks):
        if args.mode == 'seq':
            fn_args = [gpu_id, args.pdb, files[s:e], args]
        elif args.mode == 'graph':
            fn_args = [gpu_id, files[s:e], args]
        else: 
            fn_args = [gpu_id, files[s:e], all_sequences, args]
        p = master(target=process, args=fn_args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    make_common_argument(parser)
    parser.add_argument('-mp', '--metapath', default='/Arontier_1/Projects/AbAg_decoy/dataset/split/dockq_ab_c60_pos_600K.csv')
    parser.add_argument('-nc', '--num_chains', type=int, default=3)
    parser.add_argument('-sf', '--suffix', type=str, default='')
    args = parser.parse_args()
    print(args)
    all_files = pd.read_csv(args.metapath)
    if args.mode == 'embed' or args.mode == 'seq':
        metadata = all_files.groupby('query_id', as_index=False).first()
        mp.set_start_method('spawn', force=True)  # Make sure to use 'spawn' for multiprocessing with torch objects
    else:
        metadata = all_files

    metadata = metadata.sample(frac=1) # shuffle to be rebalance
    start = args.start
    end = args.end

    if args.end == -1:
        end = len(metadata)

    if end - start != len(metadata):
        ft_metadata = metadata.iloc[start:end]
    else:
        ft_metadata = metadata

    num_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    if num_tasks > 1:
        task_id = int(os.environ.get("SLURM_PROCID", 0))
        total_files = len(ft_metadata)
        files_per_task = math.ceil(total_files / num_tasks)
        task_start = task_id * files_per_task
        task_end = min((task_id + 1) * files_per_task, total_files)
        print(task_start, task_end)
        task_metadata = ft_metadata.iloc[task_start:task_end]
    else:
        task_metadata = ft_metadata
    
    print(task_metadata)
    files = [f"{q}/{q}_{f}{args.suffix}" for q,f in zip(task_metadata['query_id'], task_metadata['struct_num'])]    
    main_multi(files, args)
