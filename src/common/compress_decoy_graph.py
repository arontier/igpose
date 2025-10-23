import sys
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore")
import os, math, random
from tqdm import tqdm
import bz2
from argparse import ArgumentParser, Namespace
from multiprocessing import Process
import torch
import numpy as np
from common.utils import compress_edges, csr_to_coo

def extract_decoy_graph_features(g):
    return {
        'resid': g.ndata['resid'].numpy().astype(np.uint16),
        'label': g.ndata['label'].numpy().astype(np.uint8),
        'coord': g.ndata['coord'].numpy().astype(np.float32),
        'ntype': g.ndata['ntype'].numpy().astype(np.uint8),
        'chain_offset': g.ndata['chain_offset'].numpy().astype(np.uint16),
        'etype': g.edata['etype'].numpy().astype(np.uint8),
        'distance': g.edata['distance'].numpy().astype(np.float32),
    }

def compress_and_store(g, output_path):
    src, dst = compress_edges(g, csr=True)
    feats = extract_decoy_graph_features(g)
    with bz2.open(output_path, 'wb') as f:
        np.savez_compressed(f, src=src, dst=dst, **feats)

def compress_decoy_graph(input_path, output_path):
    try:
        if os.path.exists(output_path):
            return
        g = torch.load(input_path)
        compress_and_store(g, output_path)
    except:
        print('Error at', input_path)
    
def process_folder(folders, args):
    for f in tqdm(folders):
        os.makedirs(os.path.join(args.output_path, f), exist_ok=True)
        for file in os.listdir(os.path.join(args.input_path, f)):
            input_path = os.path.join(args.input_path, f, file)
            new_f = file.replace('_abg.pt', '_abg.npz')
            output_path = os.path.join(args.output_path, f, new_f)
            compress_decoy_graph(input_path, output_path)
    
def process_files(files, args):
    for file in tqdm(files):
        input_path = os.path.join(args.input_path, file)
        new_f = file.replace('_abg.pt', '_abg.npz')
        output_path = os.path.join(args.output_path, new_f)
        compress_decoy_graph(input_path, output_path)

def main_multi(args: Namespace):
    if args.is_folder:
        data = [f for f in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, f))]
    else:
        data = [f for f in os.listdir(args.input_path) if f.endswith('_abg.pt')]
    random.shuffle(data)
    max_workers = 64
    n = len(data)
    task_per_worker = math.ceil(n / max_workers)
    ranges = list(range(0, n, task_per_worker)) + [n]
    tasks = [(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)]
    print(tasks)

    processes = []
    for i, (s, e) in enumerate(tasks):
        if args.is_folder:
            p = Process(target=process_folder, args=[data[s:e], args])
        else:
            p = Process(target=process_files, args=[data[s:e], args])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-ip', '--input_path')
    parser.add_argument('-op', '--output_path')
    parser.add_argument('-f', '--is_folder', action='store_true')

    args = parser.parse_args()
    print(args)

    os.makedirs(f"{args.output_path}", exist_ok=True)
    main_multi(args)