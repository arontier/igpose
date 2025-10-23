"""Inference function
"""

import warnings
warnings.filterwarnings("ignore")
import os, math
import subprocess
from tqdm import tqdm
from argparse import ArgumentParser
from types import SimpleNamespace
import yaml
from pathlib import Path
from multiprocessing import Process

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader

from common.utils import scale_probability
from common.dataset_utils import collate_simple, map_separate_embeddings, process_graph_edges
from common.sampling import sample_khop, extract_cdr_seeds
from common.cdr_utils import extract_cdr_ids
from models.ensemble import EnsembleWrapper

from datasets.dataset import AntibodyGraphDatasetFolder2
from graph_gen.make_graph_rcsb import generate_sequences, generate_graphs, load_and_merge_sequences, get_all_chain_residues
from graph_gen.make_graph import generate_residue_embeddings, generate_graph, get_sequence_from_residue_list, get_id_from_residue_list

CKPT_PREFIX = '/Arontier_1/Privates/alexbui/projects/ab_affinity/ckpt/deploy_models'
DEFAULT_CLASSIFICATION_MODEL_PATH = [os.path.join(CKPT_PREFIX, 'deployed_interface.pt'), os.path.join(CKPT_PREFIX, 'deployed_cdr_interface.pt')]
DEFAULT_REGRESSION_MODEL_PATH = [os.path.join(CKPT_PREFIX, 'deployed_interface_regression.pt'), os.path.join(CKPT_PREFIX, 'deployed_cdr_interface_regression.pt')]
# DEFAULT_MODEL_THRESHOLDS = [0.8788940740000001, 0.7968086171428571]
DEFAULT_CLASSIFICATION_THRESHOLD = 0.6 # ensemble

def convert_pdb_to_pdbqt(input_pdb: str, working_dir: str):
    """
    Convert a PDB file to PDBQT format using prepare_receptor4.
    Make sure to install AutoDockTools_py3 (python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3)

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    working_dir : str
        Directory to store the output PDBQT file.

    Returns
    -------
    str
        Path to the generated PDBQT file.
    """
    working_dir_path = Path(working_dir).resolve()

    input_pdb_path = working_dir / input_pdb
    working_dir_path.mkdir(parents=True, exist_ok=True)
    output_file = input_pdb_path.stem + ".pdbqt"
    target_pdbqt = working_dir_path / output_file

    cmd = [
        "prepare_receptor4",
        "-r", str(input_pdb_path),
        "-o", str(target_pdbqt),
        "-A", "checkhydrogens",
        "-U", "nphs_lps_waters_nonstdres"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Converted {input_pdb_path} → {target_pdbqt}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"prepare_receptor4 failed for {input_pdb}") from e

    return output_file

def extract_cdr_info(h_residues, l_residues, scheme='chothia'):
    heavy_seq = get_sequence_from_residue_list(h_residues)
    light_seq = get_sequence_from_residue_list(l_residues) 
    heavy_ids = get_id_from_residue_list(h_residues)
    light_ids = get_id_from_residue_list(l_residues)
    mole_type = "Ab" if heavy_seq and light_seq else "Nb"
    cdr_info = extract_cdr_ids(heavy_ids, light_ids, heavy_seq, light_seq, molecule_type=mole_type, scheme=scheme) 
    heavy_ids = cdr_info['heavy']
    light_ids = cdr_info['light'] if 'light' in cdr_info else ""
    return heavy_ids, light_ids

def generate_single_graph(working_path, input_file, configurations):
    # sequences
    heavy_id, heavy_index, light_id, light_index, antigen_id,antigen_index = configurations['heavy_chain_id'], configurations['heavy_chain_index'], \
                                                                            configurations['light_chain_id'], configurations['light_chain_index'], \
                                                                            configurations['antigen_chain_id'], configurations['antigen_chain_index']
    
    h_residues, l_residues, antigen_residues = get_all_chain_residues(working_path, input_file, heavy_id, light_id, antigen_id,
                                                                      heavy_index, light_index, antigen_index)
    execution_args = init_execution_args(configurations)
    # embeds
    ab_residues = [r for r in [h_residues, l_residues] if r]
    all_residues = ab_residues + [antigen_residues]
    sequences = [get_sequence_from_residue_list(res) for res in all_residues]
    embeddings = generate_residue_embeddings(0, [input_file], sequences, [len(ab_residues)+1], execution_args, return_embeds=True)[0]
    # graph
    g = generate_graph(input_file, h_residues, l_residues, antigen_residues, args=execution_args, return_graph=True)
    ab_embeds = embeddings[:len(ab_residues)]
    ag_embeds = embeddings[-1:]
    embeddings = map_separate_embeddings(ab_embeds, ag_embeds, g)
    # sampling
    g.ndata['feat'] = embeddings
    g = process_graph_edges(g)
    with g.local_scope():
        # graph sampled from interface
        g1 = sample_khop(g, 3, True, node_threshold=500, mode='k_iter')
    
    with g.local_scope():
        # graph sampled from cdr
        heavy_ids, light_ids = extract_cdr_info(h_residues, l_residues, scheme=configurations['scheme'])
        seeds = extract_cdr_seeds(g, heavy_ids, light_ids).to(torch.int64)
        cdr_marking = torch.zeros((g.num_nodes(), ), dtype=torch.int32)
        cdr_marking[seeds] = 1
        g.ndata['cdr'] = cdr_marking
        g2 = sample_khop(g, 3, True, node_threshold=500, seeds=seeds, mode='k_iter')

    return g1, g2

def load_inference_config(config_path):
    """
    Load the YAML configuration file for model inference.

    Args:
        config_path (str or Path): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
        e.g. {'models': ['/path/to/model1.pt', '/path/to/model2.pt'],
            'ensemble_threshold': 0.5,
            'working_path': '/path/to/working_path',
            'input_path': '/path/to/pdb_folder',
            'output_path': '/path/to/results.csv',
            'cuda': 0,
            'cpu_workers: 8}
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
  
    if not config.get("working_path"):
        raise ValueError("`working_path` is required.")

    if not config.get("input_path"):
        raise ValueError("`input_path` is required.")
    
    if not config.get("output_path"):
        raise ValueError("`output_path` is required.")

    if not config.get("mode"):
        config['mode'] = 'classification'
    
    config['embed_model'] = 'esm2_t6_8M_UR50D'
    config['models'] = DEFAULT_CLASSIFICATION_MODEL_PATH if config.get('mode') == 'classification' else DEFAULT_REGRESSION_MODEL_PATH

    if not config.get("cuda") and not isinstance(config.get("cuda"), int):
        config['cuda'] = -1

    if not config.get("cpu_workers"):
        config['cpu_workers'] = 1

    if not config.get("batch_size"):
        config['batch_size'] = 8

    input_path = config.get('input_path')
    if input_path.endswith('.pdb') or input_path.endswith('.pdbqt'):
        if not config.get('heavy_chain_id'):
            config['heavy_chain_id'] = 'H'

        if not config.get('heavy_chain_index'):
            config['heavy_chain_index'] = 9999

        if not config.get('light_chain_id'):
            config['light_chain_id'] = ''

        if not config.get('light_chain_index'):
            config['light_chain_index'] = 0

        if not config.get('antigen_chain_id'):
            config['antigen_chain_id'] = 'T'
        
        if not config.get('antigen_chain_index'):
            config['antigen_chain_index'] = 0

        if not config.get('scheme'):
            config['scheme'] = 'chothia'

    return config

def load_metadata(metadata_path):
    if metadata_path.endswith('.tsv'):
        metadata = pd.read_csv(metadata_path, sep='\t')
    else:
        metadata = pd.read_csv(metadata_path)
    return metadata

def generate_graph_parallel(input_path, files, metadata, args, max_workers=8):
    processes = []
    n = len(files) 
    task_per_worker = math.ceil(n / max_workers)
    ranges = list(range(0, n, task_per_worker)) + [n]
    tasks = [(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)]
    for i, (s, e) in enumerate(tasks):
        fn_args = [i, input_path, files[s:e], metadata, args]
        p = Process(target=generate_graphs, args=fn_args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def init_execution_args(configurations):
    args = SimpleNamespace(file_type='.pdbqt',
                           heavy_index=9999,
                           light_index=0,
                           antigen_index=0,
                           output_prefix=configurations['working_path'],
                           model_name=configurations['embed_model'],
                           batch_size=configurations['batch_size'],
                           repr_layer=configurations['repr_layer'],
                           all_atoms=True,
                           load_antigen=True,
                           load_embeddings=False,
                           intra_cutoff=3.5,
                           inter_cutoff=10,
                           merge_path="")
    return args

def prepare_data(configurations):
    working_path = configurations['working_path']
    metadata = load_metadata(configurations['metadata_path'])
    input_path = Path(configurations['input_path'])
    
    files = sorted([f for f in os.listdir(input_path) if f.endswith('.pdbqt')])
    if configurations['cpu_workers'] in [-1, 1]:
        generate_graphs(0, input_path, files, metadata, args)
    else:
        generate_graph_parallel(input_path, files, metadata, args, max_workers=configurations['cpu_workers'])
    generate_sequences(0, input_path, files, metadata, args)
    all_sequences, all_lengths = load_and_merge_sequences(working_path)
    generate_residue_embeddings(configurations['cuda'], files, all_sequences, all_lengths, args)

def load_single_model(model_path, mode='classification'):
    ensemble_wrapper = EnsembleWrapper(model_path, mode=mode)
    args = ensemble_wrapper.configs[0]
    return ensemble_wrapper, args

def load_models(configurations: dict):
    model_paths = configurations['models']
    models, model_configs = [], []
    for mp in model_paths:
        model, model_config = load_single_model(mp, configurations['mode'])
        models.append(model)
        model_configs.append(model_config)

    return models, model_configs

def init_dataloader(configurations, model_args, metadata):
    test_set = metadata.copy()
    working_path = configurations['working_path']
    test_dataset = AntibodyGraphDatasetFolder2(working_path, test_set, [], is_binary=True, edge_dim=model_args['edge_size']//3,
                                            is_node_onehot=model_args['node_onehot'], is_di_angle=model_args['di_angle'],
                                            is_edge_onehot=model_args['edge_onehot'], embed_path=working_path, graph_fusion=False,
                                            sample_binding_site=False, khop=3, node_threshold=500, cdr=model_args['cdr'],
                                            pooling=model_args['pooling'], sep_embed=False)
    
    dataloader = GraphDataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate_simple, num_workers=8, pin_memory=True)
    return dataloader

def load_dataloaders(configurations, model_configs, metadata):
    return [init_dataloader(configurations, config, metadata) for config in model_configs]

def single_inference(model, dataloader, device, is_cls=True):
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for g, _, _ in tqdm(dataloader):
            g = g.to(device)
            logits = model(g, g.ndata['feat'], edge_attr=g.edata['attr'] if 'attr' in g.edata else None)
            if not is_cls:
                preds.append(logits)
            else:
                probs = F.softmax(logits, dim=-1)
                probs = probs[:,1].squeeze()
                preds.append(probs)
    
    probs = torch.cat(preds, 0).cpu().numpy()
    return probs

def inference(models, dataloaders, device=None, mode='classification'):
    all_probs = [single_inference(model, dataloader, device, mode=='classification')
                 for model, dataloader in zip(models, dataloaders)]
    results = {}   
    if mode == 'classification':
        probs = scale_probability(np.mean(all_probs, axis=0), DEFAULT_CLASSIFICATION_THRESHOLD, method='asym')
        preds = (probs >= 0.5).astype(np.int32)
        results.update({'prob': probs, 'pred': preds})
    else:
        preds = np.mean(all_probs, axis=0)
        results.update({'pred': preds})
    return results

def store_results(output_path, results, metadata):
    for key, value in results.items():
        metadata[key] = value
    metadata[['file'] + [c for c in metadata.columns if 'prob' in c or 'pred' in c]].to_csv(output_path)

# prediction of a folder of graphs
def predict(configurations):    
    models, model_configs = load_models(configurations)
    metadata = load_metadata(configurations['metadata_path'])
    dataloaders = load_dataloaders(configurations, model_configs, metadata)
    device = None
    if configurations['cuda'] != -1:
        device = torch.device(f'cuda:{configurations['cuda']}')
    results = inference(models, dataloaders, device, configurations['mode'])
    store_results(configurations['output_path'], results, metadata)

# prediction of a single graph
def predict_single_graph(configurations):
    input_file = configurations['input_path']
    working_path = configurations['working_path']
    if input_file.endswith('.pdb'):
        input_file = convert_pdb_to_pdbqt(input_file, working_path)

    print(f"Step 1: generate graph for {input_file}")
    graphs = generate_single_graph(working_path, input_file, configurations)
    print("Step 2: Model execution")
    models, _ = load_models(configurations)
    preds = []
    with torch.no_grad():
        for g, m in zip(graphs, models):
            m.eval()
            logits = m(g, g.ndata['feat'], edge_attr=g.edata['attr'] if 'attr' in g.edata else None)
            if configurations['mode'] == 'classification':
                probs = F.softmax(logits, dim=-1) 
                probs = probs[:,1].squeeze() # (1,)
                preds.append(probs)
            else:
                preds.append(logits.squeeze())
    if configurations['mode'] == 'classification':
        p = scale_probability(np.mean(preds, axis=0), DEFAULT_CLASSIFICATION_THRESHOLD, method='asym')
        print(f"Probability: {p}\Class {1 if p >= 0.5 else 0}")
    else:
        p = np.mean(preds)
        print(f"Predicted score: {p}")
    

# Example of usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('config_file', type=str, default=False)
    parser.add_argument('--skip-data-prepare', action='store_true', help="If you've already generated data")

    args = parser.parse_args()
    
    configurations = load_inference_config(args.config_file)
    print(configurations)
    if not configurations['input_path'].endswith('.pdb') and not configurations['input_path'].endswith('.pdbqt'):
        if args.skip_data_prepare:
            print("Skip step 1: generate graphs")
        else:
            print("Step 1: Generate graphs")
            prepare_data(configurations)
        print("Step 2: Model execution")
        predict(configurations)
    else:
        predict_single_graph(configurations)
        

        
