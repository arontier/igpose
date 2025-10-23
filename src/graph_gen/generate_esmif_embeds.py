import os, sys
from pathlib import Path
from tqdm import tqdm
MODULE_DIR = str( Path( Path(__file__).parent.resolve() ) )
sys.path.append(MODULE_DIR)
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
import esm

from biopdb_utilities import read_pdb_structure, read_cif_structure, is_pdb_file, is_cif_file

class ESMIF1Model():
    
    def __init__(self, device=None, esmif1_modelpath=None):

        self.device = device
        model_name = "esm_if1_gvp4_t16_142M_UR50"
        torch_hubdir = torch.hub.get_dir()
        torch_hubfile = Path(torch_hubdir) / "checkpoints" / f"{model_name}.pt"

        #download model from torchhub, (esmif1 source code hardcoded to load onto cpu)
        if esmif1_modelpath is None and not torch_hubfile.is_file():
            esmif1_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()       

        #if model already in torchhub, load directly onto device for speed up 
        elif esmif1_modelpath is None and torch_hubfile.is_file():
            model_data = torch.load(str(torch_hubfile), map_location=self.device)
            esmif1_model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)

        #load model from specified location
        else:
            model_data = torch.load(str(esmif1_modelpath), map_location=self.device)
            esmif1_model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)

        self.esmif1_model = esmif1_model.eval().to(self.device)
        self.alphabet = alphabet

class ChainItemDataset(Dataset):
    def __init__(self, items):
        # items is a list of tuples (coord_cat, confidence, seq)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_chain_items(batch_converter, batch, device):
    # batch: list of (coord_cat, confidence, seq)
    # we need to return exactly what your esmif1encs_forwardpass expects:
    #  batch_coords, confidence, tokens, padding_mask
    # actually we can reuse your batch_converter
    coord_cats, _, seqs, lengths = zip(*batch)
    batch = [(coord_cats[i], None, seqs[i]) for i in range(len(seqs))]
    batch_coords, confidence_tensor, _, tokens, padding_mask = batch_converter(batch, device=device)
    return batch_coords, confidence_tensor, tokens, padding_mask, lengths

def store_embeds(output_path, input_files, embeddings):
    for emb, file in zip(embeddings, input_files):
        fname = file.name.replace('.pdb', '.pdbqt_embed.pt')
        torch.save(emb, os.path.join(output_path, fname))

def generate_folder_chain_embeddings(folder: Path,
                                     output_path: str,
                                     batch_size: int = 8,
                                     gpu: int = 0,
                                     chain_orders: str = 'ABCDE',
                                     start=0,
                                     end=-1):
    """
    Returns:
        List[List[Tensor]]  where each sublist is the list of residue embeddings
        for all chains in one PDB/CIF file, in sorted file order.
    """
    device = torch.device(f'cuda:{gpu}')
    model = ESMIF1Model(device=device)

    # 1) Gather all chains into a flat list:
    items = []      # to be fed into DataLoader
    offsets = []    # how many chains per file
    pdb_paths = sorted([p for p in folder.iterdir()
                        if is_pdb_file(p) or is_cif_file(p)])
    if end != -1:
        pdb_paths = pdb_paths[start:end]
    chain_orders = {c: i for i, c in enumerate(chain_orders)}
    print("loading files")
    for pdb_file in pdb_paths:
        # read structure
        if is_pdb_file(pdb_file):
            struct = read_pdb_structure(pdb_file)
        else:
            struct = read_cif_structure(pdb_file)
        chain_ids = [c.get_id() for c in struct.get_chains()]

        # use your existing utilities to get coords & native_seqs
        loaded = esm.inverse_folding.util.load_structure(str(pdb_file), chain_ids)
        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(loaded)

        offsets.append(len(chain_ids))
        chain_content = [None] * len(chain_orders)
        for c in chain_ids:
            seq = native_seqs[c]
            coord_cat = esm.inverse_folding.multichain_util._concatenate_coords(coords, c)
            chain_content[chain_orders[c]] = (coord_cat, None, seq, len(seq))
        items.extend([c for c in chain_content if not c is None])

    batch_converter = esm.inverse_folding.util.CoordBatchConverter(model.alphabet)

    # 2) Build DataLoader
    dataset = ChainItemDataset(items)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=lambda x: collate_chain_items(batch_converter, x, device))

    # 3) Run batched inference
    print("generate embeddings")
    all_embs = []
    for batch_coords, confidence_tensor, _, padding_mask, lengths in tqdm(dataloader):
        with torch.no_grad():
            enc_out = model.esmif1_model.encoder.forward(
                batch_coords.to(model.device),
                padding_mask.to(model.device),
                confidence_tensor.to(model.device),
                return_all_hiddens=False
            )["encoder_out"][0]
            # strip BOS/EOS and take chain-length tokens
            # NOTE: adjust these indices if your model outputs differ
            emb_batch = enc_out[1:-1,:,:].cpu().transpose(0,1)  # (batch, seq_len, hidden_dim)
            for emb_seq, L in zip(emb_batch, lengths):
                emb = emb_seq[:L,:]
                all_embs.append(emb)  # [L, hidden_dim]

    # 4) Re‐group into one list per PDB
    pdb_embeddings = []
    start = 0
    for n_chains in offsets:
        end = start + n_chains
        pdb_embeddings.append(all_embs[start:end])
        start = end

    store_embeds(output_path, pdb_paths, pdb_embeddings)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path')
    parser.add_argument('-o', '--output_path')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-c', '--chain_orders', default='ABCDE')
    parser.add_argument('-s', '--start', type=int, default=0, required=False)
    parser.add_argument('-e', '--end', type=int, default=-1, required=False)

    args = parser.parse_args()
    print(args)
    generate_folder_chain_embeddings(Path(args.input_path), args.output_path, args.batch_size, args.gpu, args.chain_orders)

