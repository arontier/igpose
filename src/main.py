"""Using for merged ab-ig graph
"""

import warnings
warnings.filterwarnings("ignore")
import os, time
from argparse import ArgumentParser
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as tp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import common.utils as utils

from common.arguments import get_common_train_args, get_distributed_args, get_syntheic_args, get_lora_args
from common.train_utils import train_gnn, get_dataloaders, setup_logger, evaluate_gnn_binary, evaluate_regression
from common.mapping_utils import is_regression, get_convolution
from common.lora import get_lora_model
from models.graph_models import GraphModel
from models.graph_models_pyg import GraphModelPyG
from models.nd_graph_models import NDGraphModel
from models.mace_model import MaceModel
from models.abepitope import AbEpitope

MODEL_MAP = {
    # 'fusionv1': GraphFusionModel,
    # 'fusionv2': GraphFusionModelv2,
    # 'fusionv3': GraphFusionModelv3,
    # 'se3': GraphSE3Model,
    'sep_nondock': NDGraphModel,
    'mace': MaceModel,
    'abepitope': AbEpitope
}

def get_model(name, convs, backend):
    convs = convs.split(',')
    name = name.lower()
    if name == 'sep_nondock': 
        return MODEL_MAP[name], get_convolution(convs[0], backend)
    if name == 'mace' or name == 'abepitope':
        return MODEL_MAP[name], None
    # conv_fn = [get_convolution(conv, backend) for conv in convs] if len(convs) != 1 else get_convolution(convs[0], backend)
    # return MODEL_MAP.get(name, GraphModel), conv_fn
    return GraphModel if backend == 'dgl' else GraphModelPyG, get_convolution(convs[0], backend)

def save_metalog(args, timestamp):
    metalog_path = os.path.join(args.log_dir, 'metalog.csv')
    if not os.path.exists(metalog_path):
        content = 'timestamp,desc'
        content += f'\n{timestamp},{args.desc}'
        with open(metalog_path, 'w') as f:
            f.write(content)
    else:
        content = f'\n{timestamp},{args.desc}'
        with open(metalog_path, 'a') as f:
            f.write(content)

def main(rank, world_size, timestamp, args):
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method=args.init_method, world_size=world_size, rank=rank)        
    torch.cuda.set_device(rank)
    device = torch.device('cuda:%i' % rank)
    is_binary = not is_regression(args.pred_loss)
    dataloaders = get_dataloaders(args.datapath, args.folderpath,
                                args.filepath, args.embed_path,
                                batch_size=args.batch_size,
                                pre_fetch=args.pre_fetch,
                                is_binary=is_binary,
                                edge_dim=args.edge_size//3,
                                world_size=world_size,
                                rank=rank,
                                args=args)

    # embed_size should be esm (1280) + 3 (h,l,ag onehot), 4 for d angles +1 for cdr onehot
    embed_size = args.embed_size
    if args.node_onehot:
        embed_size += 3
    if args.di_angle:
        embed_size += 4
    if args.cdr_onehot:
        embed_size += 1

    edge_dim = args.edge_size if not args.edge_onehot else args.edge_size + 3
    Model, conv_fn = get_model(args.model_name, args.conv_fn, args.backend)
    model_args = [] if conv_fn is None else [conv_fn]
    model_args.extend([embed_size, args.hidden_size, args.output_size, args.num_layers, args.dropout])
    model_kargs = {
        'use_edge_feat': args.use_ef,
        'edge_dim': edge_dim,
        'edge_fn': args.edge_fn, 
        'aggregation_fn': args.agg_fn,
        'pooling': args.pooling,
        'activation': args.activation,
        'mdn': True if args.mdn_factor > 0 else False,
        'node_pred': True if args.node_factor > 0 else False,
        'aggregation_mode': args.agg_mode,
        'mdn_cls': args.mdn_cls,
        'num_pred_layer': args.num_pred_layer
    }
    if args.conv_fn == 'fastegnn':
        model_kargs.update({
            'virtual_channels': 3,
            'residual': args.residual
        })
    model = Model(*model_args, **model_kargs) 

    logger = None
    if not args.distributed or (args.distributed and dist.get_rank() == 0):
        os.makedirs(f"{args.log_dir}/out/syn", exist_ok=True)
        logger = setup_logger(f"{args.log_dir}/out/syn/execution_{timestamp}")
        logger.info(args)
        logger.info("number of parameters: %i" % utils.count_parameters(model))

    if args.pretrained_ckpt:
        logger.info(f"loading pretrain model from {args.pretrained_ckpt}")
        checkpoint = torch.load(args.pretrained_ckpt)
        model_dict = model.state_dict()
        reset_layers = args.reset_layers.split(",") if args.reset_layers else ""
        filtered = {}
        
        for k, v in checkpoint.items():
            flag = False
            for l in reset_layers:
                if k.startswith(l):
                    flag = True
                    break
            if flag:
                continue
            if k in model_dict and v.size() == model_dict[k].size():
                filtered[k] = v
        print("loading weights", filtered.keys())
        # 4) update your model’s weights and load
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)

    if args.lora:
        # model must be loaded with pretrained_ckpt
        # finetuning model with lora
        model = get_lora_model(model, lora_r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=args.lora_dropout)

    model = model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    model_path = f"{args.ckptpath}/{args.model_name}_{timestamp}"
    os.makedirs(model_path, exist_ok=True)

    print("Complete initialization")
    evaluate_fn = evaluate_gnn_binary if is_binary else evaluate_regression
    train_gnn(model, evaluate_fn,
              dataloaders, args.num_epochs, args.patience,
              os.path.join(args.log_dir, 'tensorboard', str(timestamp)),
              model_path, device, logger, args, is_binary=is_binary)
    
    if args.distributed:
        dist.destroy_process_group()

def run_main():
    parser = ArgumentParser()
    get_common_train_args(parser)
    get_distributed_args(parser)
    get_syntheic_args(parser)
    get_lora_args(parser)
    args = parser.parse_args()
    if args.mask_embed:
        mp.set_start_method('spawn', force=True) 
    print(args)
    timestamp = str(int(time.time())) + '_' + uuid.uuid4().hex.upper()[0:4]
    print("Timestamp", timestamp)
    
    save_metalog(args, timestamp)
    
    if not args.distributed:
        main(args.cuda, 1, timestamp, args)
    else:
        world_size = torch.cuda.device_count()
        tp.spawn(main, args=(world_size, timestamp, args), nprocs=world_size, join=True)

# Example of usage
if __name__ == "__main__":
    run_main()