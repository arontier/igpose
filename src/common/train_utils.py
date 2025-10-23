import sys
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore")
import os, random, logging, time, copy
from tqdm import tqdm
from argparse import Namespace
from typing import Callable
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
# from apex.optimizers import FusedAdam
from torch.optim import Adam
from torch.amp import GradScaler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
import numpy as np
import higher 

from dgl.dataloading import GraphDataLoader

from models.graph_models import GraphModel
from datasets.dataset import AntibodyGraphDatasetFolder, AntibodyGraphDatasetFolderPyG, AntibodyGraphDatasetND, AntibodyGraphDatasetNDSep
from common import utils
from common.dataset_utils import collate_simple, collate_pyg, create_weighted_sampler, compute_class_weights, DistributedWeightedSampler
from common.collate_embed import get_collate_embed
from common.meta_sampler import ClassAwareSampler, MetaSampler, SampleLearner
from common.lora import save_lora_model
from common.utils import evaluate_metrics, mdn_score, get_regression_pred
from common.mapping_utils import get_loss_function, is_regression
from common.losses import compute_class_frequence, confidence_penalty, ranking_loss, xai_loss, coeff_loss


class AntibodyDatasetFiles(Dataset):
    def __init__(self, files, label_mean, label_std):
        self.files = files
        self.label_mean = label_mean
        self.label_std = label_std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def split_folders(root_dir, train_ratio=0.6, eval_ratio=0.2):
    """Splits the folders into train, eval, and test sets."""
    all_folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    random.shuffle(all_folders)
    
    n_train = int(train_ratio * len(all_folders))
    n_eval = int(eval_ratio * len(all_folders))
    
    train_folders = all_folders[:n_train]
    eval_folders = all_folders[n_train:n_train + n_eval]
    test_folders = all_folders[n_train + n_eval:]
    
    return train_folders, eval_folders, test_folders

def setup_logger(log_file):
    logger = logging.getLogger('train_log')
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    
    return logger

def load_embeds(embed_path):
    return {f: [e.cpu() for e in torch.load(os.path.join(embed_path, f))] for f in os.listdir(embed_path)}

def load_separate_embeds(embed_path):
    return {'ab': torch.load(os.path.join(embed_path, 'ab_embed.pt')), 'ag': torch.load(os.path.join(embed_path, 'ag_embed.pt'))}

def get_working_set(metafiles_df, folders, key):
    if folders:
        folders = folders.split(',')
        filtered_df = metafiles_df[metafiles_df['folder'].isin(folders)]
        train_set = filtered_df[filtered_df[key] == 1]
    else:
        train_set = metafiles_df[metafiles_df[key] == 1]
    return train_set

def get_dataloader_class(backend, model_name):
    if backend == 'pyg':
        return AntibodyGraphDatasetFolderPyG
    if model_name == 'nondock':
        return AntibodyGraphDatasetND
    if model_name == 'sep_nondock':
        return AntibodyGraphDatasetNDSep
    return AntibodyGraphDatasetFolder

def get_dataloaders(root_dir, meta_folder, meta_files, embed_path, batch_size=32,
                    pre_fetch=True, is_binary=False, edge_dim=10, world_size=1, rank=0, args=None):
    metafiles_df = pd.read_csv(meta_files)
    metafiles_df = metafiles_df.sample(frac=1)
    if args.label_clamp:
        s, e = args.label_clamp.split(',')
        print('filter out sample at the boundary with DockQ', s, e)
        metafiles_df = metafiles_df[(metafiles_df['DockQ'] < float(s)) | (metafiles_df['DockQ'] >= float(e))].reset_index()

    metafiles_df['label'] = (metafiles_df['DockQ'] >= args.label_threshold).astype(int)
    train_set = get_working_set(metafiles_df, args.train_dataset, 'train')
    val_set = get_working_set(metafiles_df, args.val_dataset, 'val')
    test_set = get_working_set(metafiles_df, args.test_dataset, 'test')
    print(len(train_set), len(val_set), len(test_set))
    all_embeds = []

    device = torch.device(f'cuda:{args.cuda}')
    collate_kargs = {'node_threshold': args.node_threshold, 'device': device}
    if args.backend == 'dgl':
        # nondock mode only support dgl for now
        DataCls, GraphLoader = get_dataloader_class(args.backend, args.model_name), GraphDataLoader
        collate = collate_simple if not args.mask_embed else get_collate_embed(args.embed_model, args.repr_layer, args.khop, dynamic_khop=args.dynamic_khop, **collate_kargs)
    else:
        collate = collate_pyg
        DataCls, GraphLoader = get_dataloader_class(args.backend, args.model_name), DataLoader
    print('Data class & graph loader', DataCls, GraphLoader)

    kargs = {
        'pre_fetch': pre_fetch,
        'use_ef': args.use_ef,
        'is_binary': is_binary,
        'edge_dim': edge_dim,
        'is_node_onehot': args.node_onehot,
        'is_di_angle': args.di_angle,
        'is_edge_onehot': args.edge_onehot,
        'khop': args.khop,
        'embed_path': embed_path,
        'sep_embed': args.sep_embed,
        'sample_binding_site': args.binding_site,
        'mask_embed': args.mask_embed,
        'augmentation': args.augmentation,
        'sampling_method': args.sampling_method,
        'node_threshold': args.node_threshold,
        'rc_node_threshold': args.rc_node_threshold,
        'cdr': args.cdr,
        'cdr_onehot': args.cdr_onehot,
        'pooling': args.pooling,
        'edge_distance_threshold': args.edge_distance_threshold,
        'label_norm': args.label_norm
    }
    print(kargs)
    train_dataset = DataCls(root_dir, train_set, all_embeds, **kargs)
    label_mean, label_std = train_dataset.label_mean, train_dataset.label_std
    kargs.update({
        'label_mean': label_mean,
        'label_std': label_std,
        'augmentation': False,
        'dynamic_khop': False
    })
    eval_dataset = DataCls(root_dir, val_set, all_embeds, **kargs)
    test_dataset = DataCls(root_dir, test_set, all_embeds, **kargs)

    if world_size > 1:
        sample_weights = compute_class_weights(train_dataset.labels, args.sample_type, class_ratio=args.class_ratio)
        train_sampler = DistributedWeightedSampler(train_dataset, sample_weights, num_replicas=world_size, rank=rank)    
    else:
        if args.meta_sampler:
            sampler_learner = SampleLearner(2).to(device)
            train_sampler = MetaSampler(train_dataset.labels, batch_size, sampler_learner, device)
            meta_sampler = ClassAwareSampler(train_dataset.labels, is_infinite=True)
            meta_loader = iter(GraphDataLoader(train_dataset, batch_size=batch_size, sampler=meta_sampler, collate_fn=collate_simple, num_workers=8, pin_memory=True))
        else:
            if args.class_ratio == '':
                train_sampler = RandomSampler(train_dataset)
            else:
                train_sampler = create_weighted_sampler(train_dataset.labels, args.sample_type, class_ratio=args.class_ratio)
            sampler_learner, meta_loader, meta_sampler = None, None, None
    print('train_sampler', train_sampler)
        
    # Create dataloaders
    loader_kargs = {'batch_size': batch_size, 'collate_fn': collate, 'num_workers': 4 if not args.mask_embed else  4, 'pin_memory': True}
    train_loader = GraphLoader(train_dataset, sampler=train_sampler, **loader_kargs)
    # don't use dynamic khop for test & eval
    if args.backend == 'dgl' and args.mask_embed:
        collate = get_collate_embed(args.embed_model, args.repr_layer, args.khop, dynamic_khop=False, **collate_kargs)
    loader_kargs['collate_fn'] = collate
    eval_loader = GraphLoader(eval_dataset, shuffle=False, **loader_kargs)
    test_loader = GraphLoader(test_dataset, shuffle=False, **loader_kargs)
    
    return {
        'train': train_loader,
        'val': eval_loader,
        'test': test_loader,
        'all_embeds': all_embeds,
        'meta': meta_loader,
        'learner': sampler_learner,
        'sampler': meta_sampler
    }

def get_log_str(val_results):
    if val_results is None:
        return ""
    val_str = ""
    for k, v in val_results.items():
        val_str += f"{k}: {v:.4f} "
    return val_str

def log_results(epoch, epochs, execution_time, train_loss, val_results, logger):
    """Log results to the console. val_mae & val_mape will become precision and recall for classification
    """
    log_str = f"Epoch {epoch + 1}/{epochs} in {execution_time:.2f}s, Train Loss: {train_loss:.4f} Val Results: "
    logger.info(log_str + get_log_str(val_results))

def dockq_to_class(values):
    return (values >= 0.8).to(torch.int64)

def inference_gnn(model, loader, device=None, **kargs):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            g, batch_labels, batch_orig_labels = data
            if isinstance(g, tuple) or isinstance(g, list):
                outputs = model(*[sg.to(device) for sg in g])
            else:
                g = g.to(device)
                if kargs['backend'] == "dgl":
                    outputs = model(g, g.ndata['feat'], edge_attr=g.edata['attr'] if 'attr' in g.edata else None)
                else:
                    outputs = model(g)
            
            logits = outputs['logits']
            if batch_labels.dtype == torch.float32:
                # utils.reverse_transform(utils.denormalize(logits.squeeze(-1), loader.dataset.label_mean, loader.dataset.label_std))
                preds.append(get_regression_pred(outputs['logits'], kargs['pred_fn'], loader.dataset.label_mean, loader.dataset.label_std, loader.dataset.label_norm))
                labels.append(batch_orig_labels.to(device))
            else:
                probs = F.softmax(logits / kargs['temp'], dim=-1)
                preds.append(probs)
                labels.append(batch_labels.to(device))
    labels, preds = torch.cat(labels, dim=0), torch.cat(preds, dim=0) 
    return labels, preds

# Evaluation function for GNN
def evaluate_gnn_binary(model, loader, device=None, grad_scale=False, optimize_for='tpr', **kargs):
    labels, preds = inference_gnn(model, loader, device, **kargs)
    probs = preds[:,1].squeeze()
    if grad_scale or torch.isnan(probs).any():
        probs_mask = ~torch.isnan(probs)
        probs = probs[probs_mask]
        labels = labels[probs_mask]
    if 'logger' in kargs and not kargs['logger'] is None:
        logger = kargs['logger']
        logger.info(f"Pred stats - mean: {probs.mean().item()}, std: {probs.std().item()}, min: {probs.min().item()}, max: {probs.max().item()}")
    threshold = kargs['threshold'] if 'threshold' in kargs else None
    metrics = evaluate_metrics(probs, labels, threshold, optimize_for=optimize_for)
    return metrics

def evaluate_regression(model, loader, device=None, **kargs):
    labels, preds = inference_gnn(model, loader, device, **kargs)
    preds = preds.flatten()
    mae = utils.mean_absolute_error(labels, preds)
    coeff = coeff_loss(preds, labels)
    order = ranking_loss(preds, labels)
    metrics = {'mae': mae, 'coeff': coeff, 'order': order}
    if 'logger' in kargs and not kargs['logger'] is None:
        logger = kargs['logger']
        logger.info(f"Pred stats - mean: {preds.mean().item()}, std: {preds.std().item()}, min: {preds.min().item()}, max: {preds.max().item()}")
    return metrics

def meta_step(meta_model, sample_learner, g, labels, meta_loader, model_optimizer, meta_optimizer, device, args):
    LossFn = get_loss_function(args.pred_loss)
    criteria = LossFn(reduction='none')
    sample_learner.train()
    if args.lora:
        temp = copy.deepcopy(meta_model)
        model = temp.merge_and_unload()
        model.train()
        model_optimizer = Adam(model.parameters(), lr=args.lr)
        for p in model.parameters():
            p.requires_grad_(True)
        model = model.to(g.device)
        # print(meta_model, model)
    else:
        model = meta_model

    model_optimizer.zero_grad()
    meta_optimizer.zero_grad()
    with higher.innerloop_ctx(model, model_optimizer) as (fmodel, diffopt):
        # optain surrogate model
        with g.local_scope():
            outputs = fmodel(g, g.ndata['feat'], edge_attr=g.edata['attr'] if 'attr' in g.edata else None)
            loss = criteria(outputs['logits'], labels)
            loss = sample_learner.forward_loss(loss) # plug sample weights to grad graphs
            diffopt.step(loss)

            # optimize sample weights via surrogate model
            g, val_labels, _ = next(meta_loader)
            val_labels = val_labels.to(device)
            g = g.to(device)
            val_outputs = fmodel(g, g.ndata['feat'], edge_attr=g.edata['attr'] if 'attr' in g.edata else None)
            val_loss = criteria(val_outputs['logits'], val_labels).mean()
            val_loss.backward()
            meta_optimizer.step()
    sample_learner.eval()

def train_step(model, data, epoch, device, loss_fn, optimizer, scaler, args,
                meta_optimizer=None, meta_loader=None, meta_learner=None,
                writer=None, global_steps=0, is_binary=True, **kargs):
    """Perform a single training step."""
    g, batch_labels, batch_orig_labels = data
    batch_labels = batch_labels.to(device)
    if isinstance(g, list) or isinstance(g, tuple):
        g = [sg.to(device) for sg in g]
    else:
        g = g.to(device)

    # meta optimizer
    if args.meta_sampler:
        meta_step(model, meta_learner, g, batch_labels, meta_loader, optimizer, meta_optimizer, device, args)

    if args.backend == 'dgl':
        if isinstance(g, list):
            outputs = model(*g)
        else:
            edge_feat = g.edata['attr'] if 'attr' in g.edata else None
            outputs = model(g, g.ndata['feat'], edge_feat)
    else:
        outputs = model(g)
    
    logits = outputs['logits']

    loss = 0

    if args.cls_factor > 0.:
        args.temp = 1 if not args.temp else args.temp
        loss += args.cls_factor * loss_fn(logits/args.temp, batch_labels)
    
    if not is_binary:
        logits = get_regression_pred(logits, args.pred_fn, kargs['label_mean'], kargs['label_std'], kargs['label_norm'])

    # add mdn loss
    if args.mdn_factor > 0.:
        graph_scores = mdn_score(g, outputs['pi'], outputs['mu'], outputs['sigma'], outputs['dist']) # B x 1 
        scores = logits[:,1] if len(logits.shape) > 1 and logits.shape[-1] != 1 else logits
        loss -= args.mdn_factor * torch.corrcoef(torch.stack([scores.squeeze(), graph_scores]))[1, 0]

    # add res classification loss
    if args.node_factor > 0.:
        if args.backend == 'dgl':
            ntype = g.ndata['ntype'].to(torch.int64)
        else:
            ntype = g.ntype.to(torch.int64)
        node_loss = F.cross_entropy(outputs['node_logits'], ntype)
        loss += node_loss * args.node_factor

    if args.coeff_factor > 0.:
        batch_orig_labels = batch_orig_labels.to(device)
        scores = logits[:,1] if len(logits.shape) > 1 and logits.shape[-1] != 1 else logits
        # do the correction for the sign
        loss -= args.coeff_factor * torch.corrcoef(torch.stack([scores.squeeze(), batch_orig_labels]))[1, 0]

    if args.rank_factor > 0.:
        # use when predict regression score
        batch_orig_labels = batch_orig_labels.to(device)
        loss += args.rank_factor * ranking_loss(logits, batch_orig_labels)

    # try threshold loss
    if args.prob_factor > 0.:
        probs = F.softmax(logits, dim=-1)
        pos_idx = batch_labels[(batch_labels == 1).nonzero().flatten()]
        neg_idx = batch_labels[(batch_labels == 0).nonzero().flatten()]
        neg_probs, pos_probs = probs[:,0][neg_idx], probs[:,1][pos_idx]
        loss += args.prob_factor * (F.relu(0.5 - neg_probs).sum() + F.relu(0.5 - pos_probs).sum())

    if args.conf_factor > 0.:
        probs = F.softmax(logits, dim=-1)
        loss += args.conf_factor * confidence_penalty(probs[:,1].flatten(), batch_labels.flatten(), epsilon=0.1).sum()

    if args.xai_factor > 0.:
        x_cls_loss, x_comp_loss = xai_loss(g, outputs['h_repr'], outputs['g_h'], batch_labels, model.predictor, loss_fn)
        loss += args.xai_factor * x_comp_loss + x_cls_loss

    optimizer.zero_grad()
    if not args.grad_scale:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    
    if not writer is None:
        writer.add_scalar("Loss/train_steps", loss.item(), global_steps)

    return loss

def initialize_criteria(args, labels=None):
    LossFn = get_loss_function(args.pred_loss)
    fn_args = []
    if args.pred_loss == 'bs':
        fn_args = [compute_class_frequence(labels)]
    elif args.pred_loss == 'ws':
        fn_args = [compute_class_weights(labels)]
    elif args.pred_loss.startswith('wsb'):
        fn_args = [args.bins]
    return LossFn(*fn_args)

# Train function for GNN
def train_gnn(model: GraphModel,
              evaluate_gnn: Callable,
              dataloaders: dict,
              epochs: int,
              patience: int,
              log_dir: str,
              model_save_path: str,
              device: torch.device,
              logger: logging.Logger,
              args: Namespace,
              **kargs):
    if not is_regression(args.pred_loss) and args.output_size < 2:
        raise ValueError("output size must be >=2 to use cross entropy loss")
    
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']

    # Initialize optimizer, loss, and early stopping
    if args.meta_sampler:
        optimizer = Adam(model.parameters(), lr=args.lr)
        meta_optimizer = Adam(dataloaders['learner'].parameters(), lr=0.01) 
        meta_loader = dataloaders['meta'] 
        meta_learner = dataloaders['learner'].to(device)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
        meta_optimizer, meta_loader, meta_learner = None, None, None

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr/10)
    loss_fn = initialize_criteria(args, train_loader.dataset.labels if not is_regression(args.pred_loss) else None)
    print("loss function", loss_fn, train_loader.dataset.label_mean, train_loader.dataset.label_std)
    early_stopping = EarlyStopping(patience=patience)

    # TensorBoard setup
    global_steps = 0
    writer = SummaryWriter(log_dir=log_dir) if (not args.distributed or dist.get_rank() == 0) else None
    
    best_val = args.best_val
    world_size = dist.get_world_size() if args.distributed else 1
    scaler = GradScaler()

    for epoch in range(epochs):
        print('training epoch:', epoch)
        s = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        valid_loss, train_loss = 0, 0
        for data in tqdm(train_loader):
            loss = train_step(model, data, epoch, device, loss_fn, optimizer, scaler, args,
                            meta_optimizer=meta_optimizer, meta_loader=meta_loader, meta_learner=meta_learner,
                            writer=writer, global_steps=global_steps, is_binary=kargs['is_binary'],
                            label_mean=train_loader.dataset.label_mean, label_std=train_loader.dataset.label_std,
                            label_norm=train_loader.dataset.label_norm)
            if not torch.isnan(loss).all():
                train_loss += loss.item()
                valid_loss += 1
                global_steps += 1
        
        scheduler.step()
        if not valid_loss:
            continue

        avg_train_loss = train_loss / valid_loss
        
        if args.distributed:
            train_loss_tensor = torch.tensor(avg_train_loss).to(device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / world_size
            
        if (not args.distributed or dist.get_rank() == 0):
            metrics = evaluate_gnn(model, val_loader, device,
                                logger=logger, distributed=args.distributed,
                                backend=args.backend,
                                grad_scale=args.grad_scale,
                                optimize_for=args.pred_fn,
                                pred_fn=args.pred_fn,
                                temp=args.temp,
                                **kargs)
            e = time.time() - s
            log_results(epoch, epochs, e, avg_train_loss, metrics, logger)

            # Save model if validation loss improves
            # Early stopping
            metric_type = 'f1_score' if kargs['is_binary'] else 'coeff'
            early_stopping_score = -metrics['f1_score'] if kargs['is_binary'] else metrics[metric_type]
            early_stopping(early_stopping_score)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
            
            save_criteria = (metrics['f1_score'] > best_val and metrics['threshold'] >= 0.1) if kargs['is_binary'] else (metrics[metric_type] < best_val)
            if save_criteria or args.force_eval or epoch == 0:
                logger.info(f"Validation loss improved. Saving model...")
                best_val = metrics[metric_type]
                if args.lora:
                    save_lora_model(model, os.path.join(model_save_path, f'lora_model_{epoch}'))
                else:
                    torch.save(model.state_dict(), os.path.join(model_save_path, f'best_{epoch}.pth'))

                test_metrics = evaluate_gnn(model, test_loader, device,
                                            threshold=metrics['threshold'] if kargs['is_binary'] else 0,
                                            distributed=args.distributed,
                                            non_dock=args.grad_scale,
                                            backend=args.backend,
                                            pred_fn=args.pred_fn,
                                            temp=args.temp)
                logger.info("Test result " + get_log_str(test_metrics))

    if not writer is None:
        writer.close()

