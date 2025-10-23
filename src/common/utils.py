import re
import bz2
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Normal
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool
import dgl
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_curve, auc

def read_first_line(file_path: str) -> str:
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
    return first_line

def parse_namespace(logline: str) -> Namespace:
    pattern = r'Namespace\((.*)\)'
    match = re.search(pattern, logline)
    if not match:
        return
    namespace_content = match.group(1)
    namespace_dict = {}
    for item in namespace_content.split(", "):
        key, value = item.split("=")
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit() and '.' in value:  # Check if it's a float
            value = float(value)
        elif value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
            value = value[1:-1]  # Remove quotes
        
        namespace_dict[key] = value
    return namespace_dict

def extract_file_id(filename):
    base_name = filename
    if base_name.endswith(".pdbqt"):
        base_name = base_name[:-6]
    pattern = r'([A-Za-z0-9]{4})[-_]+([A-Za-z0-9]+)'
    matches = re.findall(pattern, filename)
    
    (id1, letters1), (id2, letters2) = matches[0], matches[1]
    return f'{id1}_{letters1}', f'{id2}_{letters2}'

def extract_query_id(filename):
    pattern = r"^(.*?)_(\d+)\.(.*)$"
    match = re.match(pattern, filename)
    query_id = match.group(1)
    idx = match.group(2)
    return query_id, idx

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

def save_features(tensors, path):
    bit_array = np.concatenate([tensor.flatten() for tensor in tensors])
    bit_array = np.array(bit_array, dtype=np.bool_)
    packed_bits = np.packbits(bit_array)
    shapes = np.array([tensor.shape if len(tensor.shape) > 1 else (tensor.shape[0], 1) for tensor in tensors], dtype=np.int32)
    with open(path, 'wb') as f:
        # Save the number of tensors
        f.write(len(shapes).to_bytes(4, 'little'))
        # Save the shapes
        f.write(shapes.tobytes())
        # Save the packed bits
        f.write(packed_bits.tobytes())

def load_features(path):
    with open(path, 'rb') as f:
        num_tensors = int.from_bytes(f.read(4), 'little')
        shapes = np.frombuffer(f.read(num_tensors*8), dtype=np.int32).reshape((num_tensors, -1))
        packed_bits = np.frombuffer(f.read(), dtype=np.uint8)
    bit_array = np.unpackbits(packed_bits)
    tensors = []
    index = 0
    for shape in shapes:
        num_elements = np.prod(shape)
        tensors.append(bit_array[index:index+num_elements].reshape(shape).astype(bool))
        index += num_elements
    return tensors

def store_compressed_edges(src, dst, coord, path):
    with  bz2.open(path, 'wb') as f:
        np.savez_compressed(f, src=src, dst=dst, coord=coord)

def coo_to_csr(src, dst):
    num_nodes = int(src.max()) + 1
    perm = np.argsort(src)
    src = src[perm]
    dst = dst[perm]
    rowptr = np.zeros(num_nodes+1, dtype=np.uint16)
    np.add.at(rowptr, src+1, 1)
    rowptr = np.cumsum(rowptr)
    return rowptr.astype(np.uint16), dst.astype(np.uint16)

def csr_to_coo(rowptr, dst):
    num_rows = len(rowptr) - 1
    src = np.repeat(np.arange(num_rows), rowptr[1:]-rowptr[:-1])
    return src, dst

def compress_edges(g, csr=False):
    src, dst = g.edges()
    src, dst = src.numpy(), dst.numpy()
    src, dst = src.astype(np.uint16), dst.astype(np.uint16)
    if csr:
        # Determine the number of nodes
        src, dst = coo_to_csr(src, dst)
    return src, dst

def load_compressed_edges(path):
    with bz2.open(path, 'rb') as f:
        data = np.load(f)
        src, dst = data['src'], data['dst']
        if len(src) != len(dst):
            src, dst = csr_to_coo(src, dst)
        return src, dst, data['coord']
    
def decompress_decoy_graph(path):
    with bz2.open(path, 'rb') as f:
        data = np.load(f)
        src, dst = data['src'], data['dst']
        src, dst = csr_to_coo(src, dst)
        g = dgl.graph((torch.from_numpy(src).int(), torch.from_numpy(dst).int()))
        g.ndata['resid'] = torch.from_numpy(data['resid'])
        g.ndata['label'] = torch.from_numpy(data['label']).int()
        g.ndata['coord'] = torch.from_numpy(data['coord']).float()
        g.ndata['ntype'] = torch.from_numpy(data['ntype'])
        g.ndata['chain_offset'] = torch.from_numpy(data['chain_offset']).int()
        g.edata['etype'] = torch.from_numpy(data['etype']).long()
        g.edata['distance'] = torch.from_numpy(data['distance']).float()
        return g

def generate_supernode_graph(num_nodes, device=None):
    src_nodes = []
    dst_nodes = []
    src_nodes = torch.full((num_nodes,), num_nodes, dtype=torch.long, device=device)
    dst_nodes = torch.arange(0, num_nodes, device=device)
    sg = dgl.graph((torch.cat([src_nodes, dst_nodes]), torch.cat([dst_nodes, src_nodes]))).to(device)
    return sg

def generate_supernode_batch_graph(g: dgl.DGLGraph):
    graphs = []
    supernode_idx = [-1]
    for i, num_nodes in enumerate(g.batch_num_nodes()):
        sg = generate_supernode_graph(num_nodes, g.device)
        graphs.append(sg)
        supernode_idx.append(supernode_idx[i-1] + 1 + num_nodes)
    
    # Form new bipartite graph by replacing the original edges
    new_g = dgl.batch(graphs)
    return new_g, torch.LongTensor(supernode_idx[1:])

def affinity_to_gfe_log(pk):
    C = -1.3642465608669379
    delta_G = C * pk
    return delta_G

def affinity_to_gfe(k):
    # ∆G = RT ln(K)
    C = 0.59248475334
    delta_G = C * np.log(k)
    return delta_G

def to_gibbs_energy(values, converted=False):
    # ∆G = −RT ln(10) · pKa or ∆G = RT ln(K)
    if not converted:
        vectorize_function = np.vectorize(affinity_to_gfe)
    else:
        vectorize_function = np.vectorize(affinity_to_gfe_log)
    return vectorize_function(values)

def concrete(adj, bias=0., beta=1.0, epoch=None, decay_rate=0.95):
    """Using this function to discretize states before softmax
    """
    if epoch is not None:
        beta = beta * (decay_rate ** epoch)
    random_noise = torch.rand(adj.size()).to(adj.device)
    if bias > 0. and bias < 0.5:
        r = 1 - bias - bias
        random_noise = r * random_noise + bias
    gate_inputs = torch.log(random_noise) - torch.log(1 - random_noise)
    gate_inputs = (gate_inputs + adj) / beta
    return gate_inputs

def create_attention_mask(graph: dgl.DGLGraph):
    """
    Create an attention mask for the given DGLGraph.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        The DGLGraph object.
    
    Returns
    -------
    attn_mask : torch.Tensor
        A boolean tensor of shape (batch_size, num_nodes, num_nodes) where 
        True indicates no attention (invalid connection), and False indicates valid attention.
    """
    adj = graph.adj().to_dense()
    attn_mask = (adj == 0)
    return attn_mask

def standard_norm(tensor):
    """Convert original values to std normalization (better & stable in training)
    """
    # Calculate mean and standard deviation over the N dimension (dim=0)
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    # Standard normalization
    normalized_tensor = (tensor - mean) / (std + 1e-8)  # Adding a small epsilon to avoid division by zero
    return normalized_tensor

def denormalize(tensor, mean, std):
    """Convert standard normalized value to original value
    """
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if isinstance(mean, torch.Tensor):
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    else:
        mean = torch.tensor(mean, device=tensor.device)
        std = torch.tensor(std, device=tensor.device)
    return tensor * (std + 1e-8) + mean

def reverse_transform(tensor):
    """Convert from log10 scale to original scale
    """
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    return 10**(-tensor)

def filter_edges(g, distances, threshold):
    # filter out edges by distances
    edges_to_remove = (distances > threshold).nonzero(as_tuple=True)[0]
    return g.remove_edges(edges_to_remove)

def get_relative_pos(g: dgl.DGLGraph, pos_key='coord') -> torch.Tensor:
    """Calculate x - y for edge in EGNN
    """
    x = g.ndata[pos_key]
    src, dst = g.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos

def mi_est(joint, margin):
    # v = torch.mean(torch.exp(margin))
    # fix nan issue
    n = margin.size()[0]
    mx = margin.max()
    v1 = torch.mean(joint)
    v2 = mx + torch.log(torch.sum(torch.exp(margin - mx))) - torch.log(torch.tensor([n], dtype=torch.float32).to(margin.device))
    est = v1 - v2
    return est

def mi_donsker(discriminator, embs, positive, num_graphs):
    shuffle_embs = embs[torch.randperm(num_graphs)]
    joint = discriminator(embs, positive)
    margin = discriminator(shuffle_embs, positive)
    return mi_est(joint, margin)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mean_squared_error(labels, preds):
    return ((labels - preds)**2).mean()

def mean_absolute_error(labels, preds):
    return torch.abs(labels - preds).mean()

def mean_absolute_percentage_error(labels, preds):
    return torch.mean(torch.abs((labels - preds) / (labels + 1e-8))) * 100

def precision_and_recall(preds, labels):
    """
    Compute precision and recall for binary classification.

    Args:
    - preds (torch.Tensor): Predicted labels (0 or 1), shape (batch_size,)
    - labels (torch.Tensor): Ground truth labels (0 or 1), shape (batch_size,)

    Returns:
    - precision (float): Precision score
    - recall (float): Recall score
    """
    # True Positives (TP): Both prediction and ground truth are 1
    TP = torch.sum((preds == 1) & (labels == 1)).item()
    # False Positives (FP): Prediction is 1, but ground truth is 0
    FP = torch.sum((preds == 1) & (labels == 0)).item()
    # False Negatives (FN): Prediction is 0, but ground truth is 1
    FN = torch.sum((preds == 0) & (labels == 1)).item()
    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall

def roc_curve(y_probs, y_true):
    """
    Compute ROC curve metrics: FPR, TPR, and thresholds.

    Parameters:
        y_true (torch.Tensor): True binary labels (1 or 0).
        y_probs (torch.Tensor): Predicted probabilities or scores.

    Returns:
        fpr (torch.Tensor): False Positive Rates.
        tpr (torch.Tensor): True Positive Rates.
        thresholds (torch.Tensor): Thresholds used for computing FPR and TPR.
    """
    # Ensure y_true and y_probs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_probs, torch.Tensor):
        y_probs = torch.tensor(y_probs)
    
    # Sort by predicted probabilities in descending order
    sorted_indices = torch.argsort(y_probs, descending=True)
    y_true = y_true[sorted_indices]
    y_probs = y_probs[sorted_indices]
    
    # Calculate the number of positive and negative samples
    n_pos = torch.sum(y_true)
    n_neg = y_true.size(0) - n_pos
    
    # Compute cumulative sum of true positives and false positives
    tp_cumsum = torch.cumsum(y_true, dim=0)
    fp_cumsum = torch.cumsum(1 - y_true, dim=0)

    # Calculate TPR and FPR using cumulative sums
    tpr = tp_cumsum / n_pos if n_pos > 0 else torch.zeros_like(tp_cumsum).to(y_probs.device)
    fpr = fp_cumsum / n_neg if n_neg > 0 else torch.zeros_like(fp_cumsum).to(y_probs.device)
    
    # Add initial points (0, 0) to TPR and FPR
    tpr = torch.cat([torch.tensor([0.0]).to(y_probs.device), tpr])
    fpr = torch.cat([torch.tensor([0.0]).to(y_probs.device), fpr])
    
    # Create thresholds with an extra threshold of 1 at the beginning
    thresholds = torch.cat([torch.tensor([1.0]).to(y_probs.device), y_probs])
    
    return fpr, tpr, thresholds

def distributed_roc_curve(y_probs, y_true, world_size, master_rank=0):
    """
    Compute ROC curve metrics in a distributed environment.

    Parameters:
        y_probs (torch.Tensor): Local predicted probabilities or scores.
        y_true (torch.Tensor): Local true binary labels (1 or 0).
        world_size (int): Number of nodes in the distributed environment.
        master_rank (int): Rank of the master process (default: 0).

    Returns:
        fpr (torch.Tensor): False Positive Rates.
        tpr (torch.Tensor): True Positive Rates.
        thresholds (torch.Tensor): Thresholds used for computing FPR and TPR.
    """
    # Ensure y_true and y_probs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_probs, torch.Tensor):
        y_probs = torch.tensor(y_probs)

    all_y_probs = [torch.zeros_like(y_probs) for _ in range(world_size)]
    all_y_trues = [torch.zeros_like(y_true) for _ in range(world_size)]
    dist.all_gather(all_y_probs, y_probs)
    dist.all_gather(all_y_trues, y_true)
    
    master_device = torch.device(f"cuda:{master_rank}" if torch.cuda.is_available() else "cpu")
    global_y_probs = torch.cat(all_y_probs, dim=-1).to(master_device)
    global_y_trues = torch.cat(all_y_trues, dim=-1).to(master_device)

    return roc_curve(global_y_probs, global_y_trues)
    
def define_threshold(fpr, tpr, thresholds):
    abs_diff = torch.abs(tpr - fpr)
    optimal_idx = torch.argmax(abs_diff)
    threshold = thresholds[optimal_idx]
    return threshold.item()

def accuracy(preds, labels):
    """
    Compute precision and recall for binary classification.

    Args:
    - preds (torch.Tensor): Predicted labels (0 or 1), shape (batch_size,)
    - labels (torch.Tensor): Ground truth labels (0 or 1), shape (batch_size,)

    Returns:
    - acc (float): Accuracy 
    """
    # True Positives (TP): Both prediction and ground truth are 1
    acc = torch.sum(preds == labels).item() / preds.shape[0]
    return acc

def pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient between two tensors.
    
    Args:
        x (torch.Tensor): 1D tensor of shape (n,)
        y (torch.Tensor): 1D tensor of shape (n,)
    
    Returns:
        torch.Tensor: Pearson correlation coefficient (scalar).
    """
    # Ensure that the inputs are 1-dimensional
    assert x.dim() == 1 and y.dim() == 1, "Inputs must be 1-dimensional tensors"
    assert x.size(0) == y.size(0), "Tensors must have the same length"
    
    # Mean normalization
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Compute the covariance
    covariance = torch.sum(x_centered * y_centered, dim=-1)
    
    # Compute the standard deviations
    x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=-1) + 1e-8)
    y_std = torch.sqrt(torch.sum(y_centered ** 2, dim=-1) + 1e-8)
    
    # Compute Pearson correlation coefficient
    pearson_corr = covariance / (x_std * y_std)
    
    return pearson_corr

def matthews_corrcoef(y_pred, y_true):
    """
    Compute the Matthews Correlation Coefficient (MCC).
    
    Args:
        y_true (torch.Tensor): Ground truth binary labels, shape (n_samples,).
        y_pred (torch.Tensor): Predicted binary labels, shape (n_samples,).
    
    Returns:
        torch.Tensor: The MCC score.
    """
    if y_true.dim() != 1 or y_pred.dim() != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional tensors")
    if y_true.size(0) != y_pred.size(0):
        raise ValueError("y_true and y_pred must have the same number of elements")
    
    # Compute confusion matrix elements
    tp = ((y_true == 1) & (y_pred == 1)).sum().float()  # True Positives
    tn = ((y_true == 0) & (y_pred == 0)).sum().float()  # True Negatives
    fp = ((y_true == 0) & (y_pred == 1)).sum().float()  # False Positives
    fn = ((y_true == 1) & (y_pred == 0)).sum().float()  # False Negatives

    # Compute MCC numerator and denominator
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8
    )

    # To handle division by zero
    mcc = numerator / denominator

    return mcc

def f1_score(p, r):
    denom = p + r
    return 2 * p * r / denom if denom != 0 else 0

def obtain_predictions(y_probs, y_true, threshold=None):
    # using youden j statistic to get the prediction
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_probs, y_true)
        abs_diff = torch.abs(tpr - fpr)
        optimal_idx = torch.argmax(abs_diff)
        optimal_proba_cutoff = thresholds[optimal_idx]
    else:
        optimal_proba_cutoff = threshold
    roc_predictions = (y_probs >= optimal_proba_cutoff).int()
    return roc_predictions, optimal_proba_cutoff

def obtain_predictions_f1(y_probs, y_true, threshold=None):
    if threshold is None:
        # Ensure the inputs are numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.detach().cpu().numpy()
        else:
            y_true_np = y_true

        if isinstance(y_probs, torch.Tensor):
            y_probs_np = y_probs.detach().cpu().numpy()
        else:
            y_probs_np = y_probs

        # Calculate precision, recall, and thresholds using precision_recall_curve
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true_np, y_probs_np)
        # Compute F1 scores; note the extra point in precisions and recalls is handled implicitly.
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_index = np.argmax(f1_scores)
        best_threshold = thresholds_pr[best_index] if best_index < len(thresholds_pr) else 0.5
        optimal_proba_cutoff = best_threshold
    else:
        optimal_proba_cutoff = threshold

    # Generate binary predictions
    # Ensure y_probs is a torch.Tensor for consistency
    if not isinstance(y_probs, torch.Tensor):
        y_probs = torch.tensor(y_probs)
        
    predictions = (y_probs >= optimal_proba_cutoff).int()
    return predictions, optimal_proba_cutoff

def define_threshold_f1(y_probs, y_true):
    # Ensure the inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = y_true

    if isinstance(y_probs, torch.Tensor):
        y_probs_np = y_probs.detach().cpu().numpy()
    else:
        y_probs_np = y_probs

    # Calculate precision, recall, and thresholds using precision_recall_curve
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true_np, y_probs_np)
    # Compute F1 scores; note the extra point in precisions and recalls is handled implicitly.
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_index] if best_index < len(thresholds_pr) else 0.5
    optimal_proba_cutoff = best_threshold

    return optimal_proba_cutoff

def define_threshold_precision(y_scores, y_true):
    # Ensure the inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
        y_score_np = y_scores.detach().cpu().numpy()
    else:
        y_true_np, y_score_np = y_true, y_scores
        
    precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_score_np)

    # Compute F1 for each point (we'll ignore the last p/r which has no corresponding threshold)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    # Only consider positions i where precision[i] > recall[i] and i < len(thresholds)
    valid = np.where((precisions[:-1] > recalls[:-1]))[0]

    if len(valid) > 0:
        # Find the valid index with highest F1
        best_idx = valid[np.argmax(f1_scores[valid])]
        best_threshold = thresholds[best_idx]
    else:
        # Pick threshold with highest F1 overall
        idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[idx]

    return best_threshold

def calculate_confusion_elements(y_probs, y_true, threshold):
    # Compute predictions using the threshold
    preds = (y_probs >= threshold).int()

    # Compute confusion matrix elements
    tp = ((y_true == 1) & (preds == 1)).sum().float()
    tn = ((y_true == 0) & (preds == 0)).sum().float()
    fp = ((y_true == 0) & (preds == 1)).sum().float()
    fn = ((y_true == 1) & (preds == 0)).sum().float()
    return tp, tn, fp, fn

def evaluate_metrics_from_confusion(tp, tn, fp, fn):
    # Precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor([0.0], device=tp.device)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor([0.0], device=tp.device)

    # F1 score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor([0.0], device=tp.device)

    # Matthews correlation coefficient (MCC)
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc = numerator / denominator if denominator > 0 else torch.tensor([0.0], device=tp.device)

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Return all metrics and the threshold
    metrics = {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1.item(),
        "mcc": mcc.item(),
        "accuracy": accuracy.item(),
    }

    return metrics

def evaluate_metrics(y_probs, y_true, threshold=None, optimize_for='f1'):
    """
    Evaluate binary classification metrics: precision, recall, F1 score, MCC, and accuracy.

    Args:
    - y_probs (torch.Tensor): Predicted probabilities, shape (n_samples,).
    - y_true (torch.Tensor): Ground truth binary labels, shape (n_samples,).
    - threshold (float, optional): Decision threshold for classification. If None, optimal threshold is used.

    Returns:
    - metrics (dict): Dictionary containing precision, recall, F1 score, MCC, and accuracy.
    - optimal_threshold (float): Optimal threshold (if computed).
    """
    # Determine optimal threshold using Youden's J statistic if not provided
    if threshold is None:
        if optimize_for == 'f1':
            threshold = define_threshold_f1(y_probs, y_true)
        elif optimize_for == 'precision':
            threshold = define_threshold_precision(y_probs, y_true)
        else:
            fpr, tpr, thresholds = roc_curve(y_probs, y_true)
            threshold = define_threshold(fpr, tpr, thresholds)
    
    tp, tn, fp, fn = calculate_confusion_elements(y_probs, y_true, threshold)
    metrics = evaluate_metrics_from_confusion(tp, tn, fp, fn)
    metrics['threshold'] = threshold
    return metrics

def scale_probs(scores, min_score, max_score):
    """Scale up/down given probabilities to a pre-defined range
    """
    if max_score == min_score:
        return scores
    rng = max_score - min_score
    normalized = (scores - min_score) / rng
    return torch.clamp(normalized, min=0., max=1.)


def distributed_evaluate_metrics(y_probs, y_true, world_size, master_rank=0, threshold=None):
    # Ensure y_true and y_probs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_probs, torch.Tensor):
        y_probs = torch.tensor(y_probs)

    if dist.get_rank() == master_rank:
        all_y_probs = [torch.zeros_like(y_probs) for _ in range(world_size)]
        all_y_trues = [torch.zeros_like(y_true) for _ in range(world_size)]
        dist.gather(y_probs, all_y_probs, dst=master_rank)
        dist.gather(y_true, all_y_trues, dst=master_rank)
    
        master_device = torch.device(f"cuda:{master_rank}" if torch.cuda.is_available() else "cpu")
        global_y_probs = torch.cat(all_y_probs, dim=-1).to(master_device)
        global_y_trues = torch.cat(all_y_trues, dim=-1).to(master_device)
        return evaluate_metrics(global_y_probs, global_y_trues, threshold=threshold)
    
    return None

def edge_reduce(g, values, mask=None, device=None):
    if isinstance(g, dgl.DGLGraph):
        edge_counts = torch.tensor(g.batch_num_edges(), device=device)
        graph_idx = torch.repeat_interleave(torch.arange(len(edge_counts), device=device), edge_counts)
    else:
        graph_idx = g.batch[g.edge_index[0]]

    if not mask is None:
        graph_idx = graph_idx[mask]

    num_graphs = g.batch_size
    graph_prob = scatter_add(values, graph_idx, dim=0, dim_size=num_graphs)
    return graph_prob

def mdn_score(g, pi, mu, sigma, y, is_cdr=False, aggressive=False):
    if len(y.shape) <= 1:
        y = y.unsqueeze(-1)
    if isinstance(g, dgl.DGLGraph):
        if is_cdr:
            cdr_nodes = (g.ndata['cdr'] == 1).nonzero().flatten()
            src, dst = g.edges()
            src_in = torch.isin(src, cdr_nodes)
            dst_in = torch.isin(dst, cdr_nodes)
            if aggressive:
                valid_mask = (g.edata['etype'] == 1) & (src_in | dst_in)
            else:
                valid_mask = src_in | dst_in
        else:
            valid_mask = g.edata['etype'] == 1
    else:
        valid_mask = g.etype == 1
        # implement for pyg later
    pi, mu, sigma, y = pi[valid_mask], mu[valid_mask], sigma[valid_mask], y[valid_mask]
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    prob = (torch.log(pi + 1e-10) + loglik).exp().sum(-1)
    graph_scores = edge_reduce(g, prob, valid_mask, prob.device)
    return graph_scores

def mdn_message(edges, mdn):
    edge_h = torch.cat([edges.src['h_repr'], edges.dst['h_repr']], dim=-1)
    dist = torch.norm(edges.src['h_repr'] - edges.dst['h_repr'], p=2, dim=1)
    pi, mu, sigma = mdn(edge_h)
    return {'pi': pi, 'mu': mu, 'sigma': sigma, 'dist': dist}

def pr_auc_score(y_true, y_scores):
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    return auc(recalls, precisions)

def get_interface_nodes(g):
    mask = g.edata['etype'] == 1
    src, dst = g.edges()
    src, dst = src[mask], dst[mask]
    masked_nodes = torch.unique(torch.cat([src, dst], dim=-1))
    return masked_nodes

def get_cdr_interface(g):
    cdr_nodes = (g.ndata['cdr'] == 1).nonzero().flatten()
    src, dst = g.edges()
    interface_mask = g.edata['etype'] == 1
    src, dst = src[interface_mask], dst[interface_mask]
    # filter out edges which are either start or end at cdr_nodes
    src_in = torch.isin(src, cdr_nodes)
    dst_in = torch.isin(dst, cdr_nodes)
    s_mask = src_in | dst_in
    selected_src, selected_dst = src[s_mask], dst[s_mask]
    final_nodes = torch.unique(torch.cat([selected_src, selected_dst], dim=0))
    return final_nodes

def adjust_probability(g, probs):
    with g.local_scope():
        cdr_interface = get_cdr_interface(g)
        cdr_interface_mask = torch.zeros((g.num_nodes(),), dtype=torch.float32, device=g.device)
        cdr_interface_mask[cdr_interface] = 1
        g.ndata['cdr'] = g.ndata['cdr'].float()
        g.ndata['valid_cdr'] = g.ndata['cdr'] * cdr_interface_mask
        num = dgl.sum_nodes(g, 'cdr')
        den = dgl.sum_nodes(g, 'valid_cdr')
        ratio = den / num
        output = probs * ratio.view(*probs.shape)
    return output

def tanh_pred(x, alpha=0.5):
    # f(x)= 0.5*(tanh(αx)+1)
    return 0.5 * (torch.tanh(alpha*x) + 1)

def get_regression_pred(logits, activation='', label_mean=None, label_std=None, label_norm='stdlog10'):   
    if activation == 'relu':
        logits = F.relu(logits)
    elif activation == 'elu':
        logits = F.elu(logits) + 1
    elif activation == 'tanh':
        logits = tanh_pred(logits)
        
    if label_norm.startswith('std'):
        logits = denormalize(logits, label_mean, label_std)
        if label_norm == 'stdlog10':
            logits = 10**(-torch.clamp(logits, 0, 8))
    elif activation and activation != 'none':
        logits = torch.clamp(logits, 0, 1)
    return logits

def linear_scale(p, threshold):
    if not (0 < threshold < 1):
        raise ValueError("threshold must be in (0,1)")
    
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p, dtype=float)
    m = p <= threshold
    out[m] = 0.5 * (p[m] / threshold)
    out[~m] = 0.5 + 0.5 * ((p[~m] - threshold) / (1 - threshold))
    return np.clip(out, 0.0, 1.0)

def logit_shift(p, t, eps=1e-12):
    # p′=σ(logit(p)−logit(t))
    # logit(x)=ln(x / (1 - x​))
    # σ(x)=1 / (1 + e^(−x))
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    return 1/(1 + np.exp(np.log(t/(1-t)) - np.log(p/(1-p))))

def asymmetric_power(p, threshold, alpha=1., eps=1e-12):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    beta = alpha * (np.log(threshold) / np.log(1 - threshold))
    num = p ** alpha
    det = num + (1-p) ** beta
    return num / det

def scale_probability(p, threshold, method='linear'):
    if method == 'asym':
        return asymmetric_power(p, threshold)
    elif method == 'logit_shift':
        return logit_shift(p, threshold)
    return linear_scale(p, threshold)



