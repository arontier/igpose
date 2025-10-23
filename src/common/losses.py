import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, pos, neg):
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)
        logit_pos = (query * pos).sum(dim=1, keepdim=True) # B x 1
        logit_neg = query @ neg.T # B x B
        logits = torch.cat([logit_pos, logit_neg], dim=1) # B x (B+1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
class WeightedSoftmaxLoss(nn.Module):
    """
    """
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        self.class_weights = self.class_weights.to(targets.device)
        return F.cross_entropy(logits, targets, weight=self.class_weights)

class BalancedSoftmaxLoss(nn.Module):
    """Implemented based the paper Balanced Meta-Softmax for Long-Tailed Visual Recognition
    """
    def __init__(self, samples_per_class):
        super().__init__()
        self.log_class_ratio = torch.log(samples_per_class + 1e-8)

    def get_proba(self, logits):
        adjusted_logits = logits + self.log_class_ratio.to(logits.device)
        return F.softmax(adjusted_logits)

    def forward(self, logits, targets):
        # log_freq = torch.log(compute_class_frequence(targets)) => need a better sampler
        adjusted_logits = logits + self.log_class_ratio.to(logits.device)
        return F.cross_entropy(adjusted_logits, targets)

class WeightedSoftBinnedMSELoss(nn.Module):
    def __init__(self, num_bins=50, target_range=(-2, 2), sigma=0.1):
        """
        Add weights to MSE using soft bins for regression
        
        Parameters:
        - num_bins (int): Number of bins to discretize the target values.
        - target_range (tuple): Range of the target values (min, max).
        - sigma (float): Standard deviation for the Gaussian kernels.
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        
        # Define the bin centers according to the target range
        self.min_target, self.max_target = target_range
        self.bin_centers = torch.linspace(self.min_target, self.max_target, num_bins).view(1, -1)

    def forward(self, predictions, targets):
        # Ensure bin_centers are on the same device as targets
        self.bin_centers = self.bin_centers.to(targets.device)

        # Step 1: Soft binning using Gaussian kernels (batch_size x C_bin)
        soft_bins = torch.exp(-((targets.unsqueeze(1) - self.bin_centers) ** 2) / (2 * self.sigma ** 2))
        # soft_bins = soft_bins / (soft_bins.sum(dim=1, keepdim=True) + 1e-8) # ~probability norm
        soft_bins = F.softmax(soft_bins)

        # Step 2: Compute class weights based on the soft bin assignments
        class_counts = soft_bins.sum(dim=0) + 1e-8  
        class_weights = soft_bins.size(0) / (self.num_bins * class_counts)
    
        # Step 3: Compute sample weights for each target based on their soft bin membership
        sample_weights = torch.sum(soft_bins * class_weights, dim=1)

        # Step 4: Compute weighted MSE loss
        mse_loss = (predictions - targets) ** 2
        loss = mse_loss * sample_weights

        return loss.mean()

class WeightedSoftBinnedCELoss(nn.Module):
    def __init__(self, num_bins=30, target_range=(-2, 2), sigma=0.1, temperature=1.):
        """
        Convert regression to classification w/ CE loss, seem like it focus more on left tail when #bins is large (<20 seems ok)
        
        Parameters:
        - num_bins (int): Number of bins to discretize the target values.
        - target_range (tuple): Range of the target values (min, max).
        - sigma (float): Standard deviation for the Gaussian kernels.
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.temperature = temperature
        
        # Define the bin centers according to the target range
        self.min_target, self.max_target = target_range
        self.bin_centers = torch.linspace(self.min_target, self.max_target, num_bins).view(1, -1)

    def forward(self, predictions, targets):
        # Ensure bin_centers are on the same device as targets
        self.bin_centers = self.bin_centers.to(targets.device)

        # Step 1: Soft binning using Gaussian kernels
        target_soft_bins = torch.exp(-((targets.unsqueeze(1) - self.bin_centers) ** 2) / (2 * self.sigma ** 2))
        target_probs = F.softmax(target_soft_bins)

        # Step 2: Compute class weights based on the soft bin assignments
        class_counts = target_probs.sum(dim=0) + 1e-8  
        class_weights = target_probs.size(0) / (self.num_bins * class_counts)
        # Step 4: Compute weighted CE loss
        pred_soft_bins = torch.exp(-((predictions.unsqueeze(1) - self.bin_centers) ** 2) / (2 * self.sigma ** 2))
        loss = -target_probs * F.log_softmax(pred_soft_bins) * class_weights

        return loss.mean()
    
class WeightedSoftBinnedBCELoss(nn.Module):
    def __init__(self, num_bins=50, target_range=(-2, 2), sigma=0.1, temperature=1.):
        """
        Differentiable Balanced Softmax Loss for regression
        
        Parameters:
        - num_bins (int): Number of bins to discretize the target values.
        - target_range (tuple): Range of the target values (min, max).
        - sigma (float): Standard deviation for the Gaussian kernels.
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.temperature = temperature
        
        # Define the bin centers according to the target range
        self.min_target, self.max_target = target_range
        self.bin_centers = torch.linspace(self.min_target, self.max_target, num_bins).view(1, -1)

    def forward(self, predictions, targets):
        # Ensure bin_centers are on the same device as targets
        self.bin_centers = self.bin_centers.to(targets.device)

        # Step 1: Soft binning using Gaussian kernels
        target_soft_bins = torch.exp(-((targets.unsqueeze(1) - self.bin_centers) ** 2) / (2 * self.sigma ** 2))
        target_probs = F.softmax(target_soft_bins)

        # Step 2: Compute log of classes' #samples
        log_class_ratio = torch.log(target_probs.sum(dim=0) + 1e-8)
    
        # Step 4: Compute weighted CE loss
        pred_soft_bins = torch.exp(-((predictions.unsqueeze(1) - self.bin_centers) ** 2) / (2 * self.sigma ** 2))
        adjusted_pred_binss = pred_soft_bins + log_class_ratio
        pred_probs = F.softmax(adjusted_pred_binss)

        loss = -target_probs * F.log_softmax(pred_probs)

        return loss.mean()

# Regression as Classification: Influence of Task Formulation on Neural Network Features

class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights, threshold=0.8):
        # similar to WeightedSoftmaxLoss but for regression
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.high_weight = class_weights[1]
        self.low_weight = class_weights[0]

    def forward(self, predictions, targets):
        weights = torch.where(targets >= self.threshold, self.high_weight, self.low_weight)
        loss = weights * (predictions - targets) ** 2
        return loss.mean()

# https://datascience.stackexchange.com/questions/114455/is-pearson-correlation-a-good-loss-function
class ProductLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return -torch.mean(y_true * y_pred)

def confidence_penalty(p, y, epsilon=0.1):
    """
    Compute the penalty for overconfident predictions.
    
    Parameters:
        p (torch.Tensor): Predicted probabilities (values between 0 and 1).
        y (torch.Tensor): Ground truth labels (0 or 1).
        lam (float): Weight for the penalty term.
        epsilon (float): Confidence threshold for applying the penalty.
        
    Returns:
        torch.Tensor: The computed penalty for each prediction.
    """
    # To avoid numerical issues (like taking square roots or logs of 0), clamp p slightly
    p = torch.clamp(p, min=1e-7, max=1-1e-7)
    
    # For positive samples (y==1), penalize if p is too close to 1:
    pos_penalty = torch.clamp(p - (1 - epsilon), min=0) ** 2
    
    # For negative samples (y==0), penalize if p is too close to 0:
    neg_penalty = torch.clamp(epsilon - p, min=0) ** 2
    
    # Use y to select the appropriate penalty for each sample:
    total_penalty = y * pos_penalty + (1 - y) * neg_penalty
    
    return total_penalty

def get_contrastive_loss(repr, temperature=0.1):
    B2, D = repr.shape
    B = B2 // 2
    G, PG = torch.chunk(repr.view(B, 2, D), 2, dim=1)
    G, PG = G.squeeze(), PG.squeeze()
    NG = PG[torch.randperm(B)]
    return InfoNCE(temperature)(G, PG, NG)

# def compute_class_weights(labels):
#     class_counts = torch.bincount(labels)
#     return len(labels) / (len(class_counts) * class_counts)

def compute_class_frequence(labels):
    cls_counts  = torch.bincount(labels)
    return cls_counts 

def coeff_loss(scores, ranks):
    # try to maximize the coefficient
    return -torch.corrcoef(torch.stack([scores, ranks]))[1, 0]

def reduce(loss, reduction):
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

def ranking_loss(scores, true_scores, reduction='mean'):
    """
    Computes the pairwise ranking loss where lower scores correspond to lower ranks.

    Args:
        scores (torch.Tensor): Tensor of predicted scores, shape (n,).
        ranks (torch.Tensor): Tensor of ground truth ranks, shape (n,).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Ensure inputs are 1D tensors
    scores = scores.view(-1)
    true_scores = true_scores.float().view(-1) 
    
    rank_probs = torch.softmax(true_scores, dim=0)
    score_probs = torch.softmax(scores, dim=0) 
    
    loss = rank_probs * torch.log(score_probs + 1e-8)  
    loss = reduce(-loss, reduction)
    return loss

def xai_loss(g, h_reprs, global_reprs, labels, predictor, loss_fn, temperature=1.):
    with g.local_scope():
        g.ndata['h'] = h_reprs
        h_g_sum = dgl.sum_nodes(g, "h") / 2
        probs = predictor(h_g_sum)
        loss1 = loss_fn(probs, labels)
        rand_perm = torch.randperm(global_reprs.shape[0])
        negative_reprs = h_g_sum[rand_perm]
        loss2 = InfoNCE(temperature)(global_reprs, h_g_sum, negative_reprs)
    return loss1, -loss2
