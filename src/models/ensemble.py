import sys
sys.path.append('..')
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Any, List, Optional
from common.utils import get_regression_pred
from common.mapping_utils import CONV_MAP
from models.graph_models import GraphModel

def _strip_module_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    """Remove DistributedDataParallel/DataParallel 'module.' prefixes if present."""
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

def _resolve_conv_fn(cfg: Dict[str, Any]):
    """
    Find the conv block class name in the config under several common keys
    and map it to the actual class via conv_fn_map.
    """
    for key in ("conv_fn", "conv", "conv_name", "block", "conv_block"):
        if key in cfg and cfg[key] is not None:
            name = str(cfg[key])
            if name in CONV_MAP:
                return CONV_MAP[name]
            raise KeyError(f"Unknown conv block '{name}'. "
                           f"Provide it in conv_fn_map (got keys: {list(CONV_MAP)})")
    raise KeyError("No conv block name found in config (tried: conv_fn/conv/conv_name/block/conv_block).")

def _sanitize_model_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare args for GraphModel(*kwargs):
    - keep required args if present: embed_size, hidden_size, out_channels, num_layers
    - include optional: dropout, use_edge_feat, activation, and all others as **kargs
    """
    embed_size = args['embed_size']
    if args['node_onehot']:
        embed_size += 3
    if args['di_angle']:
        embed_size += 4
    if args['cdr_onehot']:
        embed_size += 1
    conv_fn = _resolve_conv_fn(args)
    return [conv_fn, embed_size, args['hidden_size'], args['output_size'], args['num_layers'], args['dropout']]
    
def _sanitize_model_kwargs(cfg: Dict[str, Any], also_drop=None) -> Dict[str, Any]:
    """
    Keeps only:
      - KEYWORD_ONLY params
      - names unknown to `fn` (these are fine if `fn` has **kwargs)
    """
    cfg = dict(cfg)
    for k in (also_drop or ()):
        cfg.pop(k, None)

    sig = inspect.signature(GraphModel)
    positional_names = {
        name for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    keyword_only_names = {
        name for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())

    # Keep only keyword-only + unknowns (unknowns allowed only if **kwargs exists)
    kept = {k: v for k, v in cfg.items() if k in keyword_only_names}
    unknowns = {k: v for k, v in cfg.items()
                if k not in positional_names and k not in keyword_only_names}
    if has_var_kw:
        kept.update(unknowns)   # pass through to **kwargs
    edge_dim = cfg['edge_size'] if not cfg['edge_onehot'] else cfg['edge_size'] + 3
    model_kargs = {
        'use_edge_feat': cfg['use_ef'],
        'edge_dim': edge_dim,
        'edge_fn': cfg['edge_fn'], 
        'aggregation_fn': cfg['agg_fn'],
        'pooling': cfg['pooling'],
        'activation': cfg['activation'],
        'mdn': False,
        'node_pred': False,
        'aggregation_mode': cfg['agg_mode'],
        'mdn_cls': cfg['mdn_cls']
    }
    kept.update(model_kargs)
    return kept

# --- The Ensemble Wrapper ----------------------------------------------------

class EnsembleWrapper(nn.Module):
    """
    Loads the merged ensemble checkpoint produced by build_ensemble_checkpoint(...)
    and performs average-of-softmax probabilities at inference.

    The merged checkpoint must contain:
      - state_dicts: List[Dict[str, Tensor]]
      - configs:     List[Dict[str, Any]]  (aligns with state_dicts)
      - members:     List[Dict[str, str]]  (metadata; optional usage)
      - pairing_rule: str                  (for provenance)
    """
    def __init__(
        self,
        merged_ckpt_path: str,
        device: str | torch.device = "cpu",
        dtype: Optional[torch.dtype] = None,  # e.g., torch.float16 for fp16
        mode:str = "classification"
    ):
        super().__init__()
        data = torch.load(merged_ckpt_path, map_location="cpu")
        state_dicts: List[Dict[str, torch.Tensor]] = data["state_dicts"]
        configs:     List[Dict[str, Any]]          = data["configs"]
        self.members: List[Dict[str, str]]         = data.get("members", [])
        self.pairing_rule: str                     = data.get("pairing_rule", "unknown")

        assert len(state_dicts) == len(configs), (
            f"Mismatch: {len(state_dicts)} state_dicts vs {len(configs)} configs"
        )
        self.configs = configs
        self.models = nn.ModuleList()
        self.model_id = -1
        self.device = torch.device(device)
        self.dtype  = dtype
        self.mode = mode
        
        # Rebuild each GraphModel from its config and load weights
        for idx, (sd, cfg) in enumerate(zip(state_dicts, configs)):
            args  = _sanitize_model_args(cfg)
            kwargs  = _sanitize_model_kwargs(cfg, {'input_size'})
            model = GraphModel(*args, **kwargs)
            sd = _strip_module_prefix(sd, "module.")
            model.load_state_dict(sd, strict=False)

            if dtype is not None:
                model = model.to(dtype=dtype)
            model = model.to(self.device)
            model.eval()
            self.models.append(model)

        # Optional: basic shape sanity check for logits across members
        with torch.no_grad():
            self._out_channels: Optional[int] = None

    @torch.no_grad()
    def forward(self, g, x, edge_attr=None):
        """
        Run all members and return the averaged prediction.

        average_mode:
          - "prob": average softmax probabilities (default; recommended)
          - "logit": average logits first, then softmax once (sometimes slightly different)
        """
        probs_list  = []
        if self.model_id != -1:
            models, configs = [self.models[self.model_id]], [self.configs[self.model_id]]
        else:
            models, configs = self.models, self.configs
        for m, c in zip(models, configs):
            with g.local_scope():
                out = m(g, x, edge_attr)
                logits = out["logits"]
                if self.mode == 'classification':
                    probs_list.append(F.softmax(logits, dim=-1))
                else:
                    probs_list.append(get_regression_pred(logits, activation=c['pred_fn'], label_norm=c['label_norm']))
        probs = torch.stack(probs_list, dim=0).mean(0)
        return probs

    @torch.no_grad()
    def predict(self, g, x, edge_attr=None):
        """
        Returns:
          - if topk == 1: LongTensor [B]
          - else: (indices [B, topk], values [B, topk])
        """
        return self.forward(g, x, edge_attr)

    def to(self, *args, **kwargs):  # convenience: ensure submodules follow device/dtype
        super().to(*args, **kwargs)
        self.device = kwargs.get("device", args[0] if args else self.device)
        self.dtype  = kwargs.get("dtype", self.dtype)
        return self