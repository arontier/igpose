from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dgl.nn.pytorch import SumPooling, WeightAndSum, AvgPooling, MaxPooling
from layers.pooling import AttentionPooling, AttentiveFPPooling, WeightAndMean, WeightAndMeanV2, \
                            WeightedSumInterface, WeightedMeanInterface, \
                            WeightedSumCDR, WeightedMeanCDR, WeightedSumCDRInterface, \
                            WeightedSumInterfaceCR, WeightedSumCDRCR, WeightedSumCDRInterfaceCR, \
                            WeightedSumAb, WeightedSumAg, WeightedComponent, WeightedComponentCDR, \
                            WeightAndSumParamatization, SetPooling, SetDecoderPooling, MDNCDRPooling, MDNCDRPoolingAggressive

from layers.conv import ConvBlock, GATConvBlock, GINEConvBlock, HGTConvBlock, EGNNConvBlock, EGNNNormConvBlock, EGNNFroConvBlock
from layers.conv_pyg import GATConvBlockPyG, EGNNConvBlockPyG, FastEGNNConvBlockPyG
from common.losses import FocalLoss, BalancedSoftmaxLoss, WeightedSoftmaxLoss, WeightedMSELoss, WeightedSoftBinnedMSELoss, WeightedSoftBinnedCELoss, WeightedSoftBinnedBCELoss, ProductLoss

CONV_MAP = {'gat': GATConvBlock, 'gatv2': GATConvBlock, 'gin': GINEConvBlock, 'gt': HGTConvBlock, 'egnn': EGNNConvBlock, 'egnn_norm': EGNNNormConvBlock, 'egnn_fro': EGNNFroConvBlock}
CONV_MAP_PYG = {'gat': GATConvBlockPyG, 'egnn': EGNNConvBlockPyG, 'fastegnn': FastEGNNConvBlockPyG}
ACTIVATION_MAP = {'relu': F.relu, 'elu': F.elu, 'silu': F.silu, 'lrelu': F.leaky_relu}
ACTIVATION_MAP_NN = {'relu': nn.ReLU, 'elu': nn.ELU, 'silu': nn.SiLU, 'lrelu': nn.LeakyReLU}
POOLING_MAP = {'sum': SumPooling, 'mean': AvgPooling, 'max': MaxPooling,
               'weighted': WeightAndSum, 'weightedmean': WeightAndMean,
               'weightedmeanv2': WeightAndMeanV2,
               'weightedinterface': WeightedSumInterface,
               'weightedmeaninterface': WeightedMeanInterface,
               'weightedcdr': WeightedSumCDR,
               'weightedmeancdr': WeightedMeanCDR,
               'weightedcdrinterface': WeightedSumCDRInterface,
               'weightedcdrcr': WeightedSumCDRCR,
               'weightedinterfacecr': WeightedSumInterfaceCR,
               'weightedcdrinterfacecr': WeightedSumCDRInterfaceCR,
               'weightedcomponent': WeightedComponent,
               'weightedcomponentcdr': WeightedComponentCDR,
               'weightedab': WeightedSumAb,
               'weightedag': WeightedSumAg,
               'attention': AttentionPooling,
               'afp': AttentiveFPPooling,
               'weightedparam': WeightAndSumParamatization,
               'set': SetPooling,
               'setdecoder': SetDecoderPooling,
               'mdncdr': MDNCDRPooling,
               'mdncdras': MDNCDRPoolingAggressive}

LOSS_FUNCTION_MAP = {'mse': nn.MSELoss,
                     'ce': nn.CrossEntropyLoss,
                     'focal': FocalLoss,
                     'bs': BalancedSoftmaxLoss,
                     'ws': WeightedSoftmaxLoss,
                     'wmse': WeightedMSELoss,
                     'wsbm': WeightedSoftBinnedMSELoss,
                     'wsbc': WeightedSoftBinnedCELoss,
                     'wsbb': WeightedSoftBinnedBCELoss,
                     'nll': nn.NLLLoss,
                     'prod': ProductLoss
                    }

EMBEDDING_FOLDER_MAP = {
        'ab_fullgraph': 'ab_fullgraph',
        'nb_fullgraph': 'nb_fullgraph',
        'tcr_pmhc_fullgraph': 'tcr_pmhc_fullgraph',
        'nb_pmhc_fullgraph': 'nb_pmhc_fullgraph',
        'tcr_agnb_fullgraph': 'tcr_agnb_fullgraph',
        'ab_chai1_fullgraph': 'ab_chai1_fullgraph',
        'nb_chai1_fullgraph': 'nb_chai1_fullgraph',
        'tcr_pmhc_chai1_fullgraph': 'tcr_pmhc_chai1_fullgraph',
        'casp16_fullgraph': 'casp16_fullgraph',
    }



def is_regression(name: str) -> bool:
    if name in {'mse', 'dbs', 'wsbm', 'wsbc', 'wsbb', 'prod'}:
        return True
    return False

def get_convolution(name: str, backend: str=None) -> ConvBlock:
    print(name)
    if backend == 'pyg':
        return CONV_MAP_PYG[name]
    return CONV_MAP[name]

def get_activator(name: str) -> Callable:
    return ACTIVATION_MAP[name]

def get_activation_layer(name: str) -> nn.Module:
    return ACTIVATION_MAP_NN[name]

def get_pooling_operator(name: str = None) -> nn.Module:
    if not name:
        name = 'sum'
    return POOLING_MAP[name]

def get_loss_function(name: str) -> nn.Module:
    return LOSS_FUNCTION_MAP[name]