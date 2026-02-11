#!/bin/bash

# conda create -y -n $1 python=3.12
# conda activate $1

# install pytorch
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install dgl
conda install -y -c dglteam/label/th24_cu121 dgl

# install pyg
pip install torch_geometric 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# install other package
pip install scikit-learn
pip install matplotlib
pip install tensorboard
conda install -y mdanalysis
pip install esm 


# install apex, make sure cuda is available, cuda & pytorch cuda version are matched
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

pip install higher

# install autodocktool for MolKit
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3