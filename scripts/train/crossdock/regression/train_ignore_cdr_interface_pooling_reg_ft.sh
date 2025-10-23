#!/bin/bash

prefix=/People/alexbui/workspace/proteins/ab_affinity
data_prefix=/Arontier_1/Projects/AbAg_decoy/dataset/generated_graphs
output_prefix=/Arontier_1/Privates/alexbui/projects/ab_affinity

TIMESTAMPS=(1752158575)
CHECKPOINTS=(10)

IDX=0
DATASETS="ab_chai1_fullgraph,nb_chai1_fullgraph,tcr_pmhc_chai1_fullgraph,ab_boltz2_fullgraph,nb_boltz2_fullgraph,tcr_pmhc_boltz2_fullgraph"
TRAIN_SUBSET="ab_fullgraph,nb_fullgraph,tcr_pmhc_fullgraph,"

echo "Running $output_prefix/ckpt/best_model_${TIMESTAMPS[IDX]}/best_${CHECKPOINTS[IDX]}.pth"

python $prefix/src/main.py --desc "[WeightedCDRInterface Regression] egnn norm mdn 1e-3 rank coeff & NT 500:500" \
    --datapath $data_prefix \
    --embed_path $data_prefix/all_embeds \
    --filepath $prefix/data/metadata/ab_nondock_boltz2_metadata_cluster_v6_5_native_cdr_7decoy.csv \
    --ckptpath $output_prefix/ckpt \
    --log_dir $output_prefix/logs \
    --batch_size 32 --lr 0.00001 --use_ef \
    --edge_size 30 --num_layers 4 --dropout 0.1 --conv_fn egnn_norm --hidden_size 64 --pooling weightedcdrinterface \
    --activation silu --num_epochs 200 --pred_loss mse --output_size 1 --embed_size 320 --agg_mode double \
    --mdn_factor 0.001 --node_factor 0 --coeff_factor 0 --prob_factor 0 --cls_factor 0 --rank_factor 1 --coeff_factor 1 \
    --cuda 0 --pred_fn tanh --sample_type inverse --edge_onehot --backend dgl --sep_embed --grad_scale --class_ratio '' \
    --khop 3 --node_threshold 500 --rc_node_threshold 500 --cdr --label_norm 'none' \
    --train_dataset "${TRAIN_SUBSET}${DATASETS}" \
    --val_dataset "${TRAIN_SUBSET}${DATASETS}" \
    --test_dataset "${TRAIN_SUBSET}${DATASETS}" \
    --pretrained_ckpt $output_prefix/ckpt/best_model_${TIMESTAMPS[IDX]}/best_${CHECKPOINTS[IDX]}.pth 
