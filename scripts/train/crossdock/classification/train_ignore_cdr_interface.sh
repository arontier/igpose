#!/bin/bash

python $prefix/src/main.py --desc "[WeightedCDRInterface Ignore] 8515 egnn norm mdn 1e-3 & node & prob 2e-3 r73 NT 500:500" \
    --datapath $data_prefix \
    --embed_path $data_prefix/all_embeds \
    --filepath $prefix/data/metadata/ab_nondock_metadata_cluster_v6_3_native_cdr_v2.csv \
    --ckptpath $output_prefix/ckpt \
    --log_dir $output_prefix/logs \
    --batch_size 512 --lr 0.0001 --use_ef \
    --edge_size 30 --num_layers 4 --dropout 0.1 --conv_fn egnn_norm --hidden_size 64 --pooling weightedcdrinterface \
    --activation silu --num_epochs 50 --pred_loss ce --output_size 2 --embed_size 320 --agg_mode double \
    --mdn_factor 0.001 --node_factor 0.001 --coeff_factor 0 --prob_factor 0.001  \
    --cuda 0 --pred_fn f1 --sample_type inverse --edge_onehot --backend dgl --sep_embed --grad_scale --class_ratio '0.85,0.15' \
    --khop 3 --node_threshold 500 --rc_node_threshold 500 --cdr
