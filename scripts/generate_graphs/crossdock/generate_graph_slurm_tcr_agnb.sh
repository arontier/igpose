#!/bin/bash
prefix=/People/alexbui/workspace/proteins/ab_affinity
output_prefix=/Arontier_1/Projects/AbAg_decoy/dataset

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/tcr_agnb_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/tcr-ag_nb/rank0_pdbqts \
    -m seq -hn "B" -hi 9999 -ln "A" -agn "C" -ft ".pdbqt" -ag

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/tcr_agnb_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/tcr-ag_nb/rank0_pdbqts \
    -f2 $output_prefix/generated_graphs/tcr_agnb_fullgraph/all_files.pt \
    -m uni_seq  -hn "B" -hi 9999 -ln "A" -agn "C" -ft ".pdbqt" -ag

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/tcr_agnb_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/tcr-ag_nb/rank0_pdbqts \
    -f2 $output_prefix/generated_graphs/tcr_agnb_fullgraph/all_files.pt \
    -m embed_uni_ab  -hn "B" -hi 9999 -ln "A" -agn "C" -ft ".pdbqt" -ag

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/tcr_agnb_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/tcr-ag_nb/rank0_pdbqts \
    -f2 $output_prefix/generated_graphs/tcr_agnb_fullgraph/all_files.pt \
    -m embed_uni_ag  -hn "B" -hi 9999 -ln "A" -agn "C" -ft ".pdbqt" -ag

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/tcr_agnb_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/tcr-ag_nb/rank0_pdbqts \
    -m graph  -hn "B" -hi 9999 -ln "A" -agn "C" -ft ".pdbqt" -ag