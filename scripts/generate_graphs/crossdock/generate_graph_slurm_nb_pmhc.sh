#!/bin/bash
prefix=/People/alexbui/workspace/proteins/ab_affinity
output_prefix=/Arontier_1/Projects/AbAg_decoy/dataset

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/nb_pmhc_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/nb-pmhc/rank0_pdbqts \
    -f2 $output_prefix/generated_graphs/nb_pmhc_fullgraph/all_files.pt \
    -m uni_seq -hn "A" -hi 9999 -agn "B,C,D" -ft ".pdbqt" -ag

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/nb_pmhc_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/nb-pmhc/rank0_pdbqts \
    -f2 $output_prefix/generated_graphs/nb_pmhc_fullgraph/all_files.pt \
    -m embed_uni_ab -hn "A" -hi 9999 -agn "B,C,D" -ft ".pdbqt" -ag

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/nb_pmhc_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/nb-pmhc/rank0_pdbqts \
    -f2 $output_prefix/generated_graphs/nb_pmhc_fullgraph/all_files.pt \
    -m embed_uni_ag -hn "A" -hi 9999 -agn "B,C,D" -ft ".pdbqt" -ag

python $prefix/src/make_graph_rcsb.py -o $output_prefix/generated_graphs/nb_pmhc_fullgraph \
    -f $output_prefix/structures_chai1-single/cross/nb-pmhc/rank0_pdbqts \
    -m graph -hn "A" -hi 9999 -agn "B,C,D" -ft ".pdbqt" -ag