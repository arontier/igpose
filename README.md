Official source code for IgPose

This project is under CC-BY-NC-ND-4.0 license.

## Initial environment
- Create a conda environment ```conda create -y -n $1 python=3.12```
- Execute the ```scripts/setup.sh``` file to install required packages

## Inference 
1. Prepare inference configuration file

- You need to create an yaml file similar to "template/inference_configuration_template.yaml".
- CDR information is needed for pooling layer so please please CDR information similar to "template/cdr_infor_template.csv". If the cdr_information has more than 1 CDR field for either heavy or light chain, please merge them to a single column for each. Since current dataset loader is used for both training & test, you can set values of "DockQ" & "label" columns as 0. Both csv and tsv are accepted.


2. Inference code

Uncompress model checkpoints
```
$ cd checkpoints
$ tar -xzf deploy_models.tar.gz
```

Inference code execution
```
$ CUDA_VISIBLE_DEVICES='0' python src/predict.py /path/to/inference_configuration_template.yaml
```

Please check configs folder for configuration templates

By default, graphs & embeddings are generated. You can skip this step if graphs are generated but prediction part occurs errors with ```--skip-data-prepare```

3. Training code

Training instruction will be released soon. 