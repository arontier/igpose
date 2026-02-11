# Official Source Code for IgPose

This project is licensed under the **CC-BY-NC-ND-4.0** license.

## Environment Setup
- **Create a conda environment:** `conda create -y -n <env_name> python=3.12`
- **Install dependencies:** Execute the setup script to install all required packages:  
  `bash scripts/setup.sh`

---

## Inference 

### 1. Prepare the Inference Configuration
- **Configuration File:** Create a YAML file based on `template/inference_configuration_template.yaml`.
- **CDR Information:** The pooling layer requires CDR data. Please provide this in a format similar to `template/cdr_info_template.csv`. **For prediction of just one pdb, you don't need this file.**
    - **Formatting:** If the CDR data contains more than one field for either the heavy or light chain, merge them into a single column for each.
    - **Placeholders:** Since the current data loader is used for both training and testing, you can set the values in the "DockQ" and "label" columns to `0`. 
    - **File Types:** Both `.csv` and `.tsv` formats are accepted.

### 2. Run Inference
**Uncompress model checkpoints:**
```bash
cd checkpoints
tar -xzf deploy_models.tar.gz

```

**Execute the inference script:**

```bash
CUDA_VISIBLE_DEVICES='0' python src/predict.py /path/to/your_inference_config.yaml

```

> **Note:** Please check the `configs` folder for additional configuration templates.

By default, graphs and embeddings are generated. If graphs have already been generated but an error occurs during the prediction phase, you can skip the data preparation step using the `--skip-data-prepare` flag.

---

### 3. Training

Training instructions will be released soon.