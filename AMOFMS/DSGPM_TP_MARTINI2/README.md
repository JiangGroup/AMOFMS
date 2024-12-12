# DSGPM-TP: Deep Supervised Graph Partitioning Model with Type Prediction

## Introduction

DSGPM-TP is a deep learning-based model designed to automate the process of coarse-grained (CG) mapping from fine-grained molecular structures. By utilizing graph-based neural networks, the model partitions atomic structures into CG representations and predicts the types of CG particles. This tool reduces manual effort and enhances reproducibility and accuracy in CG mapping. More details can be accessed [here](https://doi.org/10.48550/arXiv.2408.06609).

## Features

- **Automated CG mapping**: Simplifies the partitioning of fine-grained molecular structures into CG models.
- **Type prediction**: Predicts the types of coarse-grained particles from molecular graphs.
- **Flexible customization**: DSGPM-TP can be trained on new datasets, making it adaptable to different CG mapping conventions.

## Requirements

Make sure the following dependencies are installed:

- matplotlib==3.4.3
- networkx==3.2
- numpy==1.22.3
- pexpect==4.9.0
- Pillow==10.2.0
- pyswarms==1.3.0
- rdkit==2023.9.4
- Requests==2.31.0
- scikit_learn==1.3.2
- scipy==1.8.1
- seaborn==0.13.2
- scikit-image
- torch==2.1.2
- torch_geometric==2.4.0
- tqdm==4.66.1
- tensorboard

You can creat a new conda environment and install the required packages using:

```bash
conda create -n dsgpm_tp python=3.9
conda activate dsgpm_tp
pip install -r requirements.txt
```

## Model Training

To train the DSGPM-TP model on your dataset, follow these steps:

###  1. Prepare Your Dataset

Each molecule item (`.json`) should include node features, edges, and the target CG labels.
The `.json` example can be found in our prebuilt dataset of MARTINI2 `MARTINI2_Dataset`.

###  2. Train Your Model

You can use the following Shell script to train the DSGPM-TP model.
```
cgloss=0.01
mkdir ./ckpt/${cgloss}
mkdir ./tb_log/${cgloss}

CUDA_VISIBLE_DEVICES=0 python ./train.py --data_root ../MARTINI_Dataset --epoch 500 --batch_size 32 --ckpt ./ckpt/${cgloss}  --num_workers 4 --tb_root  ./tb_log/${cgloss} --tb_log --title cgloss_${cgloss} --cg_type_loss_parameter ${cgloss} --no_charge_feat --no_aromatic_feat 
```

## Prediction
Once the model is trained, you can make predictions on new molecular structures using the `CGPredictionFromDSGPM_TP.py` script .
```
python CGPredictionFromDSGPM_TP.py --smiles 'CCCCCCCCCCCC' --json_output './test_mapping' --num_bead 3

# The prediction result (`.json` and `.png`) will be output into the folder './test_mapping/CG'. 
```
The `CGPredictionFromDSGPM_TP.py` will use our trained model parameters ('model/best_epoch.pth') for MARTINI2 CG Mapping. You can replace this model with your trained model.


## Contact
For any questions, bug reports, or contributions, contact us at `zhixuan@iccas.ac.cn`.