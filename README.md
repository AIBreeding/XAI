## Tree-Based Interpretable Machine Learning Models for GxE Prediction

This is a Python implementation of our paper:

## Interpretable machine learning uncovers complex, interacting traits associated with maize yield across diverse environments

## Getting started

### Requirements

 - Python 3.8.4
 - scikit-learn 1.3.0
 - conda

### Installation
Clone repository: 

```bash
git clone https://github.com/AIBreeding/GxE.git
```
Create environment:
```bash
conda create -n GxE python=3.8.4
conda activate GxE
```
Install packages:
```bash
cd GxE
conda install --yes --file requirements.txt
```

Follow the instructions in [data directory](data/README.md) to get dataset files.

Follow the instructions in [model/G2Pmodel](model/G2Pmodel/README.md) to get pre-trained models.

## Usage

### Hyperparameter optimization
- python Start-AutoHPO.py

### 10-fold cross-validation data partitioning
- python kfolds.py

### Model training and independent prediction
- python Start-Basic_model.py

### Basic model stacking operation
- python stacking.py

### Model Interpretation and Visualization
- Please execute the XAI.ipynb script

