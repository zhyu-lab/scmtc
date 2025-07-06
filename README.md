# scmtc
a hybrid Mamba-Transformer-CNN framework for subclonal inference from scDNA-seq data

## Requirements

* Python 3.9+.

# Installation

## Clone repository

First, download scTCA from github and change to the directory:

```bash
git clone https://github.com/zhyu-lab/scmtc
cd scmtc
```

## Create conda environment (optional)

Create a new environment named "scmtc":

```bash
conda create --name scmtc python=3.9
```

Then activate it:

```bash
conda activate scmtc
```

## Install requirements
Install PyTorch first according to your CUDA version, then install required packages as follows:
```bash
python -m pip install -r requirements.txt
```

# Usage

## Get cell embeddings

The train.py is used to train the model and get cell embeddings and predicted cluster labels.

Example:

```bash
tar -zxvf data/A_500k.tar.gz
python train.py --input A_500k.txt --output data --resolution 0.1
```

## Call CNAs
The functions for calling CNAs are implemented in MATLAB.

Example:

```bash
callcnas('../data/A_500k.txt','../data/embeddings.txt','../data',10,10)
```

# Contact

If you have any questions, please contact 12024130911@stu.nxu.edu.cn.