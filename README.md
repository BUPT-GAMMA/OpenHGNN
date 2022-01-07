# OpenHGNN

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BUPT-GAMMA/OpenHGNN)
[![Documentation Status](https://readthedocs.org/projects/openhgnn/badge/?version=latest)](https://openhgnn.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/BUPT-GAMMA/OpenHGNN)

[**启智社区（中文版）**](https://git.openi.org.cn/GAMMALab/OpenHGNN)

This is an open-source toolkit for Heterogeneous Graph Neural Network(OpenHGNN) based on [DGL [Deep Graph Library]](https://github.com/dmlc/dgl) and [PyTorch](https://pytorch.org/). We integrate SOTA models of heterogeneous graph.

## News

We release the latest version v0.1.1 on OpenI with Chinese. 

启智社区用户可以享受到如下功能：

- 全新的中文文档
- 免费的计算资源
- OpenHGNN最新功能

## Key Features

- Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and dataset. Besides, we also integrate [optuna](https://optuna.org/) to get hyperparameter optimization.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
- Efficiency: The backend dgl provides efficient APIs.

## Get Started

#### Requirements and Installation

- Python  >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/)  >= 1.7.1
- [DGL](https://github.com/dmlc/dgl) >= 0.7.0

- CPU or NVIDIA GPU, Linux, Python3

**1. Python environment (Optional):** We recommend using Conda package manager

```bash
conda create -n openhgnn python=3.7
source activate openhgnn
```

**2. Pytorch:** Install [PyTorch](https://pytorch.org/). For example:

```bash
# CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. DGL:** Install [DGL](https://www.dgl.ai/pages/start.html), follow their instructions. For example:

```bash
# CUDA versions: cpu, cu101, cu102, cu110, cu111
pip install --pre dgl-cu101 -f https://data.dgl.ai/wheels-test/repo.html
```

**4. OpenHGNN and other dependencies:**

```bash
git clone https://github.com/BUPT-GAMMA/OpenHGNN
# If you encounter a network error, try git clone from openi as following.
# git clone https://git.openi.org.cn/GAMMALab/OpenHGNN.git
cd OpenHGNN
pip install -r requirements.txt
```

#### Running an existing baseline model on an existing benchmark [dataset](./openhgnn/dataset/#Dataset)

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config --load_from_pretrained
```

usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config]

*optional arguments*:

``-h, --help``	show this help message and exit

``--model -m ``	name of models

``--task -t``	name of task

``--dataset -d``	name of datasets

``--gpu -g``	controls which gpu you will use. If you do not have gpu, set -g -1.

``--use_best_config``	use_best_config means you can use the best config in the dataset with the model. If you want to set the different hyper-parameter, modify the [openhgnn.config.ini](./openhgnn/config.ini) manually. The best_config will override the parameter in config.ini.

``--use_hpo`` Besides use_best_config, we give a hyper-parameter [example](./openhgnn/auto) to search the best hyper-parameter automatically.

``--load_from_pretrained`` will load the model from a default checkpoint.

e.g.: 

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config
```

**Note**: If you are interested in some model, you can refer to the below models list.

Refer to the [docs](https://openhgnn.readthedocs.io/en/latest/index.html) to get more basic and depth usage.

## [Models](./openhgnn/models/#Model)

### Supported Models with specific task

The link will give some basic usage.

| Model                                                    | Node classification | Link prediction    | Recommendation     |
| -------------------------------------------------------- | ------------------- | ------------------ | ------------------ |
| [Metapath2vec](./openhgnn/output/metapath2vec)[KDD 2017] | :heavy_check_mark:  |                    |                    |
| [RGCN](./openhgnn/output/RGCN)[ESWC 2018]                | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [HERec](./openhgnn/output/HERec)[TKDE 2018]              | :heavy_check_mark:  |                    |                    |
| [HAN](./openhgnn/output/HAN)[WWW 2019]                   | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [KGCN](./openhgnn/output/KGCN)[WWW 2019]                 |                     |                    | :heavy_check_mark: |
| [HetGNN](./openhgnn/output/HetGNN)[KDD 2019]             | :heavy_check_mark:  | :heavy_check_mark: |                    |
| HGAT[EMNLP 2019]                                         |                     |                    |                    |
| [GTN](./openhgnn/output/GTN)[NeurIPS 2019] & fastGTN     | :heavy_check_mark:  |                    |                    |
| [RSHN](./openhgnn/output/RSHN)[ICDM 2019]                | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [DMGI](./openhgnn/output/DMGI)[AAAI 2020]                | :heavy_check_mark:  |                    |                    |
| [MAGNN](./openhgnn/output/MAGNN)[WWW 2020]               | :heavy_check_mark:  |                    |                    |
| HGT[WWW 2020]                                            |                     |                    |                    |
| [CompGCN](./openhgnn/output/CompGCN)[ICLR 2020]          | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [NSHE](./openhgnn/output/NSHE)[IJCAI 2020]               | :heavy_check_mark:  |                    |                    |
| [NARS](./openhgnn/output/NARS)[arxiv]                    | :heavy_check_mark:  |                    |                    |
| [MHNF](./openhgnn/output/MHNF)[arxiv]                    | :heavy_check_mark:  |                    |                    |
| [HGSL](./openhgnn/output/HGSL)[AAAI 2021]                | :heavy_check_mark:  |                    |                    |
| [HGNN-AC](./openhgnn/output/HGNN_AC)[WWW 2021]           | :heavy_check_mark:  |                    |                    |
| [HeCo](./openhgnn/output/HeCo)[KDD 2021]                 | :heavy_check_mark:  |                    |                    |
| [SimpleHGN](./openhgnn/output/SimpleHGN)[KDD 2021]       | :heavy_check_mark:  |                    |                    |
| [HPN](./openhgnn/output/HPN)[TKDE 2021]                  | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [RHGNN](./openhgnn/output/RHGNN)[arxiv]                  | :heavy_check_mark:  |                    |                    |
| [HDE](./openhgnn/output/HDE)[ICDM 2021]                  |                     | :heavy_check_mark: |                    |

### Candidate models

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification[EMNLP 2019]
- [Heterogeneous Information Network Embedding with Adversarial Disentangler[TKDE 2021]](https://ieeexplore.ieee.org/document/9483653)

## Contributors

OpenHGNN Team[GAMMA LAB], DGL Team and Peng Cheng Laboratory.

See more in [CONTRIBUTING](./CONTRIBUTING.md).

