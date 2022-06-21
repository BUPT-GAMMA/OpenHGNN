# OpenHGNN

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BUPT-GAMMA/OpenHGNN)
[![Documentation Status](https://readthedocs.org/projects/openhgnn/badge/?version=latest)](https://openhgnn.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/BUPT-GAMMA/OpenHGNN)
[![visitors](https://visitor-badge.glitch.me/badge?page_id=BUPT-GAMMA.OpenHGNN)](https://github.com/BUPT-GAMMA/OpenHGNN)
[![Total lines](https://img.shields.io/tokei/lines/github/BUPT-GAMMA/OpenHGNN?color=red)](https://github.com/BUPT-GAMMA/OpenHGNN)

[**启智社区（中文版）**](https://git.openi.org.cn/GAMMALab/OpenHGNN) | [**Space4HGNN [SIGIR2022]**](../space4hgnn) | [**Benchmark&Leaderboard**](../openhgnn/dataset/ohgb.md) | [**Slack Channel**](https://app.slack.com/client/TDM5126J1/C03J6GND001)

This is an open-source toolkit for Heterogeneous Graph Neural Network based
on [DGL [Deep Graph Library]](https://github.com/dmlc/dgl) and [PyTorch](https://pytorch.org/). We integrate SOTA models
of heterogeneous graph.

## News

**2022-02-28**

We release the latest version v0.2.

- New Models
- [Space4HGNN [SIGIR2022]](../space4hgnn)
- [Benchmark&Leaderboard](../openhgnn/dataset/ohgb.md)

**2022-01-07**

启智社区用户可以享受到如下功能：

- 全新的中文文档
- 免费的计算资源
- OpenHGNN最新功能

## Key Features

- Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and dataset.
  Besides, we also integrate [optuna](https://optuna.org/) to get hyperparameter optimization.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
- Efficiency: The backend dgl provides efficient APIs.

## Get Started

#### Requirements and Installation

- Python  >= 3.6

- [PyTorch](https://pytorch.org/get-started/)  >= 1.9.0

- [DGL](https://github.com/dmlc/dgl) >= 0.8.0

- CPU or NVIDIA GPU, Linux, Python3

**1. Python environment (Optional):** We recommend using Conda package manager

```bash
conda create -n openhgnn python=3.7
source activate openhgnn
```

**2. Install Pytorch:** Follow their [tutorial](https://pytorch.org/get-started) to run the proper command according to
your OS and CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

**3. Install DGL:** Follow their [tutorial](https://www.dgl.ai/pages/start.html) to run the proper command according to
your OS and CUDA version. For example:

```bash
pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
```

**4. OpenHGNN and other dependencies:**

```bash
git clone https://github.com/BUPT-GAMMA/OpenHGNN
# If you encounter a network error, try git clone from openi as following.
# git clone https://git.openi.org.cn/GAMMALab/OpenHGNN.git
cd OpenHGNN
pip install -r requirements.txt
```

#### Running an existing baseline model on an existing benchmark [dataset](../openhgnn/dataset/#Dataset)

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config --load_from_pretrained
```

usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
[--gpu GPU] [--use_best_config]

*optional arguments*:

``-h, --help``    show this help message and exit

``--model -m ``    name of models

``--task -t``    name of task

``--dataset -d``    name of datasets

``--gpu -g``    controls which gpu you will use. If you do not have gpu, set -g -1.

``--use_best_config``    use_best_config means you can use the best config in the dataset with the model. If you want to
set the different hyper-parameter, modify the [openhgnn.config.ini](../openhgnn/config.ini) manually. The best_config
will override the parameter in config.ini.

``--use_hpo`` Besides use_best_config, we give a hyper-parameter [example](../openhgnn/auto) to search the best
hyper-parameter automatically.

``--load_from_pretrained`` will load the model from a default checkpoint.

e.g.:

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config
```

**Note**: If you are interested in some model, you can refer to the below models list.

Refer to the [docs](https://openhgnn.readthedocs.io/en/latest/index.html) to get more basic and depth usage.

## [Models](../openhgnn/models/#Model)

### Supported Models with specific task

The link will give some basic usage.

| Model                                                     | Node classification | Link prediction    | Recommendation     |
| --------------------------------------------------------- | ------------------- | ------------------ | ------------------ |
| [Metapath2vec](../openhgnn/output/metapath2vec)[KDD 2017] | :heavy_check_mark:  |                    |                    |
| [RGCN](../openhgnn/output/RGCN)[ESWC 2018]                | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [HERec](../openhgnn/output/HERec)[TKDE 2018]              | :heavy_check_mark:  |                    |                    |
| [HAN](../openhgnn/output/HAN)[WWW 2019]                   | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [KGCN](../openhgnn/output/KGCN)[WWW 2019]                 |                     |                    | :heavy_check_mark: |
| [HetGNN](../openhgnn/output/HetGNN)[KDD 2019]             | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [HeGAN](../openhgnn/output/HeGAN)[KDD 2019]               | :heavy_check_mark:  |                    |                    |
| HGAT[EMNLP 2019]                                          |                     |                    |                    |
| [GTN](../openhgnn/output/GTN)[NeurIPS 2019] & fastGTN     | :heavy_check_mark:  |                    |                    |
| [RSHN](../openhgnn/output/RSHN)[ICDM 2019]                | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [GATNE-T](../openhgnn/output/GATNE-T)[KDD 2019]           |                     | :heavy_check_mark: |                    |
| [DMGI](../openhgnn/output/DMGI)[AAAI 2020]                | :heavy_check_mark:  |                    |                    |
| [MAGNN](../openhgnn/output/MAGNN)[WWW 2020]               | :heavy_check_mark:  |                    |                    |
| [HGT](../openhgnn/output/HGT)[WWW 2020]                   | :heavy_check_mark:  |                    |                    |
| [CompGCN](../openhgnn/output/CompGCN)[ICLR 2020]          | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [NSHE](../openhgnn/output/NSHE)[IJCAI 2020]               | :heavy_check_mark:  |                    |                    |
| [NARS](../openhgnn/output/NARS)[arxiv]                    | :heavy_check_mark:  |                    |                    |
| [MHNF](../openhgnn/output/MHNF)[arxiv]                    | :heavy_check_mark:  |                    |                    |
| [HGSL](../openhgnn/output/HGSL)[AAAI 2021]                | :heavy_check_mark:  |                    |                    |
| [HGNN-AC](../openhgnn/output/HGNN_AC)[WWW 2021]           | :heavy_check_mark:  |                    |                    |
| [HeCo](../openhgnn/output/HeCo)[KDD 2021]                 | :heavy_check_mark:  |                    |                    |
| [SimpleHGN](../openhgnn/output/SimpleHGN)[KDD 2021]       | :heavy_check_mark:  |                    |                    |
| [HPN](../openhgnn/output/HPN)[TKDE 2021]                  | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [RHGNN](../openhgnn/output/RHGNN)[arxiv]                  | :heavy_check_mark:  |                    |                    |
| [HDE](../openhgnn/output/HDE)[ICDM 2021]                  |                     | :heavy_check_mark: |                    |
| [HetSANN](./openhgnn/output/HGT)[AAAI 2020]               | :heavy_check_mark:  |                    |                    |
| [ieHGCN](./openhgnn/output/HGT)[TKDE 2021]                | :heavy_check_mark:  |                    |                    |

### Candidate models

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification[EMNLP 2019]
- [Heterogeneous Information Network Embedding with Adversarial Disentangler[TKDE 2021]](https://ieeexplore.ieee.org/document/9483653)

## Contributors

OpenHGNN Team[GAMMA LAB], DGL Team and Peng Cheng Laboratory.

See more in [CONTRIBUTING](../CONTRIBUTING.md).
