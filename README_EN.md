# OpenHGNN

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BUPT-GAMMA/OpenHGNN)
[![Documentation Status](https://readthedocs.org/projects/openhgnn/badge/?version=latest)](https://openhgnn.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/BUPT-GAMMA/OpenHGNN)
[![visitors](https://visitor-badge.glitch.me/badge?page_id=BUPT-GAMMA.OpenHGNN)](https://github.com/BUPT-GAMMA/OpenHGNN)
[![Total lines](https://img.shields.io/tokei/lines/github/BUPT-GAMMA/OpenHGNN?color=red)](https://github.com/BUPT-GAMMA/OpenHGNN)

[**启智社区（中文版）**](https://git.openi.org.cn/GAMMALab/OpenHGNN) | [**OpenHGNN [CIKM2022]**](https://dl.acm.org/doi/abs/10.1145/3511808.3557664) |  [**Space4HGNN [SIGIR2022]**](../space4hgnn) | [**Benchmark&Leaderboard**](../openhgnn/dataset/ohgb.md) | [**Slack Channel**](https://app.slack.com/client/TDM5126J1/C03J6GND001)

This is an open-source toolkit for Heterogeneous Graph Neural Network based
on [DGL [Deep Graph Library]](https://github.com/dmlc/dgl) and [PyTorch](https://pytorch.org/). We integrate SOTA models
of heterogeneous graph.

## News
<details>
<summary>
2024-07-23 release v0.7
</summary>
<br/>

We release the latest version v0.7.0
- New models and datasets.
- Graph Prompt pipeline
- Data process frame: dgl.graphBolt
- New GNN aggregator: dgl.sparse
- Distributed training

</details>



<details>
<summary>
2023-07-17 release v0.5
</summary>
<br/>

We release the latest version v0.5.0
- New models and datasets.
- 4 New tasks: pretrain, recommendation, graph attacks and defenses, abnorm_event detection.
- TensorBoard visualization.
- Maintenance and test module.

</details>


<details>

<summary>
2023-02-24 OpenI Excellent Incubation Award
</summary>
<br/>

OpenHGNN won the Excellent Incubation Program Award of OpenI Community! For more details：https://mp.weixin.qq.com/s/PpbwEdP0-8wG9dsvRvRDaA

</details>

<details>

<summary>
2023-02-21 First Prize of CIE
</summary>
<br/>

The algorithm library supports the project of "Intelligent Analysis Technology and Scale Application of Large Scale Complex Heterogeneous Graph Data" led by BUPT and participated by ANT GROUP, China Mobile, Haizhi Technology, etc. This project won the first prize of the 2022 Chinese Intitute of Electronics "Science and Technology Progress Award".

</details>
<details>
<summary>
2023-01-13 release v0.4
</summary>
<br/>

We release the latest version v0.4.

- New models
- Provide pipelines for applications
- More models supporting mini-batch training
- Benchmark for million-scale graphs

</details>

<details>
<summary>
2022-08-02 paper accepted
</summary>
<br/>
Our paper [<i> OpenHGNN: An Open Source Toolkit for Heterogeneous Graph Neural Network </i>](https://dl.acm.org/doi/abs/10.1145/3511808.3557664) is accpeted at CIKM 2022 short paper track.
</details>

<details>
<summary>
2022-06-27 release v0.3
</summary>
<br/>

We release the latest version v0.3.

- New models
- API Usage
- Simply customization of user-defined datasets and models
- Visualization tools of heterogeneous graphs

</details>

<details>
<summary>
2022-02-28 release v0.2
</summary>
<br/>

We release the latest version v0.2.

- New Models
- [Space4HGNN [SIGIR2022]](../space4hgnn)
- [Benchmark&Leaderboard](../openhgnn/dataset/ohgb.md)

</details>

<details>
<summary>
2022-01-07 加入启智社区
</summary>
<br/>
启智社区用户可以享受到如下功能：

- 全新的中文文档
- 免费的计算资源—— [云脑使用教程](https://git.openi.org.cn/GAMMALab/OpenHGNN/src/branch/main/yunnao_tutorial.md)
- OpenHGNN最新功能
  - 新增模型：【KDD2017】Metapath2vec、【TKDE2018】HERec、【KDD2021】HeCo、【KDD2021】SimpleHGN、【TKDE2021】HPN、【ICDM2021】HDE、fastGTN
  - 新增日志功能
  - 新增美团外卖数据集
  </details>
  
## Key Features

- Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and dataset.
  Besides, we also integrate [optuna](https://optuna.org/) to get hyperparameter optimization.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
- Efficiency: The backend dgl provides efficient APIs.

## Get Started

#### Requirements and Installation

- Python  >= 3.6

- [PyTorch](https://pytorch.org/get-started/)  >= 2.3.0

- [DGL](https://github.com/dmlc/dgl) >= 2.2.1

- CPU or NVIDIA GPU, Linux, Python3

**1. Python environment (Optional):** We recommend using Conda package manager

```bash
conda create -n openhgnn python=3.6
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
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

**4. Install openhgnn:** 

- install from pypi
```bash
pip install openhgnn
```

- install from source
```bash
git clone https://github.com/BUPT-GAMMA/OpenHGNN
# If you encounter a network error, try git clone from openi as following.
# git clone https://git.openi.org.cn/GAMMALab/OpenHGNN.git
cd OpenHGNN
pip install .
```


**5. Install gdbi(Optional):** 

- install gdbi from git
```bash
pip install git+https://github.com/xy-Ji/gdbi.git
```

- install graph database from pypi
```bash
pip install neo4j==5.16.0
pip install nebula3-python==3.4.0
```





#### Running an existing baseline model on an existing benchmark [dataset](../openhgnn/dataset/#Dataset)

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config --load_from_pretrained
```

usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
[--gpu GPU] [--use_best_config][--use_database]

*optional arguments*:

``-h, --help``    show this help message and exit

``--model -m ``    name of models

``--task -t``    name of task

``--dataset -d``    name of datasets

``--gpu -g``    controls which gpu you will use. If you do not have gpu, set -g -1.

``--use_best_config``    use_best_config means you can use the best config in the dataset with the model. If you want to
set the different hyper-parameter, modify the [openhgnn.config.ini](../openhgnn/config.ini) manually. The best_config
will override the parameter in config.ini.

``--load_from_pretrained`` will load the model from a default checkpoint.

``--use_database`` get dataset from database

``---mini_batch_flag`` train model with mini-batchs

``---graphbolt`` mini-batch training with dgl.graphbolt

``---use_distributed`` train model with distributed way

e.g.:

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config

python main.py -m RGCN -d imdb4GTN -t node_classification -g 0 --mini_batch_flag --graphbolt
```

**Note**: If you are interested in some model, you can refer to the below models list.

Refer to the [docs](https://openhgnn.readthedocs.io/en/latest/index.html) to get more basic and depth usage.

#### Use TensorBoard to visualize your train result
```bash
tensorboard --logdir=./openhgnn/output/{model_name}/
```
e.g.：
```bash
tensorboard --logdir=./openhgnn/output/RGCN/
```
**Note**: To visualize results, you need to train the model first.


#### Use gdbi to get grpah dataset
take neo4j and imdb dataset for example
- construct csv file for dataset(node-level:A.csv,edge-level:A_P.csv)
- import csv file to database
```bash
LOAD CSV WITH HEADERS FROM "file:///data.csv" AS row
CREATE (:graphname_labelname {ID: row.ID, ... });
```
- add user information to access database in config.py file
```python
self.graph_address = [graph_address]
self.user_name = [user_name]
self.password = [password]
```

- e.g.:

```bash
python main.py -m MAGNN -d imdb4MAGNN -t node_classification -g 0 --use_best_config --use_database
```




## [Models](../openhgnn/models/#Model)

### Supported Models with specific task

The link will give some basic usage.

| Model                                                     | Node classification | Link prediction    | Recommendation     |
|-----------------------------------------------------------|---------------------|--------------------|--------------------|
| [TransE](../openhgnn/output/TransE)[NIPS 2013]            |                     | :heavy_check_mark: |                    |
| [TransH](../openhgnn/output/TransH)[AAAI 2014]            |                     | :heavy_check_mark: |                    |
| [TransR](../openhgnn/output/TransR)[AAAI 2015]            |                     | :heavy_check_mark: |                    |
| [TransD](../openhgnn/output/TransD)[ACL 2015]             |                     | :heavy_check_mark: |                    |
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
| [HetSANN](../openhgnn/output/HGT)[AAAI 2020]               | :heavy_check_mark:  |                    |                    |
| [ieHGCN](../openhgnn/output/HGT)[TKDE 2021]                | :heavy_check_mark:  |                    |                    |
| [KTN](../openhgnn/output/KTN)[NIPS 2022]                  | :heavy_check_mark: |                    |                    |

### Candidate models

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification[EMNLP 2019]
- [Heterogeneous Information Network Embedding with Adversarial Disentangler[TKDE 2021]](https://ieeexplore.ieee.org/document/9483653)

## Contributors

OpenHGNN Team[GAMMA LAB], DGL Team and Peng Cheng Laboratory.

See more in [CONTRIBUTING](../CONTRIBUTING.md).


## Cite OpenHGNN

If you use OpenHGNN in a scientific publication, we would appreciate citations to the following paper:

```
@inproceedings{han2022openhgnn,
  title={OpenHGNN: An Open Source Toolkit for Heterogeneous Graph Neural Network},
  author={Hui Han, Tianyu Zhao, Cheng Yang, Hongyi Zhang, Yaoqi Liu, Xiao Wang, Chuan Shi},
  booktitle={CIKM},
  year={2022}
}
```
