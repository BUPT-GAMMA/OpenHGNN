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

**Latest version: OpenHGNN v0.9.0.** v0.9 adds 10 model contributions, the `node_regression` task, model reproduction documentation, and clearer DGL-based model integration guidance.

## News

<details>
<summary>
2026-06-26 release v0.9
</summary>
<br/>

We release OpenHGNN v0.9.

- Added 10 model contributions, corresponding to 11 registered model names: HGDL, HGEN, HGSketch, HGOT, RMR, HERO/HERO_homo, SEHTGNN, HTGformer, HCMGNN, and RelGT.
- The current version registers 83 model names and 17 task/flow entries.
- Added the `node_regression` task for continuous node-label prediction.
- Updated documentation entry points, quick start, model overview, task overview, and model PR checklist.
- Strengthened model contribution standards around DGL implementation, trainerflow, dataset, README, and smoke-test requirements.

</details>

<details>
<summary>
2024-07-23 release v0.7
</summary>
<br/>

We released OpenHGNN v0.7.0.
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

We released OpenHGNN v0.5.0.
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

We released OpenHGNN v0.4.

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

We released OpenHGNN v0.3.

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

We released OpenHGNN v0.2.

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
- Discoverability: CLI commands can inspect the environment, registered models, registered tasks, and datasets.

Current version statistics:

- Registered model names: 83.
- Registered task/flow entries: 17.
- v0.9 model contributions: 10. `HERO` and `HERO_homo` are two registered names for one model contribution.
- v0.9 task addition: `node_regression`.

Common documentation entry points:

- [v0.9 release notes](./docs/source/release/v0.9.rst)
- [Quick start](./docs/source/get_started/quick_start.rst)
- [Model overview](./docs/source/get_started/model_overview.rst)
- [Model reproduction guide](./docs/source/get_started/reproduce_model.rst)
- [Task overview and node_regression](./docs/source/get_started/task_overview.rst)
- [Model PR checklist](./docs/source/advanced_materials/model_pr_checklist.rst)

## Get Started

#### Requirements and Installation

- Python 3.10-3.12

- [PyTorch](https://pytorch.org/get-started/) 2.3.x-2.4.x

- [DGL](https://github.com/dmlc/dgl) 2.2.x-2.4.x

- CPU or NVIDIA GPU, Linux, Python3

**1. Python environment (Optional):** We recommend using Conda package manager

Officially recommended environments:

- Primary: `Python 3.11 + PyTorch 2.4.0 + DGL 2.4.0+cu121`
- Compatibility: `Python 3.10 + PyTorch 2.3.1 + DGL 2.2.1`

The repository file `environment.yml` is pinned to the primary setup.

```bash
conda create -n openhgnn python=3.11
conda activate openhgnn
```

**2. Install Pytorch:** Follow their [tutorial](https://pytorch.org/get-started) to run the proper command according to
your OS and CUDA version. For example:

```bash
pip install torch==2.4.0 torchvision torchaudio
```

**3. Install DGL:** Follow their [tutorial](https://www.dgl.ai/pages/start.html) to run the proper command according to
your OS and CUDA version. For example:

```bash
pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
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
# To create the full pinned source environment directly, you can also use:
# conda env create -f environment.yml && conda activate openhgnn
pip install -r requirements.txt
pip install -e .
```

You can also use the packaged CLI to inspect the supported registry and current environment:

```bash
openhgnn list models
openhgnn list tasks
openhgnn env --format json
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

The current version registers 83 model names. The table below tracks the main model-task coverage maintained in README. Use `openhgnn list models` for the complete registered model list.

| Model | Node classification | Link prediction | Recommendation | Node regression | Notes |
| --- | --- | --- | --- | --- | --- |
| [TransE](../openhgnn/output/TransE)[NIPS 2013] |  | :heavy_check_mark: |  |  | Knowledge graph embedding |
| [TransH](../openhgnn/output/TransH)[AAAI 2014] |  | :heavy_check_mark: |  |  | Knowledge graph embedding |
| [TransR](../openhgnn/output/TransR)[AAAI 2015] |  | :heavy_check_mark: |  |  | Knowledge graph embedding |
| [TransD](../openhgnn/output/TransD)[ACL 2015] |  | :heavy_check_mark: |  |  | Knowledge graph embedding |
| [Metapath2vec](../openhgnn/output/metapath2vec)[KDD 2017] | :heavy_check_mark: |  |  |  | Representation learning |
| [RGCN](../openhgnn/output/RGCN)[ESWC 2018] | :heavy_check_mark: | :heavy_check_mark: |  |  |  |
| [HERec](../openhgnn/output/HERec)[TKDE 2018] | :heavy_check_mark: |  |  |  | Representation learning |
| [HAN](../openhgnn/output/HAN)[WWW 2019] | :heavy_check_mark: | :heavy_check_mark: |  |  |  |
| [KGCN](../openhgnn/output/KGCN)[WWW 2019] |  |  | :heavy_check_mark: |  |  |
| [HetGNN](../openhgnn/output/HetGNN)[KDD 2019] | :heavy_check_mark: | :heavy_check_mark: |  |  |  |
| [HeGAN](../openhgnn/output/HeGAN)[KDD 2019] | :heavy_check_mark: |  |  |  |  |
| HGAT[EMNLP 2019] |  |  |  |  | Short text classification |
| [GTN](../openhgnn/output/GTN)[NeurIPS 2019] & fastGTN | :heavy_check_mark: |  |  |  |  |
| [RSHN](../openhgnn/output/RSHN)[ICDM 2019] | :heavy_check_mark: | :heavy_check_mark: |  |  |  |
| [GATNE-T](../openhgnn/output/GATNE-T)[KDD 2019] |  | :heavy_check_mark: |  |  |  |
| [DMGI](../openhgnn/output/DMGI)[AAAI 2020] | :heavy_check_mark: |  |  |  |  |
| [MAGNN](../openhgnn/output/MAGNN)[WWW 2020] | :heavy_check_mark: |  |  |  |  |
| [HGT](../openhgnn/output/HGT)[WWW 2020] |  |  |  |  | Heterogeneous transformer |
| [CompGCN](../openhgnn/output/CompGCN)[ICLR 2020] | :heavy_check_mark: | :heavy_check_mark: |  |  |  |
| [NSHE](../openhgnn/output/NSHE)[IJCAI 2020] | :heavy_check_mark: |  |  |  |  |
| [NARS](../openhgnn/output/NARS)[arxiv] | :heavy_check_mark: |  |  |  |  |
| [MHNF](../openhgnn/output/MHNF)[arxiv] | :heavy_check_mark: |  |  |  |  |
| [HGSL](../openhgnn/output/HGSL)[AAAI 2021] | :heavy_check_mark: |  |  |  |  |
| [HGNN-AC](../openhgnn/output/HGNN_AC)[WWW 2021] | :heavy_check_mark: |  |  |  |  |
| [HeCo](../openhgnn/output/HeCo)[KDD 2021] | :heavy_check_mark: |  |  |  |  |
| [SimpleHGN](../openhgnn/output/SimpleHGN)[KDD 2021] | :heavy_check_mark: |  |  |  |  |
| [HPN](../openhgnn/output/HPN)[TKDE 2021] | :heavy_check_mark: | :heavy_check_mark: |  |  |  |
| [RHGNN](../openhgnn/output/RHGNN)[arxiv] | :heavy_check_mark: |  |  |  |  |
| [HDE](../openhgnn/output/HDE)[ICDM 2021] |  | :heavy_check_mark: |  |  |  |
| [HetSANN](../openhgnn/output/HGT)[AAAI 2020] | :heavy_check_mark: |  |  |  |  |
| [ieHGCN](../openhgnn/output/HGT)[TKDE 2021] | :heavy_check_mark: |  |  |  |  |
| [KTN](../openhgnn/output/KTN)[NeurIPS 2022] | :heavy_check_mark: |  |  |  |  |
| [HGDL](./docs/source/models/hgdl.rst)[NeurIPS 2024] | :heavy_check_mark: |  |  |  | v0.9 |
| [HGEN](./docs/source/models/hgen.rst)[IJCAI 2025] | :heavy_check_mark: |  |  |  | v0.9 |
| [HGSketch](./docs/source/models/hgsketch.rst)[SIGIR 2025] |  |  |  |  | v0.9, graph-level representation / graph classification pipeline |
| [HGOT](./docs/source/models/hgot.rst)[ICML 2025] | :heavy_check_mark: |  |  |  | v0.9 |
| [RMR](./docs/source/models/rmr.rst)[KDD 2024] | :heavy_check_mark: |  |  |  | v0.9 |
| [HERO](./docs/source/models/hero.rst)[ICLR 2024] | :heavy_check_mark: |  |  |  | v0.9, heterogeneous version |
| [HERO_homo](./docs/source/models/hero.rst)[ICLR 2024] | :heavy_check_mark: |  |  |  | v0.9, homogeneous version |
| [SEHTGNN](./docs/source/models/sehtgnn.rst)[NeurIPS 2025] | :heavy_check_mark: | :heavy_check_mark: |  | :heavy_check_mark: | v0.9 |
| [HTGformer](./docs/source/models/htgformer.rst)[SIGIR 2025] | :heavy_check_mark: | :heavy_check_mark: |  | :heavy_check_mark: | v0.9 |
| [HCMGNN](./docs/source/models/hcmgnn.rst)[IJCAI 2024] |  |  | :heavy_check_mark: |  | v0.9 |
| [RelGT](./docs/source/models/relgt.rst)[arXiv 2025] |  | :heavy_check_mark: |  |  | v0.9, RelBench task |

### v0.9 model statistics

| Item | Count | Note |
| --- | ---: | --- |
| New model contributions | 10 | Counted by paper/model contribution |
| New registered model names | 11 | `HERO` and `HERO_homo` are registered separately |
| Current registered model names | 83 | Based on `openhgnn.models.SUPPORTED_MODELS` |
| Current registered task/flow entries | 17 | Based on `openhgnn.tasks.SUPPORTED_TASKS` |

v0.9 model contributions: HGDL, HGEN, HGSketch, HGOT, RMR, HERO/HERO_homo, SEHTGNN, HTGformer, HCMGNN, and RelGT.

For v0.9 model reproduction entries, tasks, datasets, and remaining
documentation notes, see the [v0.9 release notes](./docs/source/release/v0.9.rst)
and the [model reproduction guide](./docs/source/get_started/reproduce_model.rst).

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
