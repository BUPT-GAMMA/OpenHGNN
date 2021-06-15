# OpenHGNN
This is an open-source toolkit for Heterogeneous Graph Neural Network(OpenHGNN) based on [DGL [Deep Graph Library]](https://github.com/dmlc/dgl). We integrate SOTA models of heterogeneous graph.

[Test Version]For now, we just release some models for some researchers to run the experiments.

## Keys

- Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and datasets. TODO: [Feature]AutoML & hyper-parameter search.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
- Efficiency: The backend dgl provides efficient APIs.

## Why OpenHGNN

### Relation with [DGL](https://github.com/dmlc/dgl)

OpenHGNN is a high-level package built on top of DGL; it will not cover efficient implementation of low-level components in C++.

### Relation with [OpenHINE](https://github.com/BUPT-GAMMA/OpenHINE)

| Package  | Heterograph structure | Models                            | Efficiency  | Others                                     |
| -------- | --------------------- | --------------------------------- | ----------- | ------------------------------------------ |
| OpenHINE | write by our own      | Most embedding methods            | Inefficient | ---                                        |
| OpenHGNN | heterograph           | Most graph neural network methods | Efficient   | Better Extensibility, Better Encapsulation |

To avoid reinventing the wheel, we use the DGL as our backend. So we build the new toolkit OpenHGNN and more models will be integrated into it. OpenHINE will not be updated.

![image](./docs/source/image-001.jpg)

## Get Started

#### Requirements and Installation

- Python  >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/)  >= 1.7.1
- [DGL](https://github.com/dmlc/dgl) >= 0.6.1

#### Running an existing baseline model on an existing benchmark [dataset](./openhgnn/dataset/#Dataset)

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config
```

-g controls which gpu you will use. If you do not have gpu, set -g -1.

--use_best_config means you can use the best config in the dataset with the model. If you want to set the different hyper-parameter, modify the [openhgnn.config.ini](./openhgnn/config.ini) manually.

e.g.: 

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config
```

It is under development, and we release it in a nightly build version. For now, we just give some new models, such as HetGNN, NSHE, GTN, MAGNN, RSHN.

## [Models](./openhgnn/models/#Model)

### Supported Models

- [HAN](./openhgnn/output/HAN)[WWW 2019] for node classification
- [HetGNN](./openhgnn/output/HetGNN)[KDD 2019] for node classification and link prediction
- [GTN](./openhgnn/output/GTN)[NeurIPS 2019] for node classification
- [RSHN](./openhgnn/output/RSHN)[ICDM 2019] for entity classification
- [MAGNN](./openhgnn/output/MAGNN)[WWW 2020] for node classification
- [NSHE](./openhgnn/output/NSHE)[IJCAI 2020] for node classification

### To be supported models

- Metapath2vec[KDD 2017]
- [RGCN](./openhgnn/output/RGCN)[ESWC 2018] for node classification
- CompGCN[ICLR 2020] for node classification

### Candidate models

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification[EMNLP 2019]
- Heterogeneous Graph Structure Learning for Graph Neural Networks[AAAI 2021]
- [Heterogeneous Graph Propagation Network[TKDE2021]](https://ieeexplore.ieee.org/abstract/document/9428609)
- [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning[KDD 2021]](https://arxiv.org/abs/2105.09111)

## Contributors

**[GAMMA LAB](https://github.com/BUPT-GAMMA) [BUPT]**: [Tianyu Zhao](https://github.com/Theheavens), Cheng Yang, Xiao Wang, [Chuan Shi](http://shichuan.org/)

**BUPT**: Jiahang Li

**DGL Team**: Quan Gan, [Jian Zhang](https://github.com/zhjwy9343)

