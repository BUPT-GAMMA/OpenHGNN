# OpenHGNN
This is an open-source toolkit for Heterogeneous Graph Neural Network(OpenHGNN) based on [DGL [Deep Graph Library]](https://github.com/dmlc/dgl). We integrate SOTA models of heterogeneous graph.

[Test Version]For now, we just release some models for some researchers to run the experiments.

[Release Plan]We want to release version0.1 in September.

## Key Features

- Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and datasets using [optuna](https://optuna.org/) which is a hyperparameter optimization framework.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
- Efficiency: The backend dgl provides efficient APIs.

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

--use_hpo Besides use_best_config, we give a hyper-parameter [example](./openhgnn/auto) to search the best hyper-parameter automatically.

e.g.: 

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config
```

It is under development, and we release it in a nightly build version. For now, we just give some new models, such as HetGNN, NSHE, GTN, MAGNN, RSHN.

## Why OpenHGNN

### Relation with [DGL](https://github.com/dmlc/dgl)

OpenHGNN is a high-level package built on top of DGL; it will not cover efficient implementation of low-level components in C++.

### Relation with [OpenHINE](https://github.com/BUPT-GAMMA/OpenHINE)

| Package  | Heterograph structure | Models                            | Efficiency  | Others                                     |
| -------- | --------------------- | --------------------------------- | ----------- | ------------------------------------------ |
| OpenHINE | write by our own      | Most embedding methods            | Inefficient | ---                                        |
| OpenHGNN | heterograph           | Most graph neural network methods | Efficient   | Better Extensibility, Better Encapsulation |

We aim to build more embedding methods(implemented in OpenHINE) in OpenHGNN.

To avoid reinventing the wheel, we use the DGL as our backend. So we build the new toolkit OpenHGNN and more models will be integrated into it. OpenHINE will not be updated.

![image](./docs/source/image-001.jpg)



## [Models](./openhgnn/models/#Model)

### Supported Models

- [RGCN](./openhgnn/output/RGCN)[ESWC 2018] for entity classification
- [HAN](./openhgnn/output/HAN)[WWW 2019] for node classification
- [HetGNN](./openhgnn/output/HetGNN)[KDD 2019] for node classification and link prediction
- [GTN](./openhgnn/output/GTN)[NeurIPS 2019] for node classification
- [RSHN](./openhgnn/output/RSHN)[ICDM 2019] for entity classification
- [MAGNN](./openhgnn/output/MAGNN)[WWW 2020] for node classification
- [CompGCN](./openhgnn/output/CompGCN)[ICLR 2020] for entity classification
- [NSHE](./openhgnn/output/NSHE)[IJCAI 2020] for node classification
- [NARS](./openhgnn/output/NARS) for node classification
- [MHNF](./openhgnn/output/MHNF) for node classification

### To be supported models

- Metapath2vec[KDD 2017]

### Candidate models

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification[EMNLP 2019]
- Heterogeneous Graph Structure Learning for Graph Neural Networks[AAAI 2021]
- [Heterogeneous Graph Neural Network via Attribute Completion[WWW 2021]](https://dl.acm.org/doi/10.1145/3442381.3449914)
- [Heterogeneous Graph Propagation Network[TKDE 2021]](https://ieeexplore.ieee.org/abstract/document/9428609)
- [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning[KDD 2021]](https://arxiv.org/abs/2105.09111)
- Heterogeneous Information Network Embedding with Adversarial Disentangler[TKDE 2021]

## A Pipeline of OpenHGNN

###### We define three components: [TrainerFlow](./openhgnn/trainerflow/#Trainerflow), [Model](./openhgnn/models/#Model), [Task](./openhgnn/tasks/#Task).

- For a given paper,
  - For unsupervised model, the model and the trainerflow are specific cause the loss calculation is fixed.So given a model name, we can identify the model and trainerflow, user need also specify the task name.
  - For other model, the model is fixed and the trainerflow is changeable according to task.
    User must specify the model and the task, and the trainerflow will be determined by the task.
    For example, RGCN on the node classification task will use the entity classification flow and RGCN on the link prediction task will use the dist_mult flow.

![image-20210622151239030](./docs/source/image-002.png)

## Contributors

**[GAMMA LAB](https://github.com/BUPT-GAMMA) [BUPT]**: [Tianyu Zhao](https://github.com/Theheavens), Cheng Yang, Xiao Wang, [Chuan Shi](http://shichuan.org/)

**BUPT**: Jiahang Li

**DGL Team**: Quan Gan, [Jian Zhang](https://github.com/zhjwy9343)

