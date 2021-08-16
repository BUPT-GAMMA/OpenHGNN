# NARS

Paper:[Scalable Graph Neural Networks for Heterogeneous Graphs](https://arxiv.org/abs/2011.09679)

Code from author:[https://github.com/facebookresearch/NARS](https://github.com/facebookresearch/NARS)

## How to run

Clone the Openhgnn-DGL

```python
python main.py -m NARS -t node_classification -d acm4NARS -g 0 --use_best_config
```

Candidate dataset: acm4NARS

If you do not have gpu, set -gpu -1.

## candidate dataset

acm4NARS

NOTE: NARS can handle mag oag dataset, we will add these two datasets in our further work.

### description

- Number of nodes

  | paper    | 4025 |
  | -------- | ---- |
  | author    | 17431 |
  | field   | 73 |
    
-   Number of edges

    | paper-author | 13407  |
    | -------------- | ----- |
    | paper-field    | 4025 |
    
-   Subsets: paper-author, paper-field


## performance

Node classification

| |accuracy|
|----|----|
|acm4NARS|0.93|


## TrainerFlow: node_classification

### model

- NARS

    NARS is composed of WeightedAggregator and SIGN.

- WeightedAggregator
    
    Get new features by multiplying the old features by the weight matrix.
    
- SIGN

    The MLP classifier. It is composed  of a several linear layers. Then project the node embeddings to the vector space to predict the type of the nodes.


## Hyper-parameter specific to the model

```python
R = 2
input_dropout = True
cpu_preprocess = True
ff_layer = 2
```

Best config can be found in [best_config](file:/E:/资料/科研/复现论文/OpenHGNN-main/openhgnn/utils/best_config.py)

## More

#### Contirbutor

Tianyu Zhao, Yibo Li[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [liushiliushi0@gmail.com](mailto:liushiliushi0@gmail.com).
