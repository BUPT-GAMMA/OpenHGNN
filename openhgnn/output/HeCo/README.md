# HeCo[KDD2021]

Paper: [Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning](http://shichuan.org/doc/112.pdf)

Code from author: https://github.com/liun-online/HeCo

## How to run

Clone the Openhgnn-DGL

```python
python main.py -m HeCo -d acm4HeCo -t node_classification -g 0 --use_best_config
```

Candidate dataset: acm4HeCo

If you do not have gpu, set -gpu -1.

## candidate dataset

acm4HeCo

### Performance

Node classification 

| Node classification | acm4HeCo (Macro-F1 / Micro-F1 / AUC) |
| ------------------- | ----- |
| paper               | 89.04 / 88.71 / 96.55 |
| OpenHGNN            | 88.66 / 88.35 / 96.90 (mean of 10 random seeds) |

### TrainerFlow: HeCo_trainer

The model is  trained in unsupervisied node classification.

### Hyper-parameter specific to the model

```python
hidden_dim = 64
max_epoch = 10000
eva_lr = 0.05
eva_wd = 0
patience = 5
learning_rate = 0.0008
weight_decay = 0
tau = 0.8
feat_drop = 0.3
attn_drop = 0.5
sample_rate = author-7_subject-1
lam = 0.5
```

Best config can be found in  [best_config](../../utils/best_config.py)

### Related API in DGL

[dgl.sampling.sample_neighbors](https://docs.dgl.ai/generated/dgl.sampling.sample_neighbors.html#dgl.sampling.sample_neighbors)

[GraphConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html?#graphconv)

[GATConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html?#gatconv)

## More

#### Contirbutor

Nian Liu, Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [nianliu@bupt.edu.cn](mailto:nianliu@bupt.edu.cn).
