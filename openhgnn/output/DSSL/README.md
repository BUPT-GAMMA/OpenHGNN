# DSSL[NIPS 2022]

Paper: [**Decoupled Self-supervised Learning for Graphs
**](https://arxiv.org/pdf/2206.03601.pdf)


### How to run

- Clone the Openhgnn-DGL

- Candidate dataset: Cora, Pubmed, Citeseer

  ```bash
  python main.py -m DSSL -t DSSL_trainer -d Cora -g 0
  ```
  We recommend running on a GPU
  If you do not have gpu, set -g -1.

## Performance
Task: node classfication
Evaluation : ACC

If GCNConv of the pyg library is used as the encoder of the model DSSL, the accuracy of the model on dataset will be higher than that of the GraphConv encoder of the DGL library.

encoder with GraphConv(dgl):
| Method    |  Cora  |  Pubmed  |  Citeseer  |
|-----------|--------|----------|------------|
| **DSSL**  |   77   |     75   |     64     |

encoder with GCNConv(pyg):
| Method    |  Cora  |  Pubmed  |  Citeseer  |
|-----------|--------|----------|------------|
| **DSSL**  |   80   |     82   |     67     |

Note:Since our openhgnn supports at least version 0.8 of DGL, this model does not support the Texas and Cornell heterogeneous graph datasets newly added by version 1.1 of DGL. At the same time, the tests of the Penn94 and Twitch datasets in the original paper used the datasets of the PYG library, but DGL still does not support them.
When our minimum supported DGL version is updated, we will update the model to support the above dataset

## TrainerFlow: [DSSL_trainer(node classfication)](../../trainerflow/DSSL_trainer.py)

### Model

- DSSL

## Hyper-parameter specific to the model

  You can modify the parameters in openhgnn/config.ini

#### If you have any questions,

  Submit an issue or email to [1614384662@qq.com](mailto:1614384662@qq.com).