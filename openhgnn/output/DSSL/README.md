# DSSL[NIPS 2022]

Paper: [**Decoupled Self-supervised Learning for Graphs
**](https://arxiv.org/pdf/2206.03601.pdf)


### How to run

- Clone the Openhgnn-DGL

- Candidate dataset: Cora, Pubmed, Texas

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

## TrainerFlow: [DSSL_trainer(node classfication)](../../trainerflow/DSSL_trainer.py)

### Model

- DSSL

## Hyper-parameter specific to the model

  You can modify the parameters in openhgnn/config.ini

#### If you have any questions,

  Submit an issue or email to [1614384662@qq.com](mailto:1614384662@qq.com).