# CompGCN[ICLR 2020]

Paper: [**Composition- based Multi-Relational Graph Convolutional Networks**](https://arxiv.org/abs/1911.03082)

Code: https://github.com/malllabiisc/CompGCN

#### How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m CompGCN -t node_classification -d aifb -g 0
  ```

  If you do not have gpu, set -gpu -1.

  Supported Dataset: [RDFDataset](../../dataset/#RDF_NodeCLassification)

## Performance: Node classification

| Method           | AIFB  | MUTAG | BGS  | AM   |
| ---------------- | ----- | ----- | ---- | ---- |
| **CompGCN(dgl)** | 91.66 |       |      |      |

### TrainerFlow: [node classification flow](../../trainerflow/#Node classification flow)

### Model

- [dglnn.HeteroGraphConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#heterographconv) 

## Hyper-parameter specific to the model

You can modify the parameters in openhgnn/config.ini

#### Description

```python
hidden_dim
batch_size
```

Best config can be found in [best_config](../../utils/best_config.py)

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).