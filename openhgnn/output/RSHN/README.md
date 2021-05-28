# RSHN[ICDM2019]

-   paper: [Relation Structure-Aware Heterogeneous Graph Neural Network](https://ieeexplore.ieee.org/abstract/document/8970828)
-   Code from author: [RSHN](https://github.com/CheriseZhu/RSHN)

## How to run

- Clone the Openhgnn-DGL

  ```python
  python main.py -m RSHN -t node_classification -d aifb -g 0
  ```

  If you do not have gpu, set -gpu -1.

  the rdf_dataset is supported.

## Performance: Node classification

| Method               | AIFB      | MUTAG     | BGS       | AM        |
| -------------------- | --------- | --------- | --------- | --------- |
| **RSHN**             | **97.22** | **82.35** | **93.10** | **90.40** |
| **RSHN(dgl)**（best) | **97.22** | **85.29** | **93.10** | **89.39** |

### Dataset: [RDFDataset](../../dataset/#RDF_NodeCLassification)

## TrainerFlow: entity classification trainer

### Model

- *1) Coarsened Line Graph Neural Network (CL-GNN):*
  - We implement the API [coarsened_line_graph](../../sampler/RSHN_sampler.py)
- *2) Heterogeneous Graph Neural Network (H-GNN):*

## Hyper-Parameter

You can modify the parameters in openhgnn/config.ini

#### Description

```
[RSHN]
learning_rate = 0.01
weight_decay =  0.0005
dropout = 0.6

seed = 1233
hidden_dim = 8
max_epoch = 1000
rw_len = 5
batch_size = 1000
num_node_layer = 2
num_edge_layer = 1
patience = 50
validation = True
mini_batch_flag = False
```

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).