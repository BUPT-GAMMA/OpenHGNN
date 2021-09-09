# RGCN[ESWC 2018]

-   paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
-   Author's code for entity classification: https://github.com/tkipf/relational-gcn
-   Author's code for link prediction: https://github.com/MichSchli/RelationPrediction
-   Code from dgl Team: [rgcn](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn) / [rgcn-hetero](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero). 
-   Note: The difference between rgcn and rgcn-hetero is the storage form of node/edge type. The former use the dgl.NID stored in g.ndata, but the latter use the HeteroNodeView/HeteroEdgeView. And we use the latter storage form and nearly all models in OpenHGNN use the latter storage form as possible.

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m RGCN -t node_classification -d aifb -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  -d means dataset, candidate dataset: aifb/mutag/bgs/am. Refer to [RDFDataset](../../dataset/#RDF_NodeCLassification) to get more infos.

## Performance: Node classification

| Method               | AIFB  | MUTAG | BGS   | AM    |
| -------------------- | ----- | ----- | ----- | ----- |
| **RGCN(dgl)**（best) | 97.22 | 72.06 | 96.55 | 88.89 |

## TrainerFlow: [entity classification flow](../../trainerflow/#Entity_classification_flow)

### Model

- [dglnn.HeteroGraphConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#heterographconv) 
- dglnn.WeightBasis [The basis function decomposition]

## Hyper-parameter specific to the model

You can modify the parameters[RGCN] in openhgnn/config.ini. 

Best config can be found in [best_config](../../utils/best_config.py). 

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).

