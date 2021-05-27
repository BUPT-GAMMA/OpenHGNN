# Dataset

A dataset is related to a task. So dataset should load not only a heterograph, but also some index involving training, validation and testing.

For now, we have two downstream tasks, which are node classification and link prediction.

#### NodeClassificationDataset

- ##### RDF_NodeCLassification

  - AIFB/MUTAG/BGS/AM

- ##### HIN_NodeCLassification

  - acm4GTN/academic4HetGNN

- ##### OGB_NodeCLassification

  - ogbn-mag

#### LinkPredictionDataset

- ##### HIN_LinkPrediction

  - academic4HetGNN

- ##### KG_LinkPrediction

  - 'wn18', 'FB15k', 'FB15k-237'

### How to build a new dataset

We use [dgl.heterograph](https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous) as our graph data structure.

The API [dgl.save_graphs](https://docs.dgl.ai/en/latest/generated/dgl.save_graphs.html) and  [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html#) can be used in storing graph into the local.

