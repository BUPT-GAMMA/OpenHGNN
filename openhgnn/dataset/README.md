# Dataset

A dataset is related to a task. So dataset should load not only a heterograph, but also some index involving training, validation and testing.

For now, we have two downstream tasks, which are node classification and link prediction.

#### NodeClassificationDataset

- ##### RDF_NodeCLassification

  - AIFB/MUTAG/BGS/AM

- ##### HIN_NodeCLassification

  - acm4GTN/academic4HetGNN

  - DBLP

    - 

      |            | author | paper  | Conf | Venue | Paper-Author | Paper-Conf | Paper-Term | Train | Val  | Test  |
      | ---------- | ------ | ------ | ---- | ----- | ------------ | ---------- | ---------- | ----- | ---- | ----- |
      | dblp4HAN   | 4,057  | 14,328 | 20   | 8,789 | 19,645       | 14,328     | 88,420     | 800   | 400  | 2,857 |
      | dblp4GTN   |        |        |      |       |              |            |            |       |      |       |
      | dblp4MAGNN | 4,057  | 14,328 | 20   | 7,723 | 19,645       | 14,328     | 85,810     | 400   | 400  | 3257  |

      

  - IMDB

    - |            | Movie | Actor | Director | Movie-Actor | Movie-Director | Train | Val  | Test  |
      | ---------- | ----- | ----- | -------- | ----------- | -------------- | ----- | ---- | ----- |
      | imdb4HAN   | 4,780 | 5,841 | 2,269    | 14,340      | 4,780          | 300   | 300  | 2,687 |
      | imdb4GTN   |       |       |          |             |                | 300   | 300  | 2,339 |
      | imdb4MAGNN | 4,278 | 5,257 | 2,081    | 12,828      | 4,278          | 400   | 400  | 3,478 |

- ##### OGB_NodeCLassification

  - ogbn-[mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)

#### LinkPredictionDataset

- ##### HIN_LinkPrediction

  - academic4HetGNN

- ##### KG_LinkPrediction

  - 'wn18', 'FB15k', 'FB15k-237'

### How to build a new dataset

We use [dgl.heterograph](https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous) as our graph data structure.

The API [dgl.save_graphs](https://docs.dgl.ai/en/latest/generated/dgl.save_graphs.html) and  [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html#) can be used in storing graph into the local.

