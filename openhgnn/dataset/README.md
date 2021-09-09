Dataset

A dataset is related to a task, so it is a part of task.

So dataset should load not only a heterograph[DGLGraph], but also some index involving training, validation and testing. In OpenHGNN, we preprocess the feature of dataset outside of model. Specifically, we use a linear layer with bias for each node type to map all node features to a shared feature space. And for no feature nodes, we give a embedding as its feature. Refer to [HeteroFeature](https://openhgnn.readthedocs.io/en/latest/api/layer.html#heterofeature).

#### NodeClassificationDataset

- ##### RDF_NodeCLassification

  - [AIFB/MUTAG/BGS/AM](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero)

- ##### HIN_NodeCLassification

  - ###### ACM

    |             | author | paper | Subject | Paper-Author | Paper-Subject | Features                      | Train | Val  | Test  |
    | ----------- | ------ | ----- | ------- | ------------ | ------------- | ----------------------------- | ----- | ---- | ----- |
    | acm4GTN     | 5,912  | 3,025 | 57      | 9,936        | 3,025         | 1,902                         | 600   | 300  | 2,125 |
    | acm_han_raw | 17,351 | 4,025 | 72      | 13,407       | 4,025         | 1,903                         | 808   | 401  | 2,816 |
    | acm4NSHE    | 7,167  | 4,019 | 60      | 13,407       | 4,019         | 128(Embedding from deep walk) | -     | -    | -     |

  - ###### academic4HetGNN

    |                 | author | paper  | Venue | Paper-Author | Paper-venue | Paper-paper |
    | --------------- | ------ | ------ | ----- | ------------ | ----------- | ----------- |
    | academic4HetGNN | 28,646 | 21,044 | 18    | 69,311       | 21,044      | 21,357      |

  - ###### DBLP

    |            | author | paper  | Conf | Venue | Paper-Author | Paper-Conf | Paper-Term | Train | Val  | Test  |
    | ---------- | ------ | ------ | ---- | ----- | ------------ | ---------- | ---------- | ----- | ---- | ----- |
    | dblp4HAN   | 4,057  | 14,328 | 20   | 8,789 | 19,645       | 14,328     | 88,420     | 800   | 400  | 2,857 |
    | dblp4GTN   |        |        |      |       |              |            |            |       |      |       |
    | dblp4MAGNN | 4,057  | 14,328 | 20   | 7,723 | 19,645       | 14,328     | 85,810     | 400   | 400  | 3257  |

    

  - ###### IMDB

    |            | Movie | Actor | Director | Movie-Actor | Movie-Director | Train | Val  | Test  |
    | ---------- | ----- | ----- | -------- | ----------- | -------------- | ----- | ---- | ----- |
    | imdb4HAN   | 4,780 | 5,841 | 2,269    | 14,340      | 4,780          | 300   | 300  | 2,687 |
    | imdb4GTN   | 4,661 | 5,841 | 2,270    | 13,983      | 4,661          | 300   | 300  | 2,339 |
    | imdb4MAGNN | 4,278 | 5,257 | 2,081    | 12,828      | 4,278          | 400   | 400  | 3,478 |

- **HGB_NodeClassification**

  - 
  
- ##### OGB_NodeClassification

  - ###### [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)

    - Train 629,571 predict venue labels of all papers published before 2018
    - Validation: 64879 papers published in 2018
    - Test: 41939 papers published since 2019

#### LinkPredictionDataset

- ##### HIN_LinkPrediction

  - ###### academic4HetGNN

- **HGBl-LinkPrediction**

  - 
  
- ##### KG_LinkPrediction

  - 'wn18', 'FB15k', 'FB15k-237'
  
- **HGB_LinkPrediction**    

  - HGBl-amazon

    node information:

    | type    | number | feature dimension |
    | ------- | ------ | ----------------- |
    | product | 10099  | 1156              |

    edge information:

    | type             | train number | test number |
    | ---------------- | ------------ | ----------- |
    | product-product0 | 69315        | 7609        |
    | product-product1 | 64598        | 7137        |

  - HGBl-LastFM

    node information:

    | type   | number | feature dimension |
    | ------ | ------ | ----------------- |
    | user   | 1892   | 0                 |
    | artist | 17632  | 0                 |
    | tag    | 1088   | 0                 |


    edge information:

    | type        | train number | test number |
    | ----------- | ------------ | ----------- |
    | user-artist | 74267        | 18567       |
    | user-user   | 25434        | 0           |
    | artist-tag  | 23253        | 0           |

  - HGBl-PubMed

    node information:

    | type | number | feature dimension |
    | ---- | ------ | ----------------- |
    | 0    | 13168  | 200               |
    | 1    | 19507  | 200               |
    | 2    | 25691  | 200               |
    | 3    | 2783   | 200               |

    edge information:

    | type   | train number | test number |
    | ------ | ------------ | ----------- |
    | 0_to_0 | 16105        | 0           |
    | 0_to_1 | 25962        | 0           |
    | 1_to_1 | 34109        | 8528        |
    | 2_to_0 | 31277        | 0           |
    | 2_to_1 | 51323        | 0           |
    | 2_to_2 | 62187        | 0           |
    | 2_to_3 | 6297         | 0           |
    | 3_to_0 | 3155         | 0           |
    | 3_to_1 | 5245         | 0           |
    | 3_to_3 | 798          | 0           |

    
#### RecommendationDataset

- **Amazon**

  (Containing rating and timestamp information)

  (Source : http://jmcauley.ucsd.edu/data/amazon/)

  Edata['rate'] in user-item edge is the rating.

  It addresses the two most common scenarios in collaborative filtering:

  - rating prediction (e.g. on a scale of 1 to 5 stars), and
  - item prediction from positive-only feedback.

- 
  
  |        | User  | Item  | View  | Category | Brand | User-Item | Item-View | Item-Category | Item-Brand | Test(20%)<br />User-Item |
  | ------ | ----- | ----- | ----- | -------- | ----- | --------- | --------- | ------------- | ---------- | ------------------------ |
  | Amazon | 6,170 | 2,753 | 3,857 | 22       | 334   | 195,791   | 5,694     | 5,508         | 2,753      | 39,159                   |


  **yelp4rec**


### How to build a new dataset

#### Overview

We use [dgl.heterograph](https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous) as our graph data structure.

The API [dgl.save_graphs](https://docs.dgl.ai/en/latest/generated/dgl.save_graphs.html) and  [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html#) can be used in storing graph into the local.

##### The Flow

1. Process your dataset as [dgl.heterograph](https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous). 
2. Store as *graph.bin*. Compress as *dataset_name4model_name.zip*
3. Upload the zip file to s3.
4. If the dataset is Heterogeneous Information Network, you can modify the [AcademicDataset](./academic_graph.py) directly. Or you can refer to it building a new *Class Dataset*.

We give a [demo](https://openhgnn.readthedocs.io/en/latest/install/usage.html#evaluate-a-new-dataset) to build a new dataset.