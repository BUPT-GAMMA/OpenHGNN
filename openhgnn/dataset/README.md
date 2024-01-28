## Dataset

A dataset is related to a task, so it is a part of task.

So dataset should load not only a heterograph[DGLGraph], but also some index involving training, validation and testing. In OpenHGNN, we preprocess the feature of dataset outside of model. Specifically, we use a linear layer with bias for each node type to map all node features to a shared feature space. And for no feature nodes, we give a embedding as its feature. Refer to [HeteroFeature](https://openhgnn.readthedocs.io/en/latest/api/layer.html#heterofeature).

### Open Heterogeneous Graph Benchmark

We release a benchmark of heterogeneou graph named [Open Heterogeneous Graph Benchmark (OHGB)](./ohgb.md) and leaderboard for link prediction and node classification task. 

### NodeClassificationDataset

- ##### RDF_NodeCLassification（Please use lowercase letters）
  
  - [AIFB/MUTAG/BGS/AM](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero)

- ##### HIN_NodeCLassification
  
  - ###### ACM
    
    |             | author | paper | Subject | Paper-Author | Paper-Subject | Features                      | Train | Val | Test  |
    | ----------- | ------ | ----- | ------- | ------------ | ------------- | ----------------------------- | ----- | --- | ----- |
    | acm4GTN     | 5,912  | 3,025 | 57      | 9,936        | 3,025         | 1,902                         | 600   | 300 | 2,125 |
    | acm_han_raw | 17,351 | 4,025 | 72      | 13,407       | 4,025         | 1,903                         | 808   | 401 | 2,816 |
    | acm4NSHE    | 7,167  | 4,019 | 60      | 13,407       | 4,019         | 128(Embedding from deep walk) | -     | -   | -     |
  
  - ###### academic4HetGNN
    
    |                 | author | paper  | Venue | Paper-Author | Paper-venue | Paper-paper |
    | --------------- | ------ | ------ | ----- | ------------ | ----------- | ----------- |
    | academic4HetGNN | 28,646 | 21,044 | 18    | 69,311       | 21,044      | 21,357      |
  
  - ###### DBLP
    
    |            | author | paper  | Conf | Venue | Paper-Author | Paper-Conf | Paper-Term | Train | Val | Test  |
    | ---------- | ------ | ------ | ---- | ----- | ------------ | ---------- | ---------- | ----- | --- | ----- |
    | dblp4HAN   | 4,057  | 14,328 | 20   | 8,789 | 19,645       | 14,328     | 88,420     | 800   | 400 | 2,857 |
    | dblp4GTN   |        |        |      |       |              |            |            |       |     |       |
    | dblp4MAGNN | 4,057  | 14,328 | 20   | 7,723 | 19,645       | 14,328     | 85,810     | 400   | 400 | 3257  |
  Note: dblp4HAN dataset is not avaliable now.Please use other dataset. metapath2vec embedding is not available for dblp4GTN, one solution is to run metapath2vec on the dataset to generate the embedding and then assign it as node feature before running Model(The embedding file contains all node types in the same order as g.ntypes.)
- ###### IMDB
  
  |            | Movie | Actor | Director | Movie-Actor | Movie-Director | Train | Val | Test  |
  | ---------- | ----- | ----- | -------- | ----------- | -------------- | ----- | --- | ----- |
  | imdb4HAN   | 4,780 | 5,841 | 2,269    | 14,340      | 4,780          | 300   | 300 | 2,687 |
  | imdb4GTN   | 4,661 | 5,841 | 2,270    | 13,983      | 4,661          | 300   | 300 | 2,339 |
  | imdb4MAGNN | 4,278 | 5,257 | 2,081    | 12,828      | 4,278          | 400   | 400 | 3,478 |

- **HGBn **
  
  The datasets are HGB for Node Classification
  
  **Note**：The test data labels are randomly replaced to prevent data leakage issues, refer to [HGB](https://github.com/THUDM/HGB).
  
  In OpenHGNN, you will get the test results in `./openhgnn/output/{model_name}/`.  If you want to obtain test scores, you need to submit your prediction to HGB's [website](https://www.biendata.xyz/hgb/).
  
  - HGBn-ACM
  
  | paper | author | subject | term | paper-author | paper-paper | paper-subject | paper-term | Val | Test |
  | ----- | ------ | ------- | ---- | ------------ | ----------- | ------------- | ---------- | --- | ---- |
  | 3025  | 5959   | 56      | 1902 | 9949         | 5343        | 3025          | 255619     | 907 | 2118 |
  
  - HGBn-IMDB(is not avaliable now)
  
  | movie | actor | director | keyword | actor-movie | director-movie | keyword-movie | train | test |
  | ----- | ----- | -------- | ------- | ----------- | -------------- | ------------- | ----- | ---- |
  | 4932  | 6124  | 2393     | 7971    | 14779       | 4932           | 23610         | 1371  | 3202 |
  
  - HGBn-Freebase: no feature
  
  | BOOK  | BUSINESS | FILM  | LOCATION | MUSIC | ORGANIZATION | PEOPLE | SPORTS | train | test |
  | ----- | -------- | ----- | -------- | ----- | ------------ | ------ | ------ | ----- | ---- |
  | 40402 | 7153     | 19427 | 9368     | 82351 | 2731         | 17641  | 1025   | 2386  | 5568 |
  
  - HGBn-DBLP
  
  | author | paper | term | venue | author-paper | paper-term | paper-venue | train | test |
  | ------ | ----- | ---- | ----- | ------------ | ---------- | ----------- | ----- | ---- |
  | 4057   | 14328 | 7723 | 20    | 19645        | 85810      | 14328       | 1217  | 2840 |

- ##### OGB_NodeClassification
  
  - ###### [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)
    
    - Train 629,571 predict venue labels of all papers published before 2018
    - Validation: 64879 papers published in 2018
    - Test: 41939 papers published since 2019

### LinkPredictionDataset

- ##### HIN_LinkPrediction
  
  - ###### academic4HetGNN(is not avaliable now)

- **HGBl**
  
  The datasets are HGB for Link Prediction.
  
  **Note**：The test data labels are randomly replaced to prevent data leakage issues, refer to [HGB](https://github.com/THUDM/HGB).
  
  In OpenHGNN, you will get the test results in `./openhgnn/output/{model_name}/`.  If you want to obtain test scores, you need to submit your prediction to HGB's [website](https://www.biendata.xyz/hgb/).
  
  - HGBl-amazon
    
    |             | product | features | product-product0 | product-product1 | test : product-product0 | test : product-product1 |
    | ----------- | ------- | -------- | ---------------- | ---------------- | ----------------------- | ----------------------- |
    | HGBl-amazon | 10099   | 1156     | 76924            | 71735            | 7609                    | 7137                    |
  
  - HGBl-LastFM
    
    |             | user | artist | tag  | feature | user-artist | user-user | artist-tag | test:user-artist |
    | ----------- | ---- | ------ | ---- | ------- | ----------- | --------- | ---------- | ---------------- |
    | HGBL-LastFM | 1892 | 17632  | 1088 | 0       | 92834       | 25434     | 23253      | 18567            |
  
  - HGBl-PubMed
    
    |             | node0 | node1 | node2 | node3 | feature | node0- node0 | node0-node1 | node1-node1 | node2-node0 | node2-node1 | node2-node2 | node2-node3 | node3-node0 | node3-node1 | node3-node2 | test:node1-node1 |
    | ----------- | ----- | ----- | ----- | ----- | ------- | ------------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ---------------- |
    | HGBL-LastFM | 13168 | 19507 | 25691 | 2783  | 200     | 16105        | 25962       | 42637       | 31277       | 51323       | 62187       | 6297        | 3155        | 5245        | 798         | 8528             |

- ##### KG_LinkPrediction
  
  - 'wn18', 'FB15k', 'FB15k-237'

### RecommendationDataset

- **Amazon**
  
  (Containing rating and timestamp information)
  
  (Source : http://jmcauley.ucsd.edu/data/amazon/)
  
  Edata['rate'] in user-item edge is the rating.
  
  It addresses the two most common scenarios in collaborative filtering:
  
  - rating prediction (e.g. on a scale of 1 to 5 stars), and
  - item prediction from positive-only feedback.
  
  |        | User  | Item  | View  | Category | Brand | User-Item | Item-View | Item-Category | Item-Brand | Test(20%)<br />User-Item |
  | ------ | ----- | ----- | ----- | -------- | ----- | --------- | --------- | ------------- | ---------- | ------------------------ |
  | Amazon | 6,170 | 2,753 | 3,857 | 22       | 334   | 195,791   | 5,694     | 5,508         | 2,753      | 39,159                   |

- MTWM
  
  |      | user    | poi   | Sup    | poi-contain-spu | user-buy-poi | user-buy-spu | user-click-poi |
  | ---- | ------- | ----- | ------ | --------------- | ------------ | ------------ | -------------- |
  | MTWM | 188,155 | 3,474 | 16,889 | 92,024          | 542,915      | 1,797,283    | 1,477,316      |

- 

### How to build a new dataset

#### Overview

We use [dgl.heterograph](https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous) as our graph data structure.

The API [dgl.save_graphs](https://docs.dgl.ai/en/latest/generated/dgl.save_graphs.html) and  [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html#) can be used in storing graph into the local.

##### The Flow

1. Process your dataset as [dgl.heterograph](https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous). 
2. Store as *graph.bin*. Compress as *dataset_name4model_name.zip*
3. Upload the zip file to s3.
4. If the dataset is Heterogeneous Information Network, you can modify the [AcademicDataset](./academic_graph.py) directly. Or you can refer to it building a new *Class Dataset*.

We give a [demo](https://openhgnn.readthedocs.io/en/latest/advanced_materials/developer_guide.html#evaluate-a-new-dataset) to build a new dataset.
