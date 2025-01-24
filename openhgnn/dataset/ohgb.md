# Open Heterogeneous Graph Benchmark

We release a benchmark of heterogeneou graph for link prediction and node classification task.

## Node Classification Task

| **Dataset**      | #node types | #nodes     | #link types | #links     | #attributes dimension | #attributed nodes | #label types | #labeled nodes |
| ---------------- | ----------- | ---------- | ----------- | ---------- | --------------------- | ----------------- | ------------ | -------------- |
| `ohgbn-acm`      | 3           | 8,994      | 2           | 25,922     | 1,902                 | 8,994             | 3            | 3,025          |
| `ohgbn-imdb`     | 3           | 12,772     | 4           | 37,288     | 1,256                 | 12,772            | 3            | 2,939          |
| `ohgbn-yelp2`    | 4           | 82,465     | 4           | 30,542,675 | N/A                   | N/A               | 16           | 5,484          |
| `ohgbn-Freebase` | 8           | 12,164,758 | 36          | 62,982,566 | N/A                   | N/A               | 8            | 47,190         |

### Dataset `ohgbn-acm`

|             | author | paper | Subject | Paper-Author | Paper-Subject | Features | Train | Val | Test  |
| ----------- | ------ | ----- | ------- | ------------ | ------------- | -------- | ----- | --- | ----- |
| `ohgbn-acm` | 5,912  | 3,025 | 57      | 9,936        | 3,025         | 1,902    | 600   | 300 | 2,125 |

**Graph:** The `ohgbn-acm` dataset construct a heterogeneous graph that contains 3025 papers (P), 5912 authors (A) and 57 subjects (S) from [ACM](https://dl.acm.org/), as well as four types of directed relations connecting two types of entities —an author “writes” a paper, a paper “written by” a paper, a paper "has" a specific subject, and a subject "belong to" a paper. Paper features correspond to elements of a bag-of-words represented of keywords.

**Prediction task:** Given the heterogeneous `ohgbn-acm` data, the task is to predict the venue (conference or journal) of each paper, given its keywords, subjects. Specifically, papers published in KDD, SIGMOD, SIGCOMM, MobiCOMM, and VLDB and divide the papers into three classes (Database, Wireless Communication, Data Mining).

**Dataset splitting:** Random split. Train/Validation/Test: 600/300/2,125

**Leaderboard:**

Evaluation metric is Macro-F1 and Micro-F1.

| `ohgbn-acm` | Macro-F1 | Micro-F1 |
| ----------- | -------- | -------- |
| MHNF        | 0.9259   | 0.9252   |
| RGCN        | 0.9242   | 0.9242   |
| HAN         | 0.9245   | 0.9233   |
| NARS        | 0.9241   | 0.9233   |
| RSHN        | 0.9226   | 0.9233   |
| HPN         | 0.9214   | 0.9200   |
| GTN         | 0.9203   | 0.9200   |
| SimpleHGN   | 0.9189   | 0.9181   |
| CompGCN     | 0.9128   | 0.9125   |
| HGT         | 0.8946   | 0.8941   |

### Dataset `ohgbn-imdb`

**Graph:** The `ohgbn-imdb` dataset construct a heterogeneous graph that contains 4,661
movies (M), 5841 actors (A) and 2,270 directors (D), as well as four types of directed relations connecting two types of entities —a movie “starred by” an actor, an actor “star in” a movie, a director “direct” a movie and a movie "directed by" a actor. Movie features correspond to elements of a bag-of-words represented of plots.

**Prediction task:** Given the heterogeneous `ohgbn-imdb` data, the task is to predict the genre of each movie, given its plots, actors and directors. Specifically, the movies are divided into three classes (Action, Comedy, Drama) according to their genre.

**Dataset splitting:** Random split. Train/Validation/Test: 300/300/2,339

|              | Movie | Actor | Director | Movie-Actor | Movie-Director | Train | Val | Test  |
| ------------ | ----- | ----- | -------- | ----------- | -------------- | ----- | --- | ----- |
| `ohgbn-imdb` | 4,661 | 5,841 | 2,270    | 13,983      | 4,661          | 300   | 300 | 2,339 |

**Leaderboard:**

Evaluation metric is Macro-F1 and Micro-F1.

| `ohgbn-imdb` | Macro_f1 | Micro_f1 |
| ------------ | -------- | -------- |
| RGCN         | 0.5757   | 0.6366   |
| RSHN         | 0.5914   | 0.6127   |
| MHNF         | 0.5913   | 0.6114   |
| CompGCN      | 0.5869   | 0.6148   |
| HAN          | 0.5863   | 0.6037   |
| GTN          | 0.5791   | 0.6003   |
| NARS         | 0.5470   | 0.6259   |
| HPN          | 0.5596   | 0.5703   |
| SimpleHGN    | 0.5521   | 0.5635   |
| HGT          | 0.5440   | 0.5519   |

### Dataset `ohgbn-yelp2`

**Graph:** The `ohgbn-yelp2` dataset is a heterogeneous network which contains four types of nodes, business, users, locations, and reviews from [Yelp](https://www.yelp.com/dataset/challenge). Nodes do not have features, but a large portion of businesses are labeled into sixteen categories. Each labeled business has one or multiple labels.  
**Prediction task:** Given the heterogeneous `ohgbn-yelp2` data, the task is to classify the types of business. The task is to leverage the entire network to infer the labels of the books. In total, there are 16 type of business, making the prediction task a 16-class classification problem.  
**Dataset splitting:** Random Split. We have 5,484 labeled nodes, 4387 nodes are used for training, and 1097 nodes are used for testing.

|               | business | location | stars | phrase | business-phrase | business-location | business-stars | phrase-phrase | Train | Test |
| ------------- | -------- | -------- | ----- | ------ | --------------- | ----------------- | -------------- | ------------- | ----- | ---- |
| `ohgbn-yelp2` | 7,474    | 39       | 9     | 74,943 | 3,317,891       | 7,474             | 7,474          | 27,209,836    | 4387  | 1097 |

**Leaderboard:**

Evaluation metric is Macro-F1 and Micro-F1.

| `ohgbn-yelp2` | Macro_f1 | Micro_f1 |
| ------------- | -------- | -------- |
| RGCN          | 0.0504   | 0.4044   |
| SimpleHGN     | 0.0504   | 0.4044   |

### Dataset `ohgbn-Freebase`

**Graph:** The `ohgbn-Freebase` dataset is a Knowledge Graph (KG) extracted from [Freebase](http://www.freebase.com/). It contains eight type of nodes, books, films, music, sports, people, locations, organizations, and business. There is no feature in nodes.

A large portion of books are labeled into eight genres of literature. Each labeled book has only one label.

**Prediction task:** Given the KG `ohgbn-Freebase`, the task is to classify the types of books. In total, there are 8 type of books, making the prediction task a 8-class classification problem.

**Dataset splitting:** As many of the books are not labeled, so we choose the labeled nodes and divide train/test data by the ratio 8:2. There are 44833 labeled nodes, 35866 nodes are used for training, and 8967 nodes are used for testing.

**Leaderboard:**

Evaluation metric is Macro-F1 and Micro-F1.

| `ohgbn-Freebase` | Macro_f1 | Micro_f1 |
| ---------------- | -------- | -------- |
| RGCN             | 0.5307   | 0.6933   |

## Link Prediction Task

| **Dataset**   | #node types | #nodes    | #link types | #links     | #attributes dimension |
| ------------- | ----------- | --------- | ----------- | ---------- | --------------------- |
| `ohgbl-MTWM`  | 3           | 208,518   | 4           | 3,909,538  | N/A                   |
| `ohgbl-yelp1` | 3           | 2,353,365 | 4           | 10,417,742 | N/A                   |

### Dataset `ohgbl-MTWM`

|              | user    | poi   | spu    | poi-spu | user-poi | user-spu  | user-poi  | Train     | Test(positive:negative) |
| ------------ | ------- | ----- | ------ | ------- | -------- | --------- | --------- | --------- | ----------------------- |
| `ohgbl-MTWM` | 188,155 | 3,474 | 16,889 | 92,024  | 542,915  | 1,797,283 | 1,477,316 | 1,218,439 | 578,844(0.5:0.5)        |

**Graph:** The `ohgbl-MTWM` dataset is a heterogeneous network collected from Meituan Waimai APP. It contains three types of entities—users (188,155 nodes), pois (point of interest, 3,474 nodes), and spus (standard product unit, 16,889 nodes)—as well as four types of directed relations connecting two types of entities—a user “buys” something from a poi, a user “buys” a spu, a poi “contains” a spu, and a user “clicks” a poi. All types of entities are not associated with input node features.  

**Prediction task:** Given the heterogeneous `ohgbl-MTWM` data, the task is to predict whether a user buys a spu, given his/her history behaviors.   
**Dataset splitting:** We consider user-buy-spu links and divide training/test data sequentially. The data in first three weeks is for training and the rest one week is for testing.

**Leaderboard:**

Evaluation metric is roc_auc.

| `ohgbl-MTWM` | roc_auc |
| ------------ | ------- |
| RGCN         | 0.7044  |
| CompGCN      | 0.6988  |

### Dataset `ohgbl-yelp1`

|               | user      | business | city | category | business-category | business-city | user-user | user-business | Train     | Test      |
| ------------- | --------- | -------- | ---- | -------- | ----------------- | ------------- | --------- | ------------- | --------- | --------- |
| `ohgbl-yelp1` | 2,189,457 | 160,585  | 836  | 2,487    | 708,903           | 160,585       | 585,204   | 8,963,050     | 7,728,178 | 1,234,872 |

**Graph:** The `ohgbl-yelp1` dataset is a heterogeneous network collected from [Yelp](https://www.yelp.com/dataset/challenge). It contains four types of entities—users (2,189,457 nodes), business (160,585 nodes), and city (836 nodes) and category (2,487 nodes)—as well as four types of directed relations connecting two types of entities—a business is "classified" into a category, a business is "located in" a city, a user is "a friend with" a user and a user "buys" a business . All types of entities are not associated with input node features.

**Prediction task:** Given the heterogeneous `ohgbl-yelp1` data, the task is to predict whether a user buys a business, given his/her history behaviors.  
**Dataset splitting:** We consider user-buy-business links and divide training/test data sequentially. The data in 2014-2019 is for training and in 2020-2021 is for testing.

**Leaderboard:**

Evaluation metric is roc_auc

| `ohgbl-yelp1` | roc_auc |
| ------------- | ------- |
| CompGCN       | 0.6121  |

## How to run

If you want to reproduce the results, you could run it with the following command line.

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config --load_from_pretrained
```

e.g.:

```bash
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_best_config
```

## How to download

If you just want to download the dataset,

```bash
# wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/{dataset_name}.zip

wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/ohgbn-acm.zip
```
