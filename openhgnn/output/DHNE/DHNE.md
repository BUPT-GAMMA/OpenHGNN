# DHNE[AAAI2018]

Paper: [Structural Deep Embedding for Hyper-Networks](https://arxiv.org/abs/1711.10146)

Code from author: https://github.com/tadpole/DHNE

### How to Run

Clone the OpenHGNN

```
# For hypergraph task
python main.py -m DHNE -d drug -t hypergraph -g 0 --use_best_config
```

If you do not have gpu, set -gpu -1.

Candidate dataset: 

​	drug/GPS/MovieLens/wordnet

### Dataset

Supported dataset: drug, GPS, MovieLens, wordnet

The details of datasets can be seen in the graph below:

| datasets  |                 #(V)                 |  #(E)  |
| :-------: | :----------------------------------: | :----: |
|    GPS    |  user(146) location(70) activity(5)  |  1436  |
| MovieLens |   user(2113) movie(5908) tag(9079)   | 47957  |
|   drug    |  user(12) drug(1076) reaction(6398)  | 171756 |
|  wordnet  | head(40504) relation(18) tail(40551) | 145966 |

Note: Every node in these four datasets should have the same embedding and feature dimension.

You can download the dataset by

```
#drug4DHNE
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/drug4DHNE.zip
#gps4DHNE
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/gps4DHNE.zip
#movielens4DHNE
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/movielens4DHNE.zip
#wordnet4DHNE
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/wordnet4DHNE.zip

```

Or you can run the code mentioned above and it will download automatically.

### performance

Task : Hypergraph embedding

Evaluation metric: AUC_ROC

| Hypergraph Embedding | drug               | GPS                | MovieLens          | wordnet           |
| -------------------- | ------------------ | ------------------ | ------------------ | ----------------- |
| DHNE                 | 0.8875533694940287 | 0.8124717066545948 | 0.7212837083686715 | 0.684985935830281 |

### TrainerFlow: DHNE_trainer

The model is  trained in hypergraph embedding.

### Hyper-parameter specific to the model

You can modify the parameters [DHNE] in openhgnn/config.ini

### More

#### Contirbutor

Zhaoteng Li, Yiling Yuan [GAMMA LAB]

#### If you have any questions,

Submit an issue or email to edlee@bupt.edu.cn, yuanyiling@bupt.edu.cn

