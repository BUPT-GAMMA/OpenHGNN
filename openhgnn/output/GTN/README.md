# GTN[NeurIPS2019] & fastGTN

Paper: [[GTN] Graph Transformer Networks](https://arxiv.org/abs/1911.06455)

Extension Paper: [[fastGTN] Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs](https://arxiv.org/abs/2106.06218.pdf)

Code from author: https://github.com/seongjunyun/Graph_Transformer_Networks

### How to run

Clone the OpenHGNN

```bash
# Run GTN
python main.py -m GTN -t node_classification -d acm4GTN -g 0 --use_best_config
# Run the fastGTN
python main.py -m fastGTN -t node_classification -d acm4GTN -g 0 --use_best_config
```

If you do not have gpu, set -gpu -1.

##### Candidate dataset: 

​	acm4GTN/imdb4GTN/dblp4GTN

### Performance

Node classification

| Node classification(F1 score) | acm4GTN                   | imdb4GTN                  | dblp4GTN                                 |
|-------------------------------|---------------------------|---------------------------|------------------------------------------|
| paper[GTN]                    | 92.68                     | 60.92                     | 94.18                                    |
| OpenHGNN[GTN]                 | Macro: 92.03 Micro: 92.00 | Macro: 56.97 Micro: 58.61 | 87.33(OOM on Tesla T4(16GB), cpu result) |
| OpenHGNN[fastGTN]             | Macro: 92.92 Micro: 92.85 | Macro: 60.62 Micro: 62.59 | Macro: 90.39 Micro: 91.39                |

### TrainerFlow: node_classification

The model is  trained in semi-supervisied node classification.

#### model

- [transform_relation_graph_list](../../utils/utils.py)
  - Extract a  graph list where every graph just contains a relation.
- [GTLayer](../../models/GTN_sparse.py)
  - Contain GTConv
  - Contain the product of the adjacency matrices of two graphs getting from GTConv.
- [GTConv](../../models/GTN_sparse.py)
  - Create a weighted graph whose adjacency matrix is the sum of the adjacency matrices of the given graph list.

### Dataset

Supported dataset: acm4GTN, imdb4GTN, dblp4ACM

Note: Every node in dataset should have the same features dimension.

[Dataset Description](https://openhgnn.readthedocs.io/en/latest/api/dataset.html#hgb-node-classification-dataset)

#### [acm4GTN](../../dataset/#ACM)/[imdb4GTN](../../dataset/#IMDB)

We process the acm dataset given by [HAN](https://github.com/Jhy1993/HAN). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/acm4GTN.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/imdb4GTN.zip
```

Or run the code mentioned above and it will download automaticlly.

### Hyper-parameter specific to the model

```python
num_channels = 2 # number of channel
num_layers = 3 # number of layer
adaptive_lr_flag = True # use different learning rate for weight in GTLayer.
```

Best config can be found in [best_config](../../utils/best_config.py)

### Related API in DGL

[dgl.adj_product_graph](https://docs.dgl.ai/en/latest/generated/dgl.adj_product_graph.html#dgl-adj-product-graph) which is equivalent SpSpMM.

[dgl.adj_sum_graph](https://docs.dgl.ai/en/latest/generated/dgl.adj_sum_graph.html#dgl.adj_sum_graph)

[GraphConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html?#graphconv), [EdgeWeightNorm](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html?#edgeweightnorm)

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).

