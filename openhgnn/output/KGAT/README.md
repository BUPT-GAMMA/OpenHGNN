# KGAT[KDD2019]

Paper: [**KGAT: Knowledge Graph Attention Network forRecommendation**](https://arxiv.org/pdf/1905.07854v2.pdf)

The author's [codes](https://github.com/xiangwang1223/knowledge_graph_attention_network) is written by tensorflow. 
Our implement is reproduced based on a pytorch version provided by [LunaBlack](https://github.com/LunaBlack/KGAT-pytorch)

#### How to run

Clone the Openhgnn-DGL

Candidate dataset: yelp2018,amazon-book,last-fm
To distinguish it from other models' datasets, we changed their names:
  ·yelp2018_KGAT for yelp2018
  ·amazon-book_KGAT for amazon-book
  ·LastFM_KGAT for last-fm

```bash
python main.py -m HPN -t node_classification -d acm_han_raw -g 0
```

If you do not have gpu, set -gpu -1.

#### Performance

| HPN[OpenHGNN] | Macro-F1 | Micro-F1 |
| ------------- | -------- | -------- |
| acm_han_raw   | 91.80    | 91.80    |
| acm4GTN       | 91.04    | 90.92    |
| imdb4GTN      | 60.96    | 64.00    |



### TrainerFlow: node_classification

The model is  trained in semi-supervisied node classification.

#### model

HPNLayer

SemanticFusion

### Dataset

Supported dataset: [acm_han_raw](../../dataset/#ACM), [acm4GTN](../../dataset/#ACM), [imdb4GTN](../../dataset/#IMDB)

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/acm_han_raw.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/acm4GTN.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/imdb4GTN.zip
```

### Hyper-parameter specific to the model

```python
k_layer = 1 # iterations in Semantic Propagation
alpha = 0.1 # Value of restart probability
out_embedsize = 64 # Dimension of the final embedding
edge_drop = 0 # the dropout rate on edges that controls the messages received by each node
```

Best config can be found in [best_config](../../utils/best_config.py)

### Related API in DGL

[dgl.metapath_reachable_graph](https://docs.dgl.ai/en/latest/generated/dgl.metapath_reachable_graph.html)

[APPNPConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#appnpconv)

## More

#### Contirbutor

Donglin Xia[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [donglin.xia@bupt.edu.cn](mailto:donglin.xia@bupt.edu.cn).
