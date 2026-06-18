# HGEN[IJCAI 2025]

Paper: [**Heterogeneous Graph Ensemble Networks**](https://www.ijcai.org/proceedings/2025/)

Code from author: https://github.com/Chrisshen12/HGEN

#### How to run

Clone the OpenHGNN repository

```bash
python main.py -m HGEN -t node_classification -d dblp4GTN -g 0
```

If you do not have gpu, set -gpu -1.

Candidate dataset: acm4GTN, imdb4GTN, dblp4GTN

#### Performance

Node classification (default hyperparameters, tested with official PyG datasets: DBLP, IMDB, HGBDataset(ACM))

| Dataset | Macro-F1 | Micro-F1 |
| ------- | -------- | -------- |
| DBLP (4-class author) | 88.15 | 89.09 |
| IMDB (3-class movie) | 56.31 | 57.51 |
| ACM (3-class paper)  | 89.62 | 89.46 |

> **Note**: The paper reports DBLP Macro-F1=94.18% and IMDB Micro-F1=61.58%, which require hyperparameter tuning not included in the official code. With default parameters, the official PyG code also yields similar results to the above table (verified by running the official source code directly).

### TrainerFlow: HGEN_trainer

The model is trained in semi-supervised node classification.

#### Model

We implement HGEN with MetapathConv pattern: for each meta-path, extract a homogeneous subgraph via
[dgl.metapath_reachable_graph](https://docs.dgl.ai/en/latest/generated/dgl.metapath_reachable_graph.html),
run multiple independent GCN encoders, fuse with attention (min-max normalization + residual), and decode.
Final prediction sums all meta-path logits. Loss includes L1 regularization on the path correlation matrix.

### Dataset

Supported dataset: dblp4GTN, acm4GTN, imdb4GTN

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/dblp4GTN.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/acm4GTN.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/imdb4GTN.zip
```

Or run the code mentioned above and it will download automatically.

### Hyper-parameter specific to the model

```python
hidden_dim = 64        # hidden layer dimension
out_dim = 16           # output dimension (num classes)
num_gcn = 3            # number of GCN models per meta-path
num_layers = 2         # number of GCN layers
attention_dim = 8      # attention projection dimension
dropout = 0.1          # dropout probability
lambda_cov = 0         # regularization coefficient for Gram matrix (0 = disabled)
max_epoch = 500        # max training epochs
patience = 50          # early stopping patience
```

Best config can be found in [best_config](../../utils/best_config.py)

### Related API in DGL

[dgl.metapath_reachable_graph](https://docs.dgl.ai/en/latest/generated/dgl.metapath_reachable_graph.html)

[dgl.nn.pytorch.conv.GraphConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#graphconv)

## More

#### Contirbutor

HGEN reproduction in OpenHGNN

#### If you have any questions,

Submit an issue or refer to the original paper and code repository.
