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

Node classification

| Node classification (F1 score) | dblp4GTN Macro-F1 | dblp4GTN Micro-F1 |
| ------------------------------ | ------------------ | ------------------ |
| paper[HGEN]                    | 94.18              | -                  |
| OpenHGNN[HGEN]                 | TBD                | TBD                |

| Node classification (F1 score) | imdb4GTN Macro-F1 | imdb4GTN Micro-F1 |
| ------------------------------ | ------------------ | ------------------ |
| paper[HGEN]                    | -                  | 61.58              |
| OpenHGNN[HGEN]                 | TBD                | TBD                |

### TrainerFlow: node_classification

The model is trained in semi-supervised node classification.

#### Model

We implement HGEN with MetapathConv pattern: for each meta-path, extract a homogeneous subgraph via
[dgl.metapath_reachable_graph](https://docs.dgl.ai/en/latest/generated/dgl.metapath_reachable_graph.html),
run multiple independent GCN encoders, fuse with attention (residual +1/k), and decode.
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
lambda_cov = 0.01      # regularization coefficient for Gram matrix
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
