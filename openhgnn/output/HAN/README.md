# HAN[WWW2019]

Paper: [**Heterogeneous Graph Attention Network**](https://dl.acm.org/doi/abs/10.1145/3308558.3313562)

Code from author: https://github.com/Jhy1993/HAN

Code from dgl Team: https://github.com/dmlc/dgl/tree/master/examples/pytorch/han

We use the code from dgl Team and integrate the model into our package. 

#### How to run

Clone the Openhgnn-DGL

```bash
python main.py -m HAN -t node_classification -d acm_han_raw -g 0
```

If you do not have gpu, set -gpu -1.

Candidate dataset: acm_han_raw, acm4GTN, imdb4GTN

#### Performance

| acm_han_raw   | Macro-F1 | Micro-F1 |
| ------------- | -------- | -------- |
| HAN[OpenHGNN] | 90.11    | 90.02    |

| Node classification | acm4GTN | imdb4GTN |
| ------------------- | ------- | -------- |
| GTN[OpenHGNN]       | 92.22   | 61.58    |
| HAN[OpenHGNN]       | 91.63   | 57.30    |

### TrainerFlow: node_classification

The model is  trained in semi-supervisied node classification.

#### Model

We implement HANLayer with MetapathConv.

SemanticAttention

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
num_channels = 2 # number of channel
num_layers = 3 # number of layer
adaptive_lr_flag = True # use different learning rate for weight in GTLayer.
```

Best config can be found in [best_config](../../utils/best_config.py)

### Related API in DGL

[dgl.metapath_reachable_graph](https://docs.dgl.ai/en/latest/generated/dgl.metapath_reachable_graph.html)

[GATConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#gatconv)

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).
