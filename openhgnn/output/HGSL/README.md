# HGSL[AAAI 2021]

Paper: [Heterogeneous Graph Structure Learning for Graph Neural Networks](http://www.shichuan.org/doc/100.pdf)

Code from author:
https://github.com/Andy-Border/HGSL

### How to run

Clone the Openhgnn-DGL

```bash
python main.py -m HGSL -d acm4GTN -t node_classification -g 0 --use_best_config
```

If you do not have gpu, set -gpu -1.
	
### Performance

Node classification

| Node classification | acm4GTN macro-f1 | acm4GTN micro-f1 | 
| ------------------- | ----------------  | ----------------  |
| paper               | 93.48             | 93.37             |
| OpenHGNN            | 93.28             | 93.18             |

| Node classification | dblp4GTN macro-f1 | dblp4GTN micro-f1 | 
| ------------------- | ----------------  | ----------------  |
| paper               | 91.92             | 92.77             |
| OpenHGNN            | 91.28             | 92.18             |

| Node classification | yelp4HGSL macro-f1 | yelp4HGSL micro-f1 | 
| ------------------- | ----------------  | ----------------  |
| paper               | 93.55             | 92.76             |
| OpenHGNN            | 93.27             | 92.09             |

### TrainerFlow: node_classification

The model is trained in semi-supervisied node classification.

#### model

- [GraphGenerator](../../models/HGSL.py)
  - Contain MetricCalcLayer.
  - Generate a graph according to two feature matrices.
- [MetricCalcLayer](../../models/HGSL.py)
  - Calculate the metric.
- [GraphChannelAttLayer](../../models/HGSL.py)
  - Perform the channel attention operation on several similarity graphs. 
- [GCN](../../models/HGSL.py)
  - Contain the GraphConvolution.
  - The downstream GNN.
- [GraphConvolution](../../models/HGSL.py)
  - The graph convolution layer of GCN.

### Dataset

Supported dataset: acm4GTN, dblp4GTN, yelp4HGSL

We process the [acm4GTN](../../dataset/#ACM) dataset with adding the metapath2vec embeddings obtained from the dataset of the author's code.

Requirements for datasets
- The graph should be an undirected heterogeneous graph.
- Every node type in graph should have its feature named 'h' and the same feature dimension.
- Every node type in graph should have its metapath2vec embedding feature named 'xxx_m2v_emb' and the same feature dimension.

### Hyper-parameter specific to the model

```python
hidden_dim = 16
num_heads = 2
gnn_emd_dim = 64
```
The best config for each dataset can be found in [best_config](../../utils/best_config.py).

### Related API in DGL

[dgl.DGLGraph.adj](https://docs.dgl.ai/generated/dgl.DGLGraph.adj.html#dgl.DGLGraph.adj)

## Note
This model under the best config has some slight differences compared with the code given by the paper author，which seems having little impact on performance:
1. The regularization item in loss is on all parameters of the model, while in the author's code, it is only on the generated adjacent matrix. If you want to implement the latter, a new task of OpenHGNN is needed.
2. The normalization of input adjacent matrix is separately on different adjacent matrices of different relations, while in the author's code, it is on the entire adjacent matrix composed of adjacent matrices of all relations.

## More

#### Contirbutor

Xinlong Zhai[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [zhaijojo@bupt.edu.cn](mailto:zhaijojo@bupt.edu.cn).
