# HGDL[NeurIPS 2024]

Paper: [**Heterogeneous Graph Label Distribution Learning**](https://openreview.net/forum?id=OwguhIAh8R)

Code from author: https://github.com/Listener-Watcher/HGDL

## How to run

Clone the OpenHGNN-DGL

```bash
# DBLP dataset (4-class author label distribution)
python main.py -m HGDL -t node_classification -d dblp4HGDL -g 0 --use_best_config

# ACM dataset (14-class author label distribution)
python main.py -m HGDL -t node_classification -d acm4HGDL -g 0 --use_best_config
```

If you do not have gpu, set -gpu -1.

Candidate dataset: dblp4HGDL, acm4HGDL

## Performance

Label distribution learning metrics (lower is better for KL/Canberra/Chebyshev/Clark, higher is better for Cosine/Intersection):

| Dataset     | KL↓   | Cosine↑ | Canberra↓ | Chebyshev↓ | Clark↓ | Intersection↑ |
| ----------- | ----- | ------- | --------- | ---------- | ------ | ------------- |
| dblp4HGDL   | 0.0704 | 0.0198 | 2.8117    | 0.0637     | 1.6377 | 0.9356        |
| acm4HGDL    | 0.7514 | 0.1962 | 13.1013   | 0.3516     | 3.4767 | 0.6414        |

## TrainerFlow: HGDL_trainer

HGDL is a label distribution learning (LDL) problem on heterogeneous graphs. Unlike standard node classification, the labels are real-valued distributions over classes rather than discrete class indices.

The trainerflow performs full-batch training with:
- Loss: KL divergence between predicted and ground-truth label distributions, plus an Omega consistency regularizer
- Optimizer: Adam
- Early stopping: patience=50 on validation KL divergence
- Evaluation: 6 LDL metrics (KL, Cosine, Canberra, Chebyshev, Clark, Intersection)

### Model

HGDL consists of three modules:

1. **Module 1 — Metapath-based GCN**: Builds a merged metapath graph from dataset-provided adjacency lists (e.g., APA for DBLP), then applies a GraphConv layer for initial feature projection.

2. **Module 2 — Transformer Attention (HGDLAttnLayer)**: Graph transformer attention via DGL message passing (`apply_edges` / `edge_softmax` / `update_all`), following the same pattern as HGTConv. Attention output is gated by a frozen-zero residual to preserve the GCN convergence dynamics from the upstream reference.

3. **Module 3 — Output GCN + Softmax**: A second GraphConv layer followed by row-wise softmax to produce per-node label distributions.

### Dataset

Supported datasets: [dblp4HGDL](../../dataset/#DBLP), [acm4HGDL](../../dataset/#ACM)

The DBLP dataset is a heterogeneous graph with author (A), paper (P), conference (C), and term (T) nodes. Edge types include paper-author (P-A), paper-conference (P-C), and paper-term (P-T). The task is to predict the 4-class label distribution for each author node. The ACM dataset follows a similar structure with P-A and P-S (paper-subject) edges, predicting 14-class label distributions.

The dataset files are loaded from `openhgnn/dataset/DBLP/` and `openhgnn/dataset/ACM/` respectively. The preprocessing logic (port from the upstream HGDL repo) performs a 40/10/50 train/val/test split on author nodes with seeded randomness for bit-perfect reproducibility.

### Hyper-parameter specific to the model

```python
seed = 0
lr = 0.001
weight_decay = 0
max_epoch = 2000
patience = 50
hidden_dim = 64
dropout = 0.3
attention_dim = 5
num_heads = 2
gamma = 0.0001  # Omega regularizer weight (DBLP); set to 0 for ACM
mini_batch_flag = False
evaluate_interval = 1
```

Best config can be found in [config.ini](../../config.ini)

### Related API in DGL

[dgl.nn.GraphConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#graphconv)

[edge_softmax](https://docs.dgl.ai/en/latest/api/python/dgl.functional.html#edge-softmax)

[dgl.apply_edges / dgl.update_all](https://docs.dgl.ai/en/latest/api/python/dgl.html#dgl.apply_edges)

## More

#### Contributor

武雪妍 Xueyan Wu [BTBU]

#### If you have any questions,

Submit an issue or email to [2311020120@st.btbu.edu.cn](mailto:2311020120@st.btbu.edu.cn).
