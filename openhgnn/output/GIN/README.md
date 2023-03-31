# GIN[ICLR 2019]

-   Paper: [HOW POWERFUL ARE GRAPH NEURAL NETWORKS?](https://arxiv.org/pdf/1810.00826.pdf)
-   Author's code for entity classification: https://github.com/weihua916/powerful-gnns
-   Code from dgl Team: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin
-   Note: The difference between GIN and GIN-hetero is the storage form of node/edge type. The former use the dgl stored in g.ndata, but the latter use the HeteroNodeView/HeteroEdgeView. And we use the latter storage form and nearly all models in OpenHGNN use the latter storage form as possible.



## How to run

* Clone the Openhgnn-DGL

```bash
python main.py -m GIN -d imdb4GTN -t node_classification -g 0
```

If you do not have gpu, set -gpu -1.


### Performance

Node classification

| Node classification(F1 score) | imdb4GTN                  | MUTAG |
| ----------------------------- | ------------------------- | ----- |
| OpenHGNN[GIN]                 | Macro: 54.85 Micro: 57.93 | 66.18 |

### Formula

GIN updates node representations as
$$
h_{v}^{(k)}=MLP^{(k)}(\sum_{r\in R}\sum_{u\in N^{r}(v)}\frac{1}{c_{u,r}}W_{r}^{(k-1)}h_{u}^{(k-1)}+(1+\epsilon^{(k)})·h_{v}^{(k-1)})\notag
$$




