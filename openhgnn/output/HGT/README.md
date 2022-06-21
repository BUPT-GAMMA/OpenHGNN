# Attention Network

| Model| Paper|
|:-----:|:-----:|
|HGT| [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332)|
|SimpleHGN|[Are we really making much progress? Revisiting, benchmarking,and refining heterogeneous graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3447548.3467350)|
|HetSANN|[An Attention-Based Graph Neural Network for Heterogeneous Structural Learning](https://arxiv.org/abs/1912.10832)|
|ieHGCN|[Interpretable and Efficient Heterogeneous Graph Convolutional Network](https://arxiv.org/pdf/2005.13183.pdf)|

## Attention mechanism
This part, we will give the definition of attention methanism based on **GAT** and **Transformer**.

- In [GAT](https://arxiv.org/abs/1710.10903), it defined the attentional mechanism. A shared linear transformation, parametrized by a weight matrix, $W\in\mathcal{R}^{F^{'}\times F}$, is applied to every node. Then use a shared attentional mechanism $a: \mathcal{R}^{F^{'}}\times \mathcal{R}^{F}\rightarrow \mathcal{R}$ to compute *attention coefficients*:

$$
e_{ij} = a(Wh_i, Wh_j)
$$

- this indicate the importance of node $j$'s features to node $i$. $a$ is a single-layer feedforward neural network. Finally we can normalize them across all choices of $j$ using the softmax function:

$$
\alpha_{ij} = softmax_j(e_{ij}) = \frac{\text{exp}(e_{ij})}{\sum_{k\in \mathcal{N}_i} \text{exp}(e_{ik})}
$$

- In [Transformer](https://arxiv.org/abs/1706.03762), an attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. e.g. Scaled Dot-Product Attention:

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

## Typical model

This part, we will give the definition some models.
### Direct-Aggregation models

| Model| Attention coefficient |
|:-----:|:-----:|
|HGT|$W_{Q_{\tau{(s)}}}h_s W^{ATT}_{\phi{(e)}}(W_{K_{\tau{(t)}}}h_t)^T$|
|SimpleHGN|$LeakyReLU(a^T[Wh_i \parallel Wh_j \parallel W_r r_{\psi(<i,j>)}])$|
|HetSANN|$LeakyReLU([W^{(l+1, m)}_{\phi(j),\phi(i)} h^{(l)}_i\parallel W^{(l+1, m)}_{\phi(j),\phi(i)} h^{(l)}_j]a^{(l+1, m)}_r)$|
#### HGT
- The model designed node-type and edge-type dependent parameters to characterize the heterogeneous attention over each edge, empowering
HGT to maintain dedicated representations for different types of nodes and edges.
- At each layer, Compute a multi-head attention score for each edge $(s, e, t)$ in the graph:

$$
Attention(s, e, t) = \text{Softmax}\left(||_{i\in[1,h]}ATT-head^i(s, e, t)\right) \\
ATT-head^i(s, e, t) = \left(K^i(s)W^{ATT}_{\phi(e)}Q^i(t)^{\top}\right)\cdot
\frac{\mu_{(\tau(s),\phi(e),\tau(t)}}{\sqrt{d}} \\
K^i(s) = \text{K-Linear}^i_{\tau(s)}(H^{(l-1)}[s]) \\
Q^i(t) = \text{Q-Linear}^i_{\tau(t)}(H^{(l-1)}[t]) \\
$$

- where $H^{(l)}$ is the feature in layer $l$, $s$ is the source node, $t$ is the target node, $e$ is edge, $d$ is the dimension of the feature, $\phi{(e)}$ is the type of the edge $e$, $\tau{(s)}$ is the type of the node $s$, and $\mu$ denote the general significance of each meta relation triplet, serving as an adaptive scaling to the attention.

- Compute the message to send on each edge $(s, e, t)$ :

$$
Message(s, e, t) = ||_{i\in[1, h]} MSG-head^i(s, e, t) \\
MSG-head^i(s, e, t) = \text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\phi(e)} \\
$$
- Send messages to target nodes $t$ and aggregate:

$$
\tilde{H}^{(l)}[t] = \sum_{\forall s\in \mathcal{N}(t)}\left( Attention(s,e,t)
\cdot Message(s,e,t)\right)
$$
- Compute new node features:
$$
H^{(l)}[t]=\text{A-Linear}_{\tau(t)}(\sigma(\tilde(H)^{(l)}[t])) + H^{(l-1)}[t]
$$

#### SimpleHGN

- The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.
- At each layer, we calculate the coefficient:

$$
\alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i\parallel Wh_j \parallel W_r r_{\psi(<i,j>)}]))}{\Sigma_{k \in \mathcal{E}}{exp(LeakyReLU(a^T[Wh_i \parallel Wh_k \parallel W_r r_{\psi(<i,k>)}]))}}
$$

- Residual connection including Node residual

$$
h_i^{(l)} = \sigma(\Sigma_{j\in \mathcal{N}_i} {\alpha_{ij}^{(l)}W^{(l)}h_j^{(l-1)}} + h_i^{(l-1)})
$$

- where $h_i$ and $h_j$ is the features of the source and the target node. $r_{\psi(e)}$ is a $d$-dimension embedding for each edge type $\psi(e) \in T_e$.

- and Edge residual:

$$
\alpha_{ij}^{(l)} = (1-\beta)\alpha_{ij}^{(l)}+\beta\alpha_{ij}^{(l-1)}
$$

#### HetSANN
- The model directly encodes structural information of HIN without meta-path and achieve more information representations. It models the transformation between heterogeneous vertices through a projection in low-dimensional entity spaces; and apply the GNN to aggregate multi-relational information of projected neighborhood by means of attention mechanism.
- At each layer, we apply a linear transformation, You may refer to the paper HetSANN-Section 2.1-Type-aware Attention Layer-(1):

$$
h^{(l+1, m)}_{\phi(j),i} = W^{(l+1, m)}_{\phi(j),\phi(i)} h^{(l)}_i
$$

- Computing the attention coefficient:

$$
o^{(l+1,m)}_e = \sigma(f^{(l+1,m)}_r(h^{(l+1, m)}_{\phi(j),j}, h^{(l+1, m)}_{\phi(j),i}))
$$
$$
f^{(l+1,m)}_r(e) = [h^{(l+1, m)^T}_{\phi(j),j}||h^{(l+1, m)^T}_{\phi(j),i}]a^{(l+1, m)}_r
$$
$$
\alpha^{(l+1,m)}_e = \frac{exp(o^{(l+1,m)}_e)}{\sum_{k\in \varepsilon_j} exp(o^{(l+1,m)}_k)}
$$

- where $h^{(l+1, m)}_{\phi(j),j}$ and $h^{(l+1, m)}_{\phi(j),i}$ is the features of the source and the target node after projection.

- Getting new embeddings with multi-head and residual:

$$
h^{(l + 1, m)}_j = \sigma(\sum_{e = (i,j,r)\in \varepsilon_j} \alpha^{(l+1,m)}_e h^{(l+1, m)}_{\phi(j),i})
$$

- Multi-head:
$$
h^{(l+1)}_j = \parallel^M_{m = 1}h^{(l + 1, m)}_j
$$

- Residual:
$$
h^{(l+1)}_j = h^{(l)}_j + \parallel^M_{m = 1}h^{(l + 1, m)}_j
$$

### Dual-aggregation model

#### ieHGCN

- This model first reduces node features within the same relation, which generate type-specific features in relation subgraphs:

$$
Z^{ Self - \Omega } = Y^{ Self - \Omega}=H^{\Omega} \cdot W^{Self - \Omega} \\
Z^{\Gamma - \Omega}=\hat{A}^{\Omega-\Gamma} \cdot Y^{\Gamma - \Omega} = \hat{A}^{\Omega-\Gamma} \cdot H^{\Gamma} \cdot W^{\Gamma - \Omega}
$$

- The next step is to reduce type-specific features across different relations using attention mechanism:

$$
Q^{\Omega}=Z^{Self-\Omega} \cdot W_q^{\Omega}\\
K^{Self-\Omega}=Z^{Self -\Omega} \cdot W_{k}^{\Omega}\\
K^{\Gamma - \Omega}=Z^{\Gamma - \Omega} \cdot W_{k}^{\Omega}, \quad \Gamma \in N_{\Omega}
$$
- computing attention function
$$
e^{Self-\Omega}={ELU} ([K^{ Self-\Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega})\\
e^{\Gamma - \Omega}={ELU} ([K^{\Gamma - \Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}), \Gamma \in N_{\Omega}
$$
- attention coefficients
$$
[a^{Self-\Omega}\|a^{1 - \Omega}\| \ldots . a^{\Gamma - \Omega}\|\ldots\| a^{|N_{\Omega}| - \Omega}]=
            {softmax}([e^{Self - \Omega}\|e^{1 - \Omega}\| \ldots\|e^{\Gamma - \Omega}\| \ldots \| e^{|\N_{\Omega}| - \Omega}])
$$

- output embeddings

$$
H_{i,:}^{\Omega \prime}=\sigma(a_{i}^{Self-\Omega} \cdot Z_{i,:}^{Self-\Omega}+\sum_{\Gamma \in N_{\Omega}} a_{i}^{\Gamma - \Omega} \cdot Z_{i,:}^{\Gamma - \Omega})
$$

## Implement Details

### Direct-Aggregation models
- We first implement the convolution layer for the model SimpleHGN, and HetSANN. The convolutional layer of HGT we use is [hgtconv](https://docs.dgl.ai/generated/dgl.nn.pytorch.conv.HGTConv.html?highlight=hgtconv#dgl.nn.pytorch.conv.HGTConv). The **\_\_init\_\_** parameters can be different as the models need different parameters. The parameters of the **forward** part are the same: `g` is the homogeneous graph, `h` is the features, `ntype` denotes the type of each node, `etype` denotes the type of each edge, `presorted` tells if the `ntype` or `etype` is presorted to use [TypedLinear](https://docs.dgl.ai/generated/dgl.nn.pytorch.TypedLinear.html) in **dgl.nn** conveniently. If we use [dgl.to_homogeneous](https://docs.dgl.ai/generated/dgl.to_homogeneous.html?highlight=to_homogeneous#dgl.to_homogeneous) to get the features, the features are presorted.

- Then, we use the convolution layers to implement coresponding models. We need [dgl.to_homogeneous](https://docs.dgl.ai/generated/dgl.to_homogeneous.html?highlight=to_homogeneous#dgl.to_homogeneous) to get a homogeneous graph as when we use [edge_softmax](https://docs.dgl.ai/generated/dgl.nn.functional.edge_softmax.html?highlight=edge_softmax), we put all the edges together to calculate the attention coefficient instead of distinguishing the type of edges. 
- After passing the convolution layers, we need to convert the output features to a feature dictionary in a heterogeneous graph. We design a tool in **openhgnn.utils.utils.py** named **to_hetero_feat**. This is because we do not have a better solution to get a feature dictionay using **dgl**. We can only use [dgl.to_heterogeneous](https://docs.dgl.ai/generated/dgl.to_heterogeneous.html), but it has many additional operations to make the programs slowly. After we get a feature dictionary, the model is complete.

### Dual-Aggregation model

- We refer to the idea of the implementation of [dgl.nn.HeteroGraphConv](https://docs.dgl.ai/generated/dgl.nn.pytorch.HeteroGraphConv.html?highlight=heterographconv#dgl.nn.pytorch.HeteroGraphConv). We extract the relationship subgraph based on the edge type and complete the aggregation using the convoluntion layers. Then, to aggregate type-specific features across different relations we have to compute attention coefficients step by step.

## How to run

- Clone the Openhgnn-DGL

  ```bash
  # For node classification task
  # You may select model HGT, SimpleHGN, HetSANN
  python main.py -m HGT -t node_classification -d imdb4MAGNN -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

## Performance

#### Task: Node classification

Evaluation metric: accuracy

- HGT

| Dataset   | HGBn-ACM  | acm4GTN | imdb4MAGNN | dblp4MAGNN |
| -------- | ----- | ----- | ----- | ----- |
| Macro_f1 | 89.18 | 90.24 | 49.18 | 85.35 |
| Micro_f1 | 88.95 | 90.21 | 49.37 | 87.20 |

- SimpleHGN

| Dataset   | HGBn-ACM  | acm4GTN | imdb4MAGNN | dblp4MAGNN |
| -------- | ----- | ----- | ----- | ----- |
| Macro_f1 | 66.64 | 89.28 | 48.78 | 86.79 |
| Micro_f1 | 88.40 | 89.27 | 52.25 | 86.75 |

- HetSANN

| Dataset   | HGBn-ACM  | acm4GTN | imdb4MAGNN | dblp4MAGNN |
| -------- | ----- | ----- | ----- | ----- |
| Macro_f1 | 84.63 | 92.31 | 47.44 | 84.11 |
| Micro_f1 | 84.53 | 92.24 | 52.88 | 84.96 |

- ieHGCN


| Dataset   | HGBn-ACM  | acm4GTN | imdb4MAGNN | dblp4MAGNN |
| -------- | ----- | ----- | ----- | ----- |
| Macro_f1 | 90.31 | 92.56 | 52.18 | 87.37 |
| Micro_f1 | 90.06 | 92.47 | 55.03 | 88.36 ||

## TrainerFlow: [node classification flow](../../trainerflow/#Node_classification_flow)

## Hyper-parameter specific to the model

You can modify the parameters [HGT], [SimpleHGN], [HetSANN], [ieHGCN] in openhgnn/config.ini. 
## More

#### Contirbutor

Yaoqi Liu[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [YaoqiLiu@bupt.edu.cn](mailto:YaoqiLiu@bupt.edu.cn).
