# HGT[WWW 2020]

-   paper: [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332)

## Basic Idea

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

## How to run

- Clone the Openhgnn-DGL

  ```bash
  # For node classification task
  python main.py -m HGT -t node_classification -d imdb4MAGNN -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

## Performance

#### Task: Node classification

Evaluation metric: accuracy

| Dataset   | HGBn-ACM  | acm4GTN | imdb4MAGNN | dblp4MAGNN |
| -------- | ----- | ----- | ----- | ----- |
| Macro_f1 | 89.18 | 90.24 | 49.18 | 85.35 |
| Micro_f1 | 88.95 | 90.21 | 49.37 | 87.20 |



## TrainerFlow: [node classification flow](../../trainerflow/#Node_classification_flow)

## Hyper-parameter specific to the model

You can modify the parameters[HGT] in openhgnn/config.ini. 

## More

#### Contirbutor

Yaoqi Liu[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [YaoqiLiu@bupt.edu.cn](mailto:YaoqiLiu@bupt.edu.cn).
