# SimpleHGN[KDD 2021]

-   paper: [Are we really making much progress? Revisiting, benchmarking,and refining heterogeneous graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3447548.3467350)

## Basic Idea

- The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.
- At each layer, we calculate the coefficient:

$$
\alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i||Wh_j||W_r r_{\psi(<i,j>)}]))}{\Sigma_{k\in\mathcal{E}}{exp(LeakyReLU(a^T[Wh_i||Wh_k||W_r r_{\psi(<i,k>)}]))}}

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

- Finally, a multi-head attention is used.

## How to run

- Clone the Openhgnn-DGL

  ```bash
  # For node classification task
  python main.py -m SimpleHGN -t node_classification -d imdb4MAGNN -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  ##### Supported dataset
  -   [imdb4MAGNN](../../dataset/#IMDB)

    -   Number of nodes

        | movie    | 4278 |
        | -------- | ---- |
        | director | 2081 |
        | actor    | 5257 |

    -   Number of edges

        | movie-director | 4278  |
        | -------------- | ----- |
        | movie-actor    | 12828 |

-   [dblp4MAGNN](../../dataset/#DBLP)

    -   Number of nodes

        | author | 4057  |
        | ------ | ----- |
        | paper  | 14328 |
        | term   | 7723  |
        | venue  | 20    |

    -   Number of edges

        | author-paper | 19645 |
        | ------------ | ----- |
        | paper-term   | 85810 |
        | paper-venue  | 14328 |

## Performance

#### Task: Node classification

Evaluation metric: accuracy

| Dataset   | HGBn-ACM  | HGBn-DBLP | imdb4MAGNN | dblp4MAGNN |
| -------- | ----- | ----- | ----- | ----- |
| Macro_f1 | 66.64 | 86.31 | 48.78 | 86.79 |
| Micro_f1 | 88.40 | 87.24 | 52.25 | 86.75 |



## TrainerFlow: [node classification flow](../../trainerflow/#Node_classification_flow)

## Hyper-parameter specific to the model

You can modify the parameters[SimpleHGN] in openhgnn/config.ini. 

## More

#### Contirbutor

Yaoqi Liu[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [YaoqiLiu@bupt.edu.cn](mailto:YaoqiLiu@bupt.edu.cn).