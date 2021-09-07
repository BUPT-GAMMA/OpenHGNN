# HGNN_AC[WWW2021]

-   paper: [Heterogeneous Graph Neural Network via Attribute Completion](https://dl.acm.org/doi/10.1145/3442381.3449914)
-   Code from author: [HGNN_AC](https://github.com/liangchundong/HGNN-AC)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m MAGNN_AC -t node_classification_ac -d imdb4MAGNN -g 0
  ```

  If you do not have gpu, set -gpu -1.

  the dataset imdb4MAGNN is supported.

## Performance: Node classification

-   Device: CPU, **Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz**
-   Dataset: IMDB

| Node classification | Macro-F1 | Micro-F1  |
| ------------------- | -------  | --------  |
| MAGNN               | 58.65%   | 59.20%    |
| paper               | 60.75%   | 60.98%    |
| OpenHGNN            | 60.54%   | 60.70%    |

The perform of experiments are run in the setting of paper which use SVM classification, so there are a little bit different with semi-supervised node classification. And directly run the model is using semi-supervised node classification trainerflow.

## Dataset

-   We process the IMDB dataset given by [MAGNN](https://github.com/cynricfu/MAGNN). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

### Description

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

    -   Types of metapath: `MDM, MAM, DMD, DMAMD, AMA, AMDMA`. Please note that the `M` is movie, `D` is director, `A` is actor, and the edges above are all bidirectional.


[TODO]

## TrainerFlow: Node classification trainer


-   **Graph preprocess**
    -   To get some information from the original graph. It divides the nodes with attributes into feature keep list and feature drop list. And it uses the edge information to get the nodes' adjacency matrix related to the source nodes.
-   **Attribute Completion with Attention Mechanism**
    -   HGNN-AC adopts a masked attention mechanism which means we only calculate $e_{vu}$ for nodes $u\in{N_v^+}$, 
    where $u\in{N_v^+}$ denotes the first-order neighbors of node $v$ in set $V^+$, where $V^+$ is the set of nodes with attributes.
    -    Then, softmax function is applied to get normalized weighted coefficient $a_{vu}$.
    -    HGNN-AC can perform weighted aggregation of attributes for node $v$ according to weighted coefficient $a_{vu}$:
    -    Specially, the attention process is extended to a multi-head attention to stabilize the learning process and reduce the high variance

-   **Dropping some Attributes**
    -    HGNN-AC drops some attribute according to the feature drop list. It uses the Attribute Completion to get the new feature of the nodes and compares the new feature with the original features.

-   **Combination with the HIN Model**
    -    Finally, we can apply the proposed framework to MAGNN model. For more details, you can refer to MAGNN.

## Hyper-parameter specific to the model

You can modify the parameters in openhgnn/config.ini

### Description

```python
feats_drop_rate = 0.3 # feature drop rate to get the feature drop list
attn_vec_dim = 64 # the dimesion of vector in the Attention Layer 
feats_opt = 011 # the type of nodes that need to get the new features
loss_lambda = 0.2 # the weighted coefficient to balance the two parts.
src_node_type = 0 # the type of nodes that has the raw attribute
```

## More

#### Contirbutor

Yaoqi Liu[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to YaoqiLiu@bupt.edu.cn.

