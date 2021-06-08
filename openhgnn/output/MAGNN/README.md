# MAGNN[WWW2020]

-   paper: [Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding](https://arxiv.org/abs/2002.01680)
-   Code from author: [MAGNN](https://github.com/cynricfu/MAGNN)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m MAGNN -t node_classification -d imdb4MAGNN -g 0
  ```

  If you do not have gpu, set -gpu -1.

  the dataset imdb4MAGNN/dblp4MAGNN is supported.

## Performance: Node classification

-   Device: GPU, **GeForce RTX 3090**
-   Dataset: IMDb
-   The average of 5 experiments.

| Node classification | Macro-F1                           | Micro-F1                           |
| ------------------- | ---------------------------------- | ---------------------------------- |
| 20%                 | paper: 59.35%    OpenHGNN:  61.52% | paper: 59.60%    OpenHGNN:  61.45% |
| 40%                 | paper: 60.27%    OpenHGNN:  62.65% | paper: 60.50%    OpenHGNN:  62.59% |
| 60%                 | paper: 60.66%    OpenHGNN:  63.37% | paper: 60.88%    OpenHGNN:  63.26% |
| 80%                 | paper: 61.44%    OpenHGNN:  63.86% | paper: 61.53%    OpenHGNN:  63.81% |

-   Device: CPU, **Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz**
-   Dataset: IMDb
-   The average of 5 experiments.

| Node classification | Macro-F1                           | Micro-F1                           |
| ------------------- | ---------------------------------- | ---------------------------------- |
| 20%                 | paper: 59.35%    OpenHGNN:  53.91% | paper: 59.60%    OpenHGNN:  53.96% |
| 40%                 | paper: 60.27%    OpenHGNN:  55.75% | paper: 60.50%    OpenHGNN:  55.81% |
| 60%                 | paper: 60.66%    OpenHGNN:  56.83% | paper: 60.88%    OpenHGNN:  56.80% |
| 80%                 | paper: 61.44%    OpenHGNN:  57.65% | paper: 61.53%    OpenHGNN:  57.68% |

The perform of experiments are run in the setting of paper which use SVM classification, so there are a little bit different with semi-supervised node classification. And directly run the model is using semi-supervised node classification trainerflow.

## Dataset

-   We process the IMDB dataset and DBLP given by [MAGNN](https://github.com/cynricfu/MAGNN). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

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

    -   Types of metapath: `APA, APTPA, APVPA`. Please note that the `A` is actor, `P` is paper, `T` is term, `V` is venue, and the edges above are all bidirectional.

[TODO]

## TrainerFlow: Node classification trainer

-   **Node content transformation**
    -   It applies different type-specific linear transformations on different types of ndoes to project their node features into the same vector space, thus all nodes' features share the same dimension after the transformation. This part addresses the heterogeneity of the original heterogeneous graph.
    -   **Please note that** the dimension of $h_v'$ in original author's code is **h_dim** which is the dimension of node features in hidden layers. However we set the dimension of $h_v'$ as **h_dim \* num_heads** where num_heads is the number of heads in attention mechanism of **intra-metapath aggregation**. We'll why we make this change in the following part.
-   **Intra-metapath aggregation**
    -   **intra_metapath_trans() in openhgnn/models/MAGNN.py**
    -   Firstly, we employ a metapath instance encoder to transform all the node features along a metapath instance into a single vector. **encoder() in openhgnn/models/MAGNN.py**
        -   **complex_hada() in openhgnn/models/MAGNN.py** implements the complext hadamard product used in **RotateE encoder**. Besides **RotateE**, we alse implement **Linear and Average Encoder** mentioned in paper.
    -   Next is to employ type-specific graph attention layer on different types of nodes with multi-heads. **MAGNN_attn_intra() in openhgnn/models/MAGNN.py**
        -   As to multi-heads in **intra-metapath aggregation**, author simply repeats the feature matrix after **RotateE** and concatenate them together on which employ graph attention mechanism, that is, the feature matrices across different heads are **all the same** when employ attention on them. We think it's a little counterintuitive so that we introduce multi-heads in **Node content transformation**, that is, the output dimension of **Node content transformation** will be **h_dim * num_heads** resulting in different feature matrices across different heads. 
        -   **Please note that** this change will also introduce multi-heads in **encoder**. 
-   **Inter-metapath aggregation**
    -   This part employs attention mechanism on graph  to aggregate the features of different types of metapath. Consequently we obtain the node embeddings. **inter_metapath_trans() in openhgnn/models/MAGNN.py**
    -   Then project the node embeddings to the vector space with desired output dimension. **output_projection() in openhgnn/models/MAGNN.py**

## Hyper-parameter specific to the model

You can modify the parameters in openhgnn/config.ini

### Description

```python
inter_attn_feats = 32 # the dimension of the attention vector in inter metapath 						
num_heads = 4 # the number of heads in the attention mechanism of intra metapath 		 			   
num_layers = 2 # the number of layers.
```

Best config can be found in [best_config](../../utils/best_config.py)

## More

#### Contirbutor

Jiahang Li[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to ljh1064126026@bupt.edu.cn.

