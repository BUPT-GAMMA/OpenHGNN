# KGCN[WWW2020]

-   paper: [Knowledge Graph Convolutional Networks for Recommender Systems ](https://dl.acm.org/doi/10.1145/3308558.3313417)
-   Code from author: [KGCN](https://github.com/hwwang55/KGCN)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m KGCN -d LastFM4KGCN -t recommendation -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  the dataset Last.FM is supported.

## Performance: Recommendation

-   Device: GPU, **GeForce RTX 3090**
-   Dataset: Last.FM

| Recommendation |               AUC               |               F1                |
| :------------: | :-----------------------------: | :-----------------------------: |
|    KGCN-sum    | paper: 79.4%    OpenHGNN: 79.6% | paper: 71.9%    OpenHGNN: 71.8% |
|  KGCN-concat   | paper: 79.6%    OpenHGNN: 78.9% | paper: 72.1%    OpenHGNN: 71.4% |
| KGCN-neighbor  | paper: 78.1%    OpenHGNN: 78.6% | paper: 69.9%    OpenHGNN: 71.0% |

## Dataset

-   We process the KGCN dataset given by [KGCN](https://github.com/hwwang55/KGCN). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

### Description

- Last.FM

  |              | Last.FM |
  | :----------: | :-----: |
  |     User     |  1872   |
  |     item     |  3846   |
  | interactions |  42346  |
  |   entities   |  9366   |
  |  relations   |   60    |
  |  KG triples  |  15518  |

  

## TrainerFlow: Recommendation

#### model

- ​	KGCN
  - ​		KGCN is to aggregate the entity representation and its neighborhood representation



## More

#### Contributor

Yanhu Mo[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  [myth@bupt.edu.cn](mailto:myth@bupt.edu.cn).



