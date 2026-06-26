# SE-HTGNN [NeurIPS 2025]

-   **Paper**: [Simple and Efficient Heterogeneous Temporal Graph Neural Network](https://doi.org/10.48550/arXiv.2510.18467)
-   **Code from author**: [SE-HTGNN](https://github.com/wangyili00/SE-HTGNN)

## How to run

  ```bash
  # For Node Classification (YELP)
  python main.py -m SEHTGNN -t node_classification -d sehtgnn_yelp -g 0 --use_best_config

  # For Node Regression (COVID-19)
  python main.py -m SEHTGNN -t node_regression -d sehtgnn_covid -g 0 --use_best_config

  # For Link Prediction (OGBN-MAG)
  python main.py -m SEHTGNN -t link_prediction -d sehtgnn_ogbn -g 0 --use_best_config

  # For Link Prediction (Aminer)
  python main.py -m SEHTGNN -t link_prediction -d sehtgnn_aminer -g 0 --use_best_config
  ```

If you do not have gpu, set -gpu -1.

## Performance

We compare the performance reported in the original paper with the reproduction results using this framework.

### Link Prediction

* **Device**: GPU, **GeForce RTX 3090**


* **Datasets**: OGBN-MAG, Aminer

* **Metrics**: AUC (Area Under Curve) and AP (Average Precision).

| Dataset | Metric | Paper Result (Reported) | Reproduction (Ours) |
| --- | --- | --- | --- |
| **OGBN-MAG** | **AUC** | 92.71% | 93.08% |
|  | **AP** | 93.13% | 94.21% |
| **Aminer** | **AUC** | 90.03% | 90.45% |
|  | **AP** | 91.08% | 90.82% |

### Node Classification

* **Dataset**: YELP
* **Metrics**: Macro-F1 and Recall.


| Dataset | Metric | Paper Result (Reported) | Reproduction (Ours) |
| --- | --- | --- | --- |
| **YELP** | **Macro-F1** | 44.24% | 42.27% |
|  | **Recall** | 44.68% | 43.11% |

### Node Regression

* **Dataset**: COVID-19
* **Metrics**: MAE (Mean Absolute Error) and RMSE (Root Mean Square Error).


| Dataset | Metric | Paper Result (Reported) | Reproduction (Ours) |
| --- | --- | --- | --- |
| **COVID-19** | **MAE** | 497 | 499 |
|  | **RMSE** | 1069 | 1064 |

## Dataset

We utilize the pre-processed heterogeneous temporal graph datasets described in the SE-HTGNN paper.

### Description

#### OGBN-MAG

* **Task**: Link Prediction (Author collaboration prediction).

* **Time Span**: 2010-2019 (Granularity: Year).

* **Nodes**: Author (17k), Paper (282k), Field (34k), Institution (2k).

* **Relations**: 4 types including *author-writes-paper*, *paper-cites-paper*, *author-affiliated-with-institution*, etc..

* **Snapshot**: 10 graph snapshots.

#### Aminer

* **Task**: Link Prediction (Predict whether a pair of authors will coauthor).

* **Time Span**: 1990-2005 (Granularity: Year).

* **Nodes**: Paper (18k), Author (23k), Venue (22).

* **Relations**: *paper-publish-venue*, *author-write-paper*.

* **Snapshot**: 16 graph snapshots.

#### YELP

* **Task**: Node Classification (Business Category: "American (New) Food", "Fast Food", "Sushi").

* **Time Span**: 01/2012 - 12/2021 (Granularity: Month).

* **Nodes**: User (55k), Business (12k).

* **Snapshot**: 12 graph snapshots.

#### COVID-19
* **Task**: Node Regression (Predict new daily cases).

* **Time Span**: 05/01/2020 - 02/28/2021 (Granularity: Day).

* **Nodes**: State (54), County (3223).

* **Relations**: *state-includes-county*, *state-near-state*, *county-near-county*.

* **Snapshot**: 304 graph snapshots.


## Model Details: SE-HTGNN

SE-HTGNN (Simple and Efficient Heterogeneous Temporal Graph Neural Network) proposes a novel learning paradigm to unify spatial and temporal modeling.

* **LLM-enhanced Prompt**
It uses Large Language Models (e.g., LLaMA3, GPT) to generate semantic representations for node types as prior knowledge.
These embeddings initialize the hidden states of the temporal module, enhancing the model's understanding of node type properties.

* **Simplified Spatial Aggregation**: Unlike traditional HGNNs using heavy node-level attention, SE-HTGNN employs a simplified neighbor aggregation (similar to GCN/Average) to reduce complexity, observing that intra-type neighbor variance is often low.

* **Dynamic-Attention-based Fusion**
* **Temporal-Spatial Unification**: Instead of decoupled steps, it integrates temporal modeling directly into the spatial fusion stage.

* **Mechanism**: It uses a GRU-based dynamic attention mechanism where historical attention coefficients guide the calculation of current attention weights for fusing different relations.


## Hyper-parameter specific to the model

You can modify the parameters in `openhgnn/config.ini` or pass them via command line.

## More

#### Contributor

Ruizhe Xie

#### If you have any questions,

Submit an issue or email to <cactus@bupt.edu.cn>.