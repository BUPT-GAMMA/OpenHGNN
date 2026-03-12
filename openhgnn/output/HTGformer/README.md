# HTGformer

- paper: [HTGformer: Heterogeneous Temporal Graph Transformer](https://doi.org/10.1145/3726302.3730209)

- venue: SIGIR 2025 (Short Paper)

- code from author: Not publicly available

## How to run

- Clone the OpenHGNN

  ```bash
  # OGBN-MAG link prediction
  python main.py -m HTGformer -d ogbn_mag -t link_prediction -g 0

  # Aminer link prediction
  python main.py -m HTGformer -d aminer -t link_prediction -g 0

  # YELP node classification
  python main.py -m HTGformer -d yelp -t node_classification -g 0
  ```

## Dataset

Datasets should be downloaded manually and placed in `openhgnn/dataset/data/`:

| Dataset  | File                                    | Source                                                       |
|----------|-----------------------------------------|--------------------------------------------------------------|
| OGBN-MAG | ogbn_graphs.bin + mp2vec/g0~g9.vector   | [HTGNN](https://github.com/yeslab-code/HTGNN)               |
| Aminer   | aminer_processed.pt                     | [DHGAS](https://github.com/wondergo2017/DHGAS) Google Drive  |
| YELP     | yelp_processed.pt                       | [DHGAS](https://github.com/wondergo2017/DHGAS) Google Drive  |
| COVID-19 | covid_graphs.bin                        | [HTGNN](https://github.com/yeslab-code/HTGNN)               |

## Performance

- Device: GPU, **NVIDIA GeForce RTX 4090**
- Mode: w/o_LLM (using learnable embeddings instead of LLama3 type encoding)

### Link Prediction

| Dataset  |               AUC                |               AP                 |
|:--------:|:--------------------------------:|:--------------------------------:|
| OGBN-MAG | paper: 92.56%  OpenHGNN: 94.61% | paper: 91.64%  OpenHGNN: 93.98% |
|  Aminer  | paper: 89.78%  OpenHGNN: 85.98% | paper: 88.03%  OpenHGNN: 80.83% |

### Node Classification

| Dataset |              Macro-F1              |              Recall              |
|:-------:|:----------------------------------:|:--------------------------------:|
|  YELP   | paper: 43.24%  OpenHGNN: 35.91%   | paper: 43.86%  OpenHGNN: 40.91% |

### Node Regression

| Dataset  |              MAE               |
|:--------:|:------------------------------:|
| COVID-19 | paper: 532  OpenHGNN: 511.59   |

### Note

The performance gap on Aminer and YELP is mainly due to the w/o_LLM mode. The paper uses LLama3 for node type encoding, while this reproduction uses learnable embeddings. The paper's ablation study (Figure 3) shows that w/o_LLM leads to noticeable performance degradation. OGBN-MAG results surpass the paper's reported values.

## More

#### Contributor

NancyLis1 [GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [NancyLis1](https://github.com/NancyLis1).
