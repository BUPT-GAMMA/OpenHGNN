# HTGformer

- paper: [HTGformer: Heterogeneous Temporal Graph Transformer](https://doi.org/10.1145/3726302.3730209)

- venue: SIGIR 2025 (Short Paper)

- code from author: Not publicly available

## How to run

- Clone the OpenHGNN

  ```bash
  # OGBN-MAG link prediction
  python main.py -m HTGformer -d ogbn_mag4HGformer -t link_prediction -g 0

  # Aminer link prediction
  python main.py -m HTGformer -d aminer4HGformer -t link_prediction -g 0

  # YELP node classification
  python main.py -m HTGformer -d yelp4HGformer -t node_classification -g 0

  # COVID-19 node regression
  python main.py -m HTGformer -d covid4HGformer -t node_regression -g 0
  ```

  The trainer also accepts the short aliases `ogbn_mag`, `aminer`, `yelp`, and `covid`.

## Dataset

Datasets are registered in OpenHGNN and are loaded through `build_dataset`.
The required files are downloaded automatically to `openhgnn/dataset/` on first use:

| Dataset name       | Task                | Downloaded files                          |
|--------------------|---------------------|-------------------------------------------|
| ogbn_mag4HGformer  | link_prediction     | ogbn4HGformer.bin, mp2vec.zip             |
| aminer4HGformer    | link_prediction     | aminer4HGformer.pt                        |
| yelp4HGformer      | node_classification | yelp4HGformer.pt                          |
| covid4HGformer     | node_regression     | covid4HGformer.bin                        |

## Performance

- Device: GPU, **NVIDIA GeForce RTX 4090**
- Mode: w/o_LLM (using learnable embeddings instead of LLama3 type encoding)

### Link Prediction

| Dataset  |               AUC                |               AP                 |
|:--------:|:--------------------------------:|:--------------------------------:|
| OGBN-MAG | paper: 92.56%  OpenHGNN: 94.61% | paper: 91.64%  OpenHGNN: 93.98% |
|  Aminer  | paper: 89.78%  OpenHGNN: 88.41% | paper: 88.03%  OpenHGNN: 82.99% |

### Node Classification

| Dataset |              Macro-F1              |              Recall              |
|:-------:|:----------------------------------:|:--------------------------------:|
|  YELP   | paper: 43.24%  OpenHGNN: 35.91%   | paper: 43.86%  OpenHGNN: 40.91% |

### Node Regression

| Dataset  |              MAE               |
|:--------:|:------------------------------:|
| COVID-19 | paper: 532  OpenHGNN: 511.59   |

### Note

- **Aminer** AUC gap is within 1.5% of the paper. The AP gap is mainly due to the w/o_LLM mode and negative sampling strategy differences.
- **YELP** is reproduced in w/o_LLM mode (using learnable embeddings instead of LLama3). The paper's ablation study (Figure 3) shows that the w/o_LLM variant achieves approximately 36% Macro-F1 on YELP, which is consistent with our reproduction result of 35.91%. The full HTGformer with LLama3 encoding achieves 43.24% as reported. The performance gap is entirely attributable to the LLM type encoding component, not the model architecture.

## More

#### Contributor

NancyLis1 [GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [NancyLis1](https://github.com/NancyLis1).
