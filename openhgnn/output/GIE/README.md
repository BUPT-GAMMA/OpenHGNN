# GIE[AAAI 2022]

- paper: [Geometry interaction knowledge graph embeddings](https://ojs.aaai.org/index.php/AAAI/article/view/20491/20250)
- Code from author: [https://github.com/Lion-ZS/GIE](https://github.com/Lion-ZS/GIE)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  # For link prediction task
  python main.py -m GIE -t link_prediction -d wn18 -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  ### Supported dataset

  - [FB15k](../../dataset/#FB15k)

    - Number of entities and relations

        | entities | relations |
        | -------- | --------- |
        | 14,951   |    1,345  |

    - Size of dataset

        | set type       | size   |
        | -------------- | -----  |
        | train set      | 483,142|
        | validation set | 50,000 |
        | test set       | 59,071 |

  - [WN18](../../dataset/#WN18)

    - Number of entities and relations

        | entities | relations |
        | -------- | --------- |
        | 40,493   |    18     |

    - Size of dataset

        | set type       | size    |
        | -------------- | -----   |
        | train set      | 141,442 |
        | validation set | 5,000   |
        | test set       | 5,000   |

## Performance

### Task: Link Prediction

Evaluation metric: mrr

| Dataset      | Mean Rank | Hits@10 |
|--------------|-----------|---------|
| FB15k(raw)   | 268       | 34.7    |
| FB15k(filt.) | 140       | 46.2    |
| WN18(raw)    | 635       | 69.3    |
| WN18(filt.)  | 675       | 73.2    |



## TrainerFlow: [TransX flow](../../trainerflow/#TransX_flow)

## Hyper-parameter specific to the model

You can modify the parameters[GIE] in openhgnn/config.ini
