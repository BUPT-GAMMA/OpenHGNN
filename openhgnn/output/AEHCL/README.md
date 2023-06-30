# AEHCL[SDM 2023]

Paper: [**Abnormal Event Detection via Hypergraph Contrastive Learning
**](http://www.shichuan.org/doc/145.pdf)

Code: https://github.com/BUPT-GAMMA/AEHCL

### How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m AEHCL -t abnorm_event_detection -d aminer -g 0
  ```
  
  If you do not have gpu, set -g -1.

## Performance

#### Task: Abnormal Event Detection

Evaluation metric: AUC

| Method    | Aminer |
|-----------|--------|
| **AEHCL** | 88.54  |

Evaluation metric: AP

| Method    | Aminer |
|-----------|--------|
| **AEHCL** | 50.81  |

## TrainerFlow: [abnormal event detection](../../trainerflow/AbnormEventDetection.py)

### Model

- [AEHCL](https://github.com/BUPT-GAMMA/AEHCL/tree/main)

## Hyper-parameter specific to the model

  You can modify the parameters in openhgnn/config.ini

#### If you have any questions,

  Submit an issue or email to [1287581579@qq.com](mailto:1287581579@qq.com).