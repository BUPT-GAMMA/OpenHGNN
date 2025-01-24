# SACN

Paper:[End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://arxiv.org/pdf/1811.04441.pdf)

## How to run

* Clone the Openhgnn-DGL

```bash
python main.py -m SACN -d SACN -t link_prediction -g 0 
```

If you do not have gpu, set -gpu -1.

## Performance

| Dataset | FB15k-237                  |
|---------|----------------------------|
| MRR     | Paper:0.35 OpenHGNN:0.3528 |
| H@1     | Paper:0.26 OpenHGNN:0.2575 |
| H@3     | Paper:0.39 OpenHGNN:0.3938 |
| H@10    | Paper:0.54 OpenHGNN:0.5397 |

### TrainerFlow

```SACN_trainer```

### model

```SACN```

### Dataset

SACN_dataset


#### Contributor

Zikai Zhou[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to 460813395@qq.com