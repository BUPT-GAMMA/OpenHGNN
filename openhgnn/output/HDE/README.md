# HDE[ICDM2021]

Paper:[Heterogeneous Graph Neural Network with Distance Encoding](http://www.shichuan.org/doc/116.pdf)

## How to run

* Clone the Openhgnn-DGL

```bash
python main.py -m HDE -d HGBl-IMDB -t link_prediction -g 0 --use_best_config
```

If you do not have gpu, set -gpu -1.

## Performance

| Dataset   | AUC_ROC |
| --------- | ------- |
| HGBl-IMDB | 0.9151  |
| HGBl-ACM  | 0.8741  |
| HGBl-DBLP | 0.9836  |

### TrainerFlow

```hde_trainer```

### model

```HDE```

### Dataset

Supported datasets:

* HGBl-IMDB
* HGBl-ACM
* HGBl-DBLP

### Hyper-parameter specific to the model

```python
emb_dim = 128  # dimension of HDE
k_hop = 2  # radius when sampling ego graph for a node
max_dist = 3  # max value of SPD, usually set to k_hop + 1
```

## More

#### Contributor

Zhiyuan Lu[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [luzy@bupt.edu.cn](luzy@bupt.edu.cn)