# DMGI

Paper:[Unsupervised Attributed Multiplex Network Embedding](https://doi.org/10.1609/aaai.v34i04.5985/)

Code from author:[https://github.com/pcy1302/DMGI](https://github.com/pcy1302/DMGI)

## How to run

Clone the Openhgnn-DGL

```python
python main.py -m DMGI -t node_classification -d acm_han_raw -g 0 --use_best_config
```

Candidate dataset: acm_han_raw

If you do not have gpu, set -gpu -1.

## candidate dataset

acm_han_raw

NOTE: DMGI can handle imdb dataset, we will add the dataset in our further work.

### description

- Number of nodes

  | paper    | 4025 |
  | -------- | ---- |
  | author    | 17431 |
  | field   | 73 |
  
-   Number of edges

    | paper-author | 13407  |
    | -------------- | ----- |
    | paper-field    | 4025 |
    
-   Subsets: paper-author, paper-field


## performance

Node classification

| |accuracy|
|----|----|
|DMGI|0.895 ± 0.003|


## TrainerFlow: DMGI_trainer




## Hyper-parameter specific to the model

```python
learning_rate = 0.0005
l2_coef = 0.0001
sc = 3
dropout = 0.5
reg_coef = 0.001
sup_coef = 0.1

patience =20
hid_unit = 64
num_heads = 1
isSemi = False
isBias = False
isAttn = False
```



## More

#### Contirbutor

Tianyu Zhao, Siyuan Zhang[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [1580124318@qq.com](mailto:1580124318@qq.com).
