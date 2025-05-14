# HERO

Paper:[Self-Supervised Heterogeneous Graph Learning](https://openreview.net/forum?id=3FJOKjooIj)
## How to run

Clone the Openhgnn-DGL

```python
python main.py -m HERO -t node_classification -d acm4GTN -g 0 --use_best_config
```

Candidate dataset: acm4GTN

If you do not have gpu, set -gpu -1.

## candidate dataset

acm4GTN

NOTE: HERO can handle mag oag dataset, we will add these two datasets in our further work.

### description

- Number of nodes

  | paper    | 3025 |
  | -------- | ---- |
  | author    | 5912 |
  | subject   | 57 |
    
-   Subsets: paper-author, paper-subject


## performance

Node classification

| |f1_marco|f1_micro|
|----|----|----|
|acm4GTN|0.917|0.917|


## TrainerFlow: herotrainer

## Hyper-parameter specific to the model

```python
lr = 0.0025
patience = 60
nb_epochs = 500
hid_units = 256
hid_units2 = 128
out_ft = 256
g_dim = 64
g_equidim = 256
p_equidim = 256
edge_rate = 0.9
alpha = 100
beta = 500
gamma = 10
eta = 1
lambbda = 1
gpu = 1
mode = 0
save_dir = './saved_models'
```


## More

#### Contirbutor

Tianyi Wang[Gamma LAB]
![My Page](www.bimu.site)

#### If you have any questions,

Submit an issue or email to [bimu@bupt.edu.cn](mailto:bimu@bupt.edu.cn).