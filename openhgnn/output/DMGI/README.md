# DMGI[AAAI2020]

Paper:[Unsupervised Attributed Multiplex Network Embedding](https://ojs.aaai.org//index.php/AAAI/article/view/5985)

Code from author:[https://github.com/pcy1302/DMGI](https://github.com/pcy1302/DMGI)

## How to run

Clone the Openhgnn-DGL

```python
python main.py -m DMGI -t node_classification -d acm_han_raw -g 0 --use_best_config
```

Candidate dataset: acm_han_raw

If you do not have gpu, set -gpu -1.

## candidate dataset

acm_han_raw/imdb4GTN

NOTE: DMGI can handle imdb dataset, we will add the dataset in our further work.

### Performance

Node classification 

| Node classification | acm   | imdb4GTN |
| ------------------- | ----- | -------- |
| paper               | 89.8  | ---      |
| OpenHGNN            | 89.73 | 52.52    |

### TrainerFlow: DMGI_trainer

The model is  trained in unsupervisied node classification.

### Hyper-parameter specific to the model

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
max_epoch = 10000
isSemi = False
isBias = False
isAttn = False
```

Best config can be found in  [best_config](../../utils/best_config.py)

### Related API in DGL

[dgl.add_self_loop](https://docs.dgl.ai/generated/dgl.add_self_loop.html?highlight=add_self#dgl.add_self_loop)

[GraphConv](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html?#graphconv)

## More

#### Contirbutor

Siyuan Zhang, Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [1580124318@qq.com](mailto:1580124318@qq.com).
