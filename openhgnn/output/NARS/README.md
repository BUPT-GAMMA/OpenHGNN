# NARS

Paper:[Scalable Graph Neural Networks for Heterogeneous Graphs](https://arxiv.org/abs/2011.09679)

Code from author:[https://github.com/facebookresearch/NARS](https://github.com/facebookresearch/NARS)

## How to run

## candidate dataset

acm4NARS

mag4NARS

## performance

Node classification

## TrainerFlow: node_classification

### model

NARS

WeightedAggregator

SIGN



### Hyper-parameter specific to the model

```python
R = 2
input_dropout = True
cpu_preprocess = True
ff_layer = 2
```

Best config can be found in [best_config](file:/E:/资料/科研/复现论文/OpenHGNN-main/openhgnn/utils/best_config.py)

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).