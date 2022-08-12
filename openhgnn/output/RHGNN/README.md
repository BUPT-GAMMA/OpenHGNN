# RHGNN

Paper:[Heterogeneous Graph Representation Learning with Relation Awareness](https://arxiv.org/abs/2105.11122)

Code from author:[https://github.com/yule-BUAA/R-HGNN](https://github.com/yule-BUAA/R-HGNN)

## How to run

Clone the Openhgnn-DGL

```bash
python main.py -m RHGNN -t node_classification -d imdb4GTN -g 0 --use_best_config
```

Candidate dataset: imdb4GTN/acm4GTN

If you do not have gpu, set -gpu -1.

## candidate dataset

imdb4GTN

NOTE: imdb4GTN is a small-scale dataset. We will add large-scale datasets (OGB-MAG, OAG-Venue and OAG-L1-Field) in our further work.


## performance

Node classification

| |accuracy|
|----|----|
|imdb4GTN|0.5883|


## TrainerFlow: node_classification

### model


- [RHGNN](../../utils/utils.py)
    - Contain two R_HGNN_layer、 relation_fusing and one classifier  

- [R_HGNN_layer](../../models/RHGNN.py)
  - first step: hetero_conv and residual connection
    - [HeteroGraphConv](../../models/RHGNN.py)
    - Use RelationGraphConv to create hetero_conv, each RelationGraphConv deals with a single type of relation
    - A generic module for computing convolution on heterogeneous graphs
  
  - second step: relation_crossing_layer
    - [RelationCrossing](../../models/RHGNN.py)
    - Create relation_crossing_layer
    - Establish connections of node representations to improve message passing across different relations and automatically distinguish the importance of relations
  
  - third step: relation_propagation_layer
  
  R-HGNN layer is composed of these three components and stack L layers to receive information from multi-hop neighbors. Finally, the L layers could provide relation-aware node representations for target node.

- [RelationFusing](../../models/RHGNN.py)
  - Aggregate the relation-aware node representations into a compact node representation
  - Create relation_fusing
  

## Hyper-parameter specific to the model

```python
num_heads  = 8            
hidden_units = 64
relation_hidden_units = 8
num_layers = 2
learning_rate = 0.001
dropout = 0.5
residual = True

```

Best config can be found in [best_config](../../utils/best_config.py)

## More

#### Contirbutor

Tianyu Zhao, Qi Zhang[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [1049371730@qq.com](mailto:1049371730@qq.com).
