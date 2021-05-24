# HetGNN[KDD2019]

Paper: [Heterogeneous Graph Neural Network](https://dl.acm.org/doi/abs/10.1145/3292500.3330961)

Code from author: https://github.com/chuxuzhang/KDD2019_HetGNN

#### How to run

Clone the Openhgnn-DGL

```python
python main.py -m HetGNN -t node_classification -d academic4HetGNN -g 0
```

If you do not have gpu, set -gpu -1.

#### Performance

Node classification 

| Node classification | Macro-F1 | Micro-F1 |
| ------------------- | -------- | -------- |
| paper               | 0.978    | 0.979    |
| OpenHGNN            | 0.9701   | 0.9705   |

Author link prediction

| A-II(type-1) authors link prediction | AUC    | F1     |
| ------------------------------------ | ------ | ------ |
| paper                                | 0.717  | 0.669  |
| OpenHGNN                             | 0.7175 | 0.7178 |

### Dataset

We process the Academic II dataset given by [HetGNN](https://github.com/chuxuzhang/KDD2019_HetGNN/tree/master/data/academic). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/academic4HetGNN.zip
```

Or run the code mentioned above and it will download automaticlly.

##### Description

Number of nodes:

- Author: 28646
- Paper: 21044
- Venue: 18

Number of edges:

- author-paper: 69311
- paper-author: 69311
- Paper-paper: 21357
- Paper-venue: 21044
- Venue-paper:21044

#### TrainerFlow: HetGNNTrainer

- Sampling Heterogeneous Neighbors (C1)
  - Build a hetgnn_graph, which is used in aggregation neighbours. So it limit the scale of graph though we give the mini-batch trainer. [openhgnn/sampler/HetGNN_sampler.py]
  - [TODO] Combine the sampler with NeighborSampler without generating the hetgnn_graph.
- Encoding Heterogeneous Contents (C2)
  - 
- Aggregating Heterogeneous Neighbors (C3)

#### Parameter

You can modify the parameters in openhgnn/config.ini

##### Description

```
seed = 0
learning_rate = 0.01
weight_decay = 0.0001

dim = 128
max_epoch = 100
batch_size = 32
window_size = 5
num_workers = 0
batches_per_epoch = 20

rw_length = 10
rw_walks = 30
rwr_prob = 0.5

patience = 100
mini_batch_flag = True
```

