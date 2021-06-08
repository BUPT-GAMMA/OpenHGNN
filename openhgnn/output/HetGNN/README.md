# HetGNN[KDD2019]

Paper: [Heterogeneous Graph Neural Network](https://dl.acm.org/doi/abs/10.1145/3292500.3330961)

Code from author: https://github.com/chuxuzhang/KDD2019_HetGNN

## How to run

Clone the Openhgnn-DGL

```bash
python main.py -m HetGNN -t node_classification -d academic4HetGNN -g 0
```

If you do not have gpu, set -g -1.

## Performance

#### Node classification for academic4HetGNN

| Node classification | Macro-F1 | Micro-F1 |
| ------------------- | -------- | -------- |
| paper               | 0.978    | 0.979    |
| OpenHGNN            | 0.9701   | 0.9705   |

#### Author link prediction for academic4HetGNN

| A-II(type-1) authors link prediction | AUC    | F1     |
| ------------------------------------ | ------ | ------ |
| paper                                | 0.717  | 0.669  |
| OpenHGNN                             | 0.7235 | 0.7205 |

## Dataset

We process the Academic II dataset given by [HetGNN](https://github.com/chuxuzhang/KDD2019_HetGNN/tree/master/data/academic). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/academic4HetGNN.zip
```

Or run the code mentioned above and it will download automaticlly.

##### Description: [academic4HetGNN](../../dataset/#academic4HetGNN)

## TrainerFlow: HetGNNTrainer

- Sampling Heterogeneous Neighbors (C1)
  - It designs a heterogeneous neighbors sampling strategy based on random walk with restart (RWR). Build a [hetgnn_graph](../../sampler/HetGNN_sampler.py), which is used in aggregation neighbours in form of a full graph. So it limits the scale of graph though we give the mini-batch trainer. [openhgnn/sampler/HetGNN_sampler.py]
  - [TODO] Combine the sampler with NeighborSampler without generating the hetgnn_graph.
- Encoding Heterogeneous Contents (C2)
  - encoder_het_content(nn.Module)
  - Note: A node can carry unstructured heterogeneous con- tents,e.g.,attributes,textorimage. So if you build a new dataset, overwrite the *extract_feature* func in HetGNN(BaseModel) [openhgnn/models/HetGNN.py]
- Aggregating Heterogeneous Neighbors (C3)
  - aggregate_het_neigh(nn.Module)

## Hyper-Parameter

You can modify the parameters in openhgnn/config.ini

#### Description

```python
batch_size # the batch_size of nodes sampled in SkipGramBatchSampler
window_size # window size for relation extration
num_workers	# how many subprocesses to use for data loading. ''0'' means that the data 										will be loaded in the main process.
batches_per_epoch # the batches trained in a epoch

# the three parameters decided in what nodes aggregated.[used in building hetgnn_graph]
rw_length # length of each walk
rw_walks # window size for relation extration
rwr_prob # the probability of restart

#train epoch and early stop patience
max_epoch
patience
mini_batch_flag = True # we only support the mini-batch training.
```

Best config can be found in [best_config](../../utils/best_config.py)

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).

