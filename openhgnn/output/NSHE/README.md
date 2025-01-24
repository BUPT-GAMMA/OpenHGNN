# NSHE[IJCAI 2020]

Paper: [Network Schema Preserving Heterogeneous Information Network Embedding](https://www.ijcai.org/Proceedings/2020/0190.pdf)

Code from author: https://github.com/Andy-Border/NSHE

## How to run

Clone the Openhgnn-DGL

```bash
python main.py -m NSHE -t node_classification -d acm4NSHE -g 0 --use_best_config
```

If you do not have gpu, set -gpu -1.

## Performance

#### Node classification for acm4NSHE

| Node classification | Macro-F1 | Micro-F1 |
| ------------------- | -------- | -------- |
| paper               | 83.27    | 84.12    |
| OpenHGNN            | 84.78    | 84.95    |

## Dataset

We process the acm dataset given by [NSHE](https://github.com/Andy-Border/NSHE/tree/master/data). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/acm4NSHE.zip
```

Or run the code mentioned above and it will download automaticlly.

##### Description: [acm4NSHE](../../dataset/#acm)

## TrainerFlow: NSHETrainer

- Model:Encoder
  - GCN
  - Context-encoder
- Preserving Pairwise Proximity
  - Sample positive edge and negative edge
- Preserving Network Schema Proximity
  - Network Schema Instance Sampling

Note: [TODO] We will use the dataloader to combine the two sampler without storing the temporal file and use mini-batch trainer to improve the training efficiency.

## Hyper-parameter specific to the model

```python
num_e_neg = 2 # number of negative edges
num_ns_neg = 3 # number of negative schemas
```

Best config can be found in [best_config](../../utils/best_config.py)

## More

#### Contirbutor

Tianyu Zhao[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [tyzhao@bupt.edu.cn](mailto:tyzhao@bupt.edu.cn).