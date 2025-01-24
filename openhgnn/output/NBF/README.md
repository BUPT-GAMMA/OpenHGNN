# NBF_Net[NIPS 2021]

Paper: [**Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction**](https://proceedings.neurips.cc/paper_files/paper/2021/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf)

Code from author: https://github.com/DeepGraphLearning/NBFNet

#### How to run

Clone the Openhgnn-DGL

```bash
python main.py -m NBF -t link_prediction -d NBF_WN18RR -g 0
```

If you do not have gpu, set -gpu -1.

Candidate dataset: NBF_WN18RR , NBF_FB15k-237

#### Performance


| Metric | WN18RR | 
| ------------------- | ------- | 
| MRR       | 69.72   | 
| HITS@1       | 60.37   | 
| HITS@3       | 77.66   | 
| HITS@10       | 82.71   | 
| HITS@10_50       | 95.40   | 

#### Model

We implement NBF_Net with GeneralizedRelationalConv

### Dataset

Supported dataset: WN18RR , FB15k-237

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/WN18RR.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/FB15k-237.zip
```

### Hyper-parameter specific to the model

```python
input_dim = 32
hidden_dims = [32, 32, 32, 32, 32, 32]
message_func = distmult
aggregate_func = pna
short_cut = True
layer_norm = True
dependent = False
num_negative = 32
strict_negative = True
adversarial_temperature = 1
lr = 0.005
gpus = [0]
batch_size = 64
num_epoch = 20
log_interval = 100

```

All config can be found in [config.ini](../../config.ini)



## More

#### If you have any questions,

Submit an issue or email to [zhaozihao@bupt.edu.cn](mailto:zhaozihao@bupt.edu.cn).
