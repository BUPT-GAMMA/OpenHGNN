# DisenKGAT[CIKM 2021]

Paper: [**DisenKGAT: Knowledge Graph Embedding with Disentangled
Graph Attention Network**](https://dl.acm.org/doi/10.1145/3459637.3482424)

Code from author:  https://github.com/Wjk666/DisenKGAT

#### How to run

Clone the Openhgnn-DGL

```bash
python main.py -m DisenKGAT -t link_prediction -d DisenKGAT_WN18RR -g 0
```

If you do not have gpu, set -gpu -1.

Candidate dataset: DisenKGAT_WN18RR , DisenKGAT_FB15k-237

#### Performance


| Metric | DisenKGAT_WN18RR | DisenKGAT_FB15k-237 |
| ------------------- | ------- | ------- | 
| MR       |1406   | 167  | 
| MRR       | 0.455   | 0.354   | 
| HITS@1       | 0.431   | 0.244     |  
| HITS@3       | 0.485   | 0.387  | 
| HITS@10       | 0.521   | 0.511  | 


#### Model

We implement DisenKGAT with DisenKGAT_TransE,DisenKGAT_DistMult,DisenKGAT_ConvE,DisenKGAT_InteractE

### Dataset

Supported dataset: WN18RR , FB15k-237

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/DisenKGAT_WN18RR.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/DisenKGAT_FB15k-237.zip
```

### Hyper-parameter specific to the model

```python
#   str
name = Disen_Model
#   data =  DisenKGAT_WN18RR
#   model = DisenKGAT
score_func = interacte
opn = cross
#  gpu = 2
logdir = ./log/
config = ./config/
strategy = one_to_n
form = plain
mi_method = club_b
att_mode = dot_weight
score_method = dot_rel
score_order = after
gamma_method = norm


#   int
k_w = 10
batch = 2048
test_batch = 2048
epoch = 1500
num_workers = 10
seed = 41504
init_dim = 100
gcn_dim = 200
embed_dim = 200
gcn_layer = 1
k_h = 20
num_filt = 200
ker_sz = 7
num_bases = -1
neg_num = 1000
ik_w = 10
ik_h = 20
inum_filt = 200
iker_sz = 9
iperm = 1
head_num = 1
num_factors = 3
early_stop = 200
mi_epoch = 1

#   float
feat_drop = 0.3
hid_drop2 = 0.3
hid_drop = 0.3
gcn_drop = 0.4
gamma = 9.0
l2 = 0.0
lr = 0.001
lbl_smooth = 0.1
iinp_drop = 0.3
ifeat_drop = 0.4
ihid_drop = 0.3
alpha = 1e-1
max_gamma = 5.0
init_gamma = 9.0

#   boolean
restore = False
bias = False
no_act = False
mi_train = True
no_enc = False
mi_drop = True
fix_gamma = False

```

All config can be found in [config.ini](../../config.ini)



## More

#### If you have any questions,

Submit an issue or email to [zhaozihao@bupt.edu.cn](mailto:zhaozihao@bupt.edu.cn).
