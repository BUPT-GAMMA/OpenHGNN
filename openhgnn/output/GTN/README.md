# GTN[NeurIPS2019]

Paper: [Graph Transformer Networks](https://arxiv.org/abs/1911.06455)

Code from author: https://github.com/seongjunyun/Graph_Transformer_Networks

#### How to run

Clone the Openhgnn-DGL

```python
python main.py -m GTN -t node_classification -d acm4GTN -g 0
```

If you do not have gpu, set -gpu -1.

#### Performance

Node classification 

| Node classification | GTN-I | GTN(proposed) |
| ------------------- | ----- | ------------- |
| paper               | 91.13 | 92.68         |
| OpenHGNN            | -     | -             |

### Dataset

We process the acm dataset given by [HAN](https://github.com/Jhy1993/HAN). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/acm4GTN.zip
```

Or run the code mentioned above and it will download automaticlly.

##### Description

Number of nodes:

- Author: 3025
- Paper: 5912
- Subject: 57

Number of edges:

- author-paper: 9936
- paper-author: 9936
- paper-subject: 3025
- subject-paper: 3025

Dimensions of features is 1902

