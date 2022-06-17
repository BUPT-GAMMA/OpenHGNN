# GATNE [KDD2019]

Paper: [[GATNE] Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/abs/1905.01669)

Code from author: https://github.com/THUDM/GATNE

Code from dgl: https://github.com/dmlc/dgl/tree/master/examples/pytorch/GATNE-T

### How to run

Clone the OpenHGNN

```bash
python main.py -m GATNE-T -t link_prediction -d HGBl-amazon -g 0 --use_best_config
```

If you do not have gpu, set -gpu -1.

##### Candidate dataset: 

amazon4GATNE, twitter4GATNE, youtube4GATNE

### Performance

Link Prediction

Evaluation metric: roc_auc

| amazon4GATNE | twitter4GATNE | youtube4GATNE |
|--------------|---------------|---------------|
|              |               |               |

#### Contirbutor

Hongyi Zhang[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [zhanghongyi@bupt.edu.cn](mailto:zhanghongyi@bupt.edu.cn).

