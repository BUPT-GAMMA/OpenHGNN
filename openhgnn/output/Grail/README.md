# Grail[ICML 2020]

-   Paper: [Inductive Relation Prediction by Subgraph Reasoning](https://arxiv.org/pdf/1911.06962.pdf)
-   Author's code: https://github.com/kkteru/grail


## How to run

* Clone the Openhgnn-DGL

```bash
python main.py -m Grail -d WN18RR_v1 -t link_prediction -g 0
```
| inductive |               WN18RR_v1              
|:---------:| :-----------------------------:
|   H@10    | paper: 82.45   OpenHGNN: 0.8431


If you do not have gpu, set -gpu -1.
#### Contributor

Shuaikun Liu, Fengqi Liang[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to
liushuaikun@bupt.edu.cn, lfq@bupt.edu.cn






