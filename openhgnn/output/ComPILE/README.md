# ComPILE[AAAI 2021]

-   Paper: [Communicative Message Passing for Inductive Relation Reasoning](https://arxiv.org/pdf/2012.08911.pdf)
-   Author's code: https://github.com/TmacMai/CoMPILE_Inductive_Knowledge_Graph
-   Note: The difference between ComPILE and Grail is mainly on the model structure.

## How to run

* Clone the Openhgnn-DGL

```bash
python main.py -m Grail -d WN18RR_v1 -t link_prediction -g 0
```
| inductive |               WN18RR_v1              
|:---------:| :-----------------------------:
|   H@10    | paper: 83.60    OpenHGNN: 87.23
We report the result of the best valid epoch.

If you do not have gpu, set -gpu -1.
#### Contributor

Shuaikun Liu, Fengqi Liang[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to
liushuaikun@bupt.edu.cn, lfq@bupt.edu.cn








