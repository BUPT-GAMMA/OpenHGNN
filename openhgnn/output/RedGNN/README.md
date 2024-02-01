# Adaprop[WWW2022]

-   paper: https://arxiv.org/pdf/2108.06040.pdf
-   Code from author: [RedGNN](https://github.com/LARS-research/RED-GNN)

This model is divided into two scenarios: transactional and inductive , so we will integrate them into two models: REDGNNT and REDGNN
## AdapropT
### How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m RedGNN -d family -t link_prediction -g 0
  ```

  If you do not have gpu, set -gpu -1.

  the dataset family is supported.

### Performance: Family

-   Device: GPU, **GeForce RTX 3090**
-   Dataset: AdapropT

| transductive |               Family              
|:------------:| :-----------------------------: 
|     MRR      | paper: 0.992    OpenHGNN: 0.9836 
|     H@1      | paper: 98.8    OpenHGNN: 0.9771 
|     H@10     | paper: 99.7    OpenHGNN: 0.9907 





### TrainerFlow: RedGNNT_trainer

#### model

- ​	RedGNNT 
  - ​		RedGNN is a GNN based knowledge graph reasoning model.


  

## RedGNN
### How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m RedGNN -d fb237_v1 -t link_prediction -g 0
  ```

  If you do not have gpu, set -gpu -1.

  the dataset AdapropI is supported.

 ## Performance: fb15k237_v1

-   Device: GPU, **GeForce RTX 2080Ti**
-   Dataset: AdapropI
- 
| inductive |               fb15k237_v1              
|:---------:| :-----------------------------: 
|    MRR    | paper: 0.369    OpenHGNN: 0.3543 
|    H@1    | paper: 0.302    OpenHGNN: 0.2968
|   H@10    | paper: 0.483    OpenHGNN: 0.4672





### TrainerFlow: RedGNN

#### model

- ​	RedGNN 
  - ​		RedGNN is a GNN based knowledge graph reasoning model.

#### Contributor

fengqi liang[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  lfq@bupt.edu.cn



