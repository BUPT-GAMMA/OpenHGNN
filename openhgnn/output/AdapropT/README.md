# AdapropT[KDD2023]

-   paper: https://arxiv.org/pdf/2205.15319.pdf
-   Code from author: [AdapropT](https://github.com/LARS-research/AdaProp)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m AdapropT -d AdapropT -t link_prediction -g 0
  ```

  If you do not have gpu, set -gpu -1.

  the dataset AdapropT is supported.

## Performance: Recommendation

-   Device: GPU, **GeForce RTX 3090**
-   Dataset: AdapropT

| Recommendation |               AUC               
|:--------------:| :-----------------------------: 
|     1-day      | paper: 72.4%    OpenHGNN: 72.6% 
|     3-day      | paper: 73.6%    OpenHGNN: 72.9% 
|     5-day      | paper: 74.1%    OpenHGNN: 74.6% 

## Dataset

-   We process the AdapropT dataset given by [MeiREC](https://github.com/LARS-research/AdaProp/blob/main/transductive/load_data.py). 
### Description

- Last.FM

  |            | Last.FM |
  |:-------:|:-----------:|
  | User       |  15000  |
  | item       |  45000  |
  | query      |  42346  |
  | term       | 280000  |
  | relations  |    4    |



## TrainerFlow: Knowledge Graph

#### model

- ​	AdapropT 
  - ​		AdapropT is a meta-path based recommendation model, which is based on the idea of meta-path based heterogeneous graph neural network.



## More

#### Contributor

zikai Zhou[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  460813395@qq.com



