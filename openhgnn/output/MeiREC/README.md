# MEIRec[KDD2019]

-   paper: [Metapath-guided Heterogeneous Graph Neural Network for Intent Recommendation ](https://dl.acm.org/doi/abs/10.1145/3292500.3330673)
-   Code from author: [MEIRec](https://github.com/googlebaba/KDD2019-MEIRec)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m MeiREC -d meirec -t meirec -g 0 
  ```

  If you do not have gpu, set -gpu -1.

  the dataset meirec is supported.

## Performance: Recommendation

-   Device: GPU, **GeForce RTX 3090**
-   Dataset: meirec

| Recommendation |               AUC               
|:--------------:| :-----------------------------: 
|     1-day      | paper: 72.4%    OpenHGNN: 72.6% 
|     3-day      | paper: 73.6%    OpenHGNN: 72.9% 
|     5-day      | paper: 74.1%    OpenHGNN: 74.6% 

## Dataset

-   We process the meirec dataset given by [MeiREC](https://github.com/googlebaba/KDD2019-MEIRec). 
### Description

- Last.FM

  |            | Last.FM |
  |:-------:|:-----------:|
  | User       |  15000  |
  | item       |  45000  |
  | query      |  42346  |
  | term       | 280000  |
  | relations  |    4    |



## TrainerFlow: Recommendation

#### model

- ​	MeiREC
  - ​		MeiREC is to aggregate the entity representation and its neighborhood representation based on different metapaths



## More

#### Contributor

Saisai Geng[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  15010713603@qq.com



