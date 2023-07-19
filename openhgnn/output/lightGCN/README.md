# lightGCN[SIGIR 2020]

-   paper: [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3397271.3401063)
-   Code from author: [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m lightGCN -d gowalla -t recommendation -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  the dataset gowalla, yelp2018, amazon-book are supported.

## Performance: Recommendation

-   Device: GPU, **GeForce RTX 4070**

| Recommendation |              recall               |               ndcg                |
|:--------------:|:---------------------------------:|:---------------------------------:|
|    gowalla     | paper: 0.1830    OpenHGNN: 0.1841 | paper: 0.1554    OpenHGNN: 0.1553 |
|    yelp2018    | paper: 0.0649    OpenHGNN: 0.0648 | paper: 0.0530    OpenHGNN: 0.0532 |
|  amazon-book   | paper: 0.0411    OpenHGNN: 0.0414 | paper: 0.0315    OpenHGNN: 0.0316 |

## Dataset

-   We process the lightGCN dataset given by [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch). 

### Description

- Last.FM

  |              |   gowalla    | yelp2018     | amazon-book |
  |:------------:|:------------:|:-----------:|:---- ------:|
  | User         |    29858     |    31668    |   52643     |
  | item         |    40981     |    38048    |   91599     | 
  | interactions |   1027370    |   1561406   |   2984108   | 
  | Density      |   0.00084    |   0.00130   |   0.00062   |


  

## TrainerFlow: Recommendation

#### model

- ​	lightGCN
  - ​		lightGCN is only to aggregate but don't use feature transformation and nonlinear activation



## More

#### Contributor

Saisai Geng[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  [15010713603@qq.com](mailto:15010713603@qq.com).



