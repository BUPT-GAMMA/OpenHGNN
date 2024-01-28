# LTE

-paper: [Rethinking Graph Convolutional Networks in Knowledge
Graph Completion
](https://arxiv.org/pdf/2202.05679.pdf)


## How to run
- Clone the Openhgnn-DGL
  ```bash
  python main.py -m LTE -d LTE -t link_prediction -g 0
  ```

for high efficiency, only gpu

## Performance: Recommendation

-   Device: GPU, **GeForce GTX 3090**
-   Dataset:LTE_dataset


| LTE-TransE |               FB237               |
|:--------------:|:---------------------------------:|
|      MRR       | paper: 0.334    OpenHGNN: 0.3272  |
|      H@1       | paper: 0.241    OpenHGNN: 0.23527 | 
|      H@3       |  paper: 0.370   OpenHGNN: 0.3616  | 
|      H@10      |  paper: 0.519   OpenHGNN: 0.5111  | 


#### Contributor

Zikai Zhou[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  460813395@qq.com


