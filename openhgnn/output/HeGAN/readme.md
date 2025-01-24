# HeGAN[KDD2019]

-   paper: [Adversarial Learning on Heterogeneous Information Networks](https://dl.acm.org/doi/10.1145/3292500.3330970)
-   Code from author: [HeGAN](https://github.com/librahu/HeGAN)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m HeGAN -d yelp4HeGAN -t node_classification -g -1
  ```

  If you do not have gpu, set -gpu -1.


## Performance: NodeClassification

|dataset| paper's | author's | ours |
| ----- | -------- | ---- |----|
| yelp(%) | 85.24 | 80.01 |79.35|


## Dataset

-   We process the yelp dataset given by [yelp](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Yelp_2). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)


### Description


- Yelp  
 [Dataset Link](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Yelp_2)

  
  ### Entity Statistics
  
  | Entity      | #Entity |
  | ----------- | ------- |
  | User        | 1,286   |
  | Business    | 2,614   |
  | Service     | 2       |
  | Star level  | 9       |
  | Reservation | 2       |
  | Category    | 3       |
  

  

## TrainerFlow: node_classification

#### model

- 	HeGAN
  - 		HeGAN is a wrapper of Generator and Discriminator.
- 	Generator
   - 	A Discriminator `D` eveluates the connectivity between the pair of nodes `u` and `v` w.r.t. a relation `r`.

- 	Discriminator
   -    A generator `G` samples fake node embeddings from a continuous distribution. The distribution is Gaussian distribution.


## More

#### Contributor

Hui Han [GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  hanhui@bupt.edu.cn