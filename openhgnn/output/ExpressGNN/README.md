# ExpressGNN[ICLR2020]

-   paper: [Efficient Probabilistic Logic Reasoning With Graph Neural Networks](https://arxiv.org/abs/2001.11850)
-   Code from author: [ExpressGNN](https://github.com/expressGNN/ExpressGNN)


## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m ExpressGNN -d EXP_FB15k-237 -t link_prediction -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.


## Performance: link_prediction

|           |               fb15k237             
|:---------:| :-----------------------------: 
|    MRR    | paper: 0.49    OpenHGNN: 0.4399
|   H@10    | paper: 0.608   OpenHGNN: 0.5668

## Dataset

Supported dataset: cora, fb15k-237, kinship, uw_cse

### Cora

The Cora dataset is a widely used academic citation network, containing scientific papers categorized into various research topics, making it valuable for research in machine learning and graph analysis.

### fb15k-237

The FB15k-237 dataset is a knowledge graph dataset that focuses on entity and relation prediction tasks, derived from Freebase with 237 relations.

### kinship

The Kinship dataset is a collection of genealogical information, encompassing family relationships and demographics, often used for studying kinship recognition in computer vision and social sciences.

### uw_cse


The UW CSE (Computer Science and Engineering) dataset is a repository of academic information, including faculty, courses, and research projects, commonly used for educational and research purposes in computer science.

## TrainerFlow: link_prediction

#### model

ExpressGNN is an extension of Graph Neural Networks (GNNs).

### Graph Neural Networks (GNNs)

GNNs model node embeddings in a graph-structured data by recursively aggregating neighbor information.

### ExpressGNN Extension

ExpressGNN extends GNNs by incorporating temporal information, capturing evolving relationships in dynamic graphs.


## More

#### Contributor

Ziyao Lin, Fengqi Liang[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  ziyao_lin@hust.edu.cn, lfq@bupt.edu.cn