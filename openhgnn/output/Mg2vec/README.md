# mg2vec[TKDE2020]

-   paper: [mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via Metagraph Embedding](https://ieeexplore.ieee.org/document/9089251)
-   Code from author: [mg2vec](https://github.com/zwtywwq/mg2vec)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m Mg2vec -t edge_classification -d dblp4Mg2vec_4 -g 0 # meta_graph's size is up to 4 
  python main.py -m Mg2vec -t edge_classification -d dblp4Mg2vec_5 -g 0 # meta_graph's size is up to 5
  ```

  If you do not have gpu, set -gpu -1.

  the dataset dblp4Mg2vec is supported.

## Performance: Relation Prediction

-   Device: GPU, **GeForce RTX 2080ti**
-   Dataset: DBLP
-   The average of 5 experiments.

| Dataset             | Macro-F1                           | Accuracy                            |
| ------------------- |------------------------------------|-------------------------------------|
| DBLP                | paper: 87.38%    OpenHGNN:  87.34% | paper:  89.90%    OpenHGNN:  89.97% |

## Data Process

-   To learn how we process the raw graph file and get the input file like meta.txt(which you will find in dataset/dblp4Mg2vec after you run this model), please go to the [DataMaker-For-Mg2vec](https://github.com/null-xyj/DataMaker-For-Mg2vec)

## TrainerFlow:

-   **Read input file**
    - Read the input file(meta.txt), which can be achieved by DataMaker mentioned above. This file contains the metagraph information we mined.
    - Considering the huge file size(maybe nearly 20G), we split the file into several blocks and read one block into memory at a time.
-   **Train the node embeddings**
    -   Firstly, we initialize the embedding for every node and metagraph
    -   Next, we adopt an unsupervised method to train the node embeddings and metagraph embeddings. In detail, for every node, we keep its embedding close to the metagraph it belongs to and far away from the metagraph we get by negative sampling.
-   **The Downstream Task: Relation Prediction**
    - After we achieve the trained node embeddings, we use the embeddings to complete the relation prediction task.
    - The relation prediction task is achieved by edge classification task. If two nodes are connected with a relation, we see the relation as an edge. Then we can adopt the edge classification to complete relation prediction task.
    - In this edge classification task, We use an SVM classifier with C=100 and gamma="auto". 

## Hyper-parameter specific to the model

You can modify the parameters in openhgnn/config.ini

### Description

```python
dim = 128 # the dimension of the node embedding and metagraph embedding 						
sample_num = 10 # the number of negative sampling metagraphs		 			   
alpha = 0.5 # a parameter to balance the first-order embedding and the second-order embedding
```

Best config can be found in [best_config](../../utils/best_config.py)

## Note
- We will output the trained node embedding in output/MG2vec, and we will offer both text format and numpy format.
- We will offer you two options about the size of the meta_graph, you can set "-d dblp4Mg2vec_4" or "-d dblp4Mg2vec_5" to choose the size of the meta_graph. If you want to get a meta file which is up to 6 or more (which is really huge), you should go to the [DataMaker-For-Mg2vec](https://github.com/null-xyj/DataMaker-For-Mg2vec).

| meta-graph size | meta-graph count | Accuracy |
|-----------------|------------------|----------|
| 4               | 3                | 77.73%   | 
| 5               | 17               | 90.07%   | 
| 6               | 88               | 89.97%   | 


## More

#### Contirbutor

Yujie Xing[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to 1468334071@qq.com.

