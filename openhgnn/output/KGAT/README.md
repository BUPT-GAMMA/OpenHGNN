# KGAT[KDD2019]

Paper: [**KGAT: Knowledge Graph Attention Network forRecommendation**](https://arxiv.org/pdf/1905.07854v2.pdf)
-   
-   The author's [codes](https://github.com/xiangwang1223/knowledge_graph_attention_network) is written by tensorflow.
-   Our KGAT is reproduced based on a pytorch version provided by [here](https://github.com/LunaBlack/KGAT-pytorch)

## How to run

- Clone the Openhgnn-DGL

Candidate dataset: yelp2018,amazon-book,last-fm

To distinguish it from other models' datasets, we changed their names:
 * yelp2018_KGAT for yelp2018
 * amazon-book_KGAT for amazon-book
 * LastFM_KGAT for last-fm
 
 haha
 
  ```bash
  python main.py -m KGAT -d amazon-book_KGAT -t recommendation -g 0 
  ```
  
  ```bash
  python main.py -m KGAT -d amazon-book_KGAT -t recommendation -g 0 
  ```
  
  ```bash
  python main.py -m KGAT -d amazon-book_KGAT -t recommendation -g 0 
  ```

  If you do not have gpu, set -gpu -1.


## Datasets

-   We process the KGCN dataset given by [KGCN](https://github.com/hwwang55/KGCN). It saved as dgl.heterograph and can be loaded by [dgl.load_graphs](https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html)

| | | Amazon-book | Last-FM | Yelp2018 |
|:---:|:---|---:|---:|---:|
|User-Item Interaction| Users | 70,679 | 23,566 | 45,919|
| | Items | 24,915 | 48,123 | 45,538|
| | Interactions | 847,733 | 3,034,796 | 1,185,068|
|Knowledge Graph | Entities | 88,572 | 58,266 | 90,961 |
| | Relations | 39 | 9 | 42 |
| | Triplets | 2,557,746 | 464,567 | 1,853,704|

## Performance:

* Note:

* `amazon-book`：

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| paper | all test users         | --         | ---------            | 0.1489              | 0.1006              |
| ours  | all test users         | 39         | 0.0149               | 0.1431              | 0.0769              |

* `last-fm`：

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| paper | all test users         | --         | ---------            | 0.0870              | 0.1325              |
| ours  | all test users         | 82         | 0.0332               | 0.0832              | 0.0721              |

* `yelp2018`：

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| paper | all test users         | --         | ---------            | 0.0712              | 0.0867              |
| ours  | all test users         | 16         | 0.0160               | 0.0678              | 0.0446              |

## TrainerFlow: kgat_trainer

#### model

* KGAT
    * Key point:
        * Model the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.
        * Train KG part and CF part in turns.

## More

#### Contributor

Yifei Shao

#### If you have any questions,

Submit an issue or email to  [myth@bupt.edu.cn](mailto:myth@bupt.edu.cn).
