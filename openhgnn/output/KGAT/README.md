# KGAT[KDD2019]

-   Paper: [**KGAT: Knowledge Graph Attention Network for Recommendation**](https://arxiv.org/pdf/1905.07854v2.pdf)
-   The author's [code](https://github.com/xiangwang1223/knowledge_graph_attention_network) is written by tensorflow.
-   Our KGAT is reproduced based on a pytorch version [here](https://github.com/LunaBlack/KGAT-pytorch)

## How to run

- Clone the Openhgnn-DGL

- Candidate dataset: yelp2018, amazon-book, last-fm

- To distinguish it from other models' datasets, we change their names:

     * yelp2018_KGAT for yelp2018
     * amazon-book_KGAT for amazon-book
     * LastFM_KGAT for last-fm

- Arguments

```
--lr                     Learning rate                                                                                Default is 0.0001.
--max_epoch              Max_epoch                                                                                    Default is 1000.
--stopping_steps         Early stopping                                                                               Default is 20.
--use_pretrain           0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model       Default is 1.
--aggregation_type       Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}              Default is bi-interaction.
--entity_dim             User / entity Embedding size                                                                 Default is 64.
--relation_dim           Relation Embedding size                                                                      Default is 64.
--conv_dim_list          Output sizes of every aggregation layer                                                      Default is [64, 32, 16].
--mess_dropout           message dropout for each deep layer                                                          Default is [0.1, 0.1, 0.1].
--cf_l2loss_lambda       Lambda when calculating cf l2 loss                                                           Default is 1e-5.
--kg_l2loss_lambda       Lambda when calculating kg l2 loss                                                           Default is 1e-5.
--cf_batch_size          CF batch size                                                                                Default is 1024.
--kg_batch_size          KG batch size                                                                                Default is 1024. 
--test_batch_size        Test batch size (the user number to test every batch)                                        Default is 1024. 
--multi_gpu              train with multi_gpu                                                                         Default is False.
--K                      Calculate metric@K when evaluating                                                           Default is 20. 
--pretrain_model_path    Trained model                                                                                Default is None.
```
  
  We use pretrain embedding provided by author by default. The embedding is trained using [bprmf](https://dl.acm.org/citation.cfm?id=1795167)
  Since we use some custom parameters, we need to instantiate the Experiment class ourselves.
  
  ```bash
  python main.py -m KGAT -d amazon-book_KGAT -t recommendation -g 0
  #or
  from openhgnn import Experiment
  e=Experiment(model='KGAT',dataset='yelp2018_KGAT',gpu=-1,task='recommendation',use_pretrain=1)
  e.run()
  ```
  
  If you want to train your model without pretrain embedding, please set use_pretrain to 0 and adjust stopping_steps. 
 
  ```bash
  from openhgnn import Experiment
  e=Experiment(model='KGAT',dataset='yelp2018_KGAT',gpu=-1,task='recommendation',use_pretrain=0,stopping_steps=800)
  e.run()
  ```
  
  If you want to use a trained model, please set use_pretrain to 2 and point out pretrain_model_path.
  
  ```bash
  from openhgnn import Experiment
  e=Experiment(model='KGAT',dataset='yelp2018_KGAT',gpu=-1,task='recommendation',use_pretrain=2,pretrain_model_path=(up to you))
  e.run()
  ```

  If you do not have gpu, set -gpu -1.


## Datasets

-   We implement class: KGAT_recommendation to download and process the datasets. 
-   Note: VPN is needed to download the datasets. Or you can still run the code and follow the instruction to download the datasets and place them on the right path.

| | | Amazon-book | Last-FM | Yelp2018 |
|:---:|:---|---:|---:|---:|
|User-Item Interaction| Users | 70,679 | 23,566 | 45,919|
| | Items | 24,915 | 48,123 | 45,538|
| | Interactions | 847,733 | 3,034,796 | 1,185,068|
|Knowledge Graph | Entities | 88,572 | 58,266 | 90,961 |
| | Relations | 39 | 9 | 42 |
| | Triplets | 2,557,746 | 464,567 | 1,853,704|

## Performance:

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

* Note: Recall@20 matches the performance of the original model, but NDCG@20 does not match the performance of the original model. This is probably because NDCG@20 is implemented differently in the pytorch version from the source code. We will fix this in the future.

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
