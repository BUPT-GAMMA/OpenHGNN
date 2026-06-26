# HCMGNN[IJCAI2024]

-   paper: [Heterogeneous Causal Metapath Graph Neural Network for Gene-Microbe-Disease Association Prediction](https://arxiv.org/pdf/2406.19156)
-   Code from author: [HCMGNN](https://github.com/zkxinxin/HCMGNN)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python main.py -m HCMGNN -d GMD4HCMGNN -t recommendation -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.


## Performance: Recommendation

-   Device: GPU, **GeForce RTX 2080**
-   Dataset: dataHCMGNN

|               | Hits@5  | Hits@3  | Hits@1  | NDCG@5  | NDCG@3  | NDCG@1  |  MRR   |
| :------------ | :------ | :------ | :------ | :------ | :------ | :------ | :----- |
| HCMGNN-CV     | 0.9647  | 0.9381  | 0.7791  | 0.8861  | 0.8752  | 0.7791  | 0.8629 |
| HCMGNN-Indep  | 0.9650  | 0.9534  | 0.8222  | 0.9064  | 0.9015  | 0.8222  | 0.8899 |
| paper         | 0.9641  | 0.9417  | 0.7947  | 0.8923  | 0.8827  | 0.7947  | 0.8712 |

## Dataset

-   We process the GMD4HCMGNN dataset given by [HCMGNN](https://github.com/zkxinxin/HCMGNN). The dataset is stored in s3://dgl-data/dataset/openhgnn/GMD4HCMGNN.zip.

### Description

-   The implementation of this paper is mainly divided into two parts: 5-fold cross-validation and independent testing. Therefore, the running results show that it is approximately required to run 6 times, and due to the early stop mechanism, it generally will not complete one full epoch.

  

## TrainerFlow: Recommendation

#### model

* HCMGNN
    * Key point:
        * The key point of the HCMGNN model lies in integrating the meta-paths in heterogeneous graphs with causal relationships, and using graph neural networks to capture the complex correlations among genes, microorganisms and diseases, thereby achieving the prediction of the correlations among these three elements.

## More

#### Contributor

Hongyu Zuo[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  [zuohy@bupt.edu.cn](mailto:myth@bupt.edu.cn).



