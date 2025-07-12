# HCMGNN[IJCAI2024]

-   paper: [Heterogeneous Causal Metapath Graph Neural Network for Gene-Microbe-Disease Association Prediction](https://arxiv.org/pdf/2406.19156)
-   Code from author: [HCMGNN](https://github.com/zkxinxin/HCMGNN)

## How to run

- Clone the Openhgnn-DGL

  ```bash
  python -m HCMGNN -d dataHCMGNN -t recommendation -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  the dataset Last.FM is supported.

## Performance: Recommendation

-   Device: GPU, **GeForce RTX 2090**
-   Dataset: dataHCMGNN

|               | Hits@5  | Hits@3  | Hits@1  | NDCG@5  | NDCG@3  | NDCG@1  |  MRR   |
| :------------ | :------ | :------ | :------ | :------ | :------ | :------ | :----- |
| HCMGNN-CV     | 0.8181  | 0.7789  | 0.6244  | 0.7325  | 0.7165  | 0.6244  | 0.7181 |
| HCMGNN-Indep  | 0.9913  | 0.9767  | 0.8280  | 0.9229  | 0.9168  | 0.8280  | 0.9005 |
| paper         | 0.9641  | 0.9417  | 0.7947  | 0.8923  | 0.8827  | 0.7947  | 0.8712 |

## Dataset

-   We process the dataHCMGNN dataset given by [HCMGNN](https://github.com/zkxinxin/HCMGNN). It saved as zip and can be loaded by download.

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



