# TransH[AAAI 2014]

- paper: [Knowledge Graph Embedding by Translating on Hyperplanes](https://ojs.aaai.org/index.php/AAAI/article/view/8870)

## Basic Idea

- Entities and relations are embedded in a continuous low dimensional vector space.
- Judge whether triplet (entities, relationships, entities) can be considered as a fact by similarity based on distance.
- First, the entity vector is projected onto the hyperplane related to the relation vector, and then use L2 normal form as the similarity calculation method, and the formula is as follows:

$$
f(\textbf{h},\textbf{r},\textbf{t}) = \| (\textbf{h}-\textbf{w}_{r}^{\top }\textbf{h}\textbf{w}_{r})+\textbf{r}-(\textbf{t}-\textbf{w}_{r}^{\top }\textbf{t}\textbf{w}_{r})\|_{2}^{2} 
$$

- The negative samples are constructed by destroying the fact triples to train the model. 

$$
    S' = \left \{ (h',r,t) \right |h'\in E \} \cup \left \{ (h,r,t') \right |t'\in E \}
$$

- The loss calculation method is as follows:

$$
    Loss = \sum_{(\textbf{h},\textbf{r},\textbf{t}) \in S}\sum_{(\textbf{h}',\textbf{r},\textbf{t}') \in S'}[\gamma - f(\textbf{h},\textbf{r},\textbf{t}) + f(\textbf{h}',\textbf{r},\textbf{t}')]_{+}
$$

- Finally, use BP to update the model.

## How to run

- Clone the Openhgnn-DGL

  ```bash
  # For link prediction task
  python main.py -m TransH -t link_prediction -d FB15k -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  ### Supported dataset

  - [FB15k](../../dataset/#FB15k)

    - Number of entities and relations

        | entities | relations |
        | -------- | --------- |
        | 14,951   |    1,345  |

    - Size of dataset

        | set type       | size  |
        | -------------- | ----- |
        | train set      | 483,142|
        | validation set | 50,000 |
        | test set       | 59,071 |

  - [WN18](../../dataset/#WN18)

    - Number of entities and relations

        | entities | relations |
        | -------- | --------- |
        | 40,493   |    18     |

    - Size of dataset

        | set type       | size    |
        | -------------- | -----   |
        | train set      | 141,442 |
        | validation set | 5,000   |
        | test set       | 5,000   |

## Performance

### Task: Link Prediction

Evaluation metric: mrr

| Dataset      | Mean Rank | Hits@10 |
|--------------|-----------|---------|
| FB15k(raw)   | 180       | 50.2    |
| FB15k(filt.) | 65        |66.1    |
| WN18(raw)    | 355       | 54.0    |
| WN18(filt.)  | 379       | 56.7    |



## TrainerFlow: [TransX flow](../../trainerflow/#TransX_flow)

## Hyper-parameter specific to the model

You can modify the parameters[TransE] in openhgnn/config.ini

## More

### Contirbutor

Xiaoke Yang

#### If you have any questions

Submit an issue or email to [x.k.yang@qq.com](mailto:x.k.yang@qq.com).
