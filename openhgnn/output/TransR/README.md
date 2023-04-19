# TransR[AAAI 2015]

- paper: [Learning Entity and Relation Embeddings for Knowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/)

## Basic Idea

- Entities are embedded in a continuous low dimensional vector space and each relations are embedded in different vector space.
- Judge whether triplet (entities, relationships, entities) can be considered as a fact by similarity based on distance.
- First, the entity vector is projected onto the vector space related to the relation, and then use L2 normal form as the similarity calculation method, and the formula is as follows:

$$
f(\textbf{h},\textbf{r},\textbf{t}) = \| \textbf{M}_{r}\textbf{h}+\textbf{r}-\textbf{M}_{r}\textbf{t}\|_{2}^{2}
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

- In addition, CTransR uses the results of TransE to pre cluster entities, and then learns a relationship vector $ \textbf{r}_{c} $ for each cluster. The new loss is as follows:

$$
f_{r}(h,t)=\| \textbf{M}_{r}\textbf{h}+\textbf{r}-\textbf{M}_{r}\textbf{t} \|_{2}^{2} + \alpha \| \textbf{r}_{c}-\textbf{r} \|_{2}^{2}
$$

- What`s more, the dimensions of the entity embedding vector and the relationship embedding vector can be different.

## How to run

- Clone the Openhgnn-DGL
- Run transE model first
  ```bash
  # For link prediction task
  python main.py -m TransE -t link_prediction -d FB15k -g 0 --use_best_config
  ```
- Run transR model

  ```bash
  # For link prediction task
  python main.py -m TransR -t link_prediction -d FB15k -g 0 --use_best_config
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

Testing model performance...



## TrainerFlow: [TransX flow](../../trainerflow/#TransX_flow)

## Hyper-parameter specific to the model

You can modify the parameters[TransE] in openhgnn/config.ini

## More

### Contirbutor

Xiaoke Yang

#### If you have any questions

Submit an issue or email to [x.k.yang@qq.com](mailto:x.k.yang@qq.com).
