# MHGCN[KDD 2022]

-   paper: [Multiplex Heterogeneous Graph Convolutional Network](https://doi.org/10.1145/3534678.3539482)
-   Author's code : https://github.com/NSSSJSS/MHGCN

## How to run

- Clone the Openhgnn-DGL

  ```bash
  # For node classification task
  python main.py -m MHGCN -t node_classification -d dblp4mhgcn -g 0 --use_best_config
  # For link prediction task
  python main.py -m MHGCN -t link_prediction -d dblp4mhgcn -g 0 --use_best_config
  ```

  If you do not have gpu, set -gpu -1.

  ##### Supported dataset

  dblp4mhgcn / imdb4mhgcn / alibaba4mhgcn



## Performance

#### Task: Node classification

Evaluation metric: Micro F1 / Macro F1

| Method        | DBLP              | IMDB              | Alibaba           |
|               | Macro F1| Micro F1| Macro F1| Micro F1| Macro F1| Micro F1|
| ------------- | ------- | ------- | ------- | ------- | ------- | ------- |
| OpenHGNN      | 93.28   | 93.75   | 66.81   | 70.07   | 18.76   | 39.05   |
| Author's code | 93.73   | 95.25   | 68.85   | 68.70   | 25.76   | 43.03   |
Time consumption (seconds)
| Method        | DBLP    | IMDB    | Alibaba |
| ------------- | ------- | ------- | ------- |
| OpenHGNN      | 13 s    | 6 s     | 6 s     |
| Author's code | 381 s   | 101 s   | 405 s   |


#### Task: Link prediction

Evaluation metric: R-AUC / PR-AUC / F1

| Method        | DBLP                    | IMDB                    | Alibaba                 |
|               | R-AUC | PR-AUC | F1     | R-AUC | PR-AUC | F1     | R-AUC | PR-AUC | F1     |
| ------------- | ----- | ------ | ------ | ----- | ------ | ------ | ----- | ------ | ------ |
| OpenHGNN      | 81.06 | 83.02  | 73.87  | 58.22 | 54.88  | 53.25  | 98.58 | 99.12  | 96.83  |
| Author's code | 72.11 | 72.23  | 67.66  | 66.48 | 61.76  | 59.84  | 93.15 | 93.75  | 96.55  |
Time consumption (seconds)
| Method        | DBLP    | IMDB    | Alibaba |
| ------------- | ------- | ------- | ------- |
| OpenHGNN      | 41 s    | 24 s    | 22 s    |
| Author's code | 4139 s  | 774 s   | 4502 s  |


## TrainerFlow: 
- node classification : MHGCN_LP_Trainer
- link prediction : MHGCN_LP_Trainer
### Model

- ​	MHGCN 

## Hyper-parameter specific to the model

You can modify the parameters[MHGCN] in openhgnn/config.ini. 

Best config can be found in [best_config](../../utils/best_config.py). 

## More

#### Contirbutor

Yang Liu[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [liuyang1999@bupt.edu.cn](mailto:liuyang1999@bupt.edu.cn).
