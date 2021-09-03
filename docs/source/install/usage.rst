OpenHGNN basic Usage
==========================

Run experiments
-----------------
#### Running an existing baseline model on an existing benchmark api_dataset_

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config
```

usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config]

*optional arguments*:
    -h, --help            show this help message and exit
    --model MODEL,	-m MODEL	name of models
    --task TASK,	-t TASK	name of task
    --dataset DATASET,	-d DATASET	name of datasets
    --gpu GPU, -g GPU	controls which gpu you will use. If you do not have gpu, set -g -1.
    --use_best_config	use_best_config means you can use the best config in the dataset with the model. If you want to set the different hyper-parameter, modify the [openhgnn.config.ini](./openhgnn/config.ini) manually.
    --use_hpo Besides use_best_config, we give a hyper-parameter [example](./openhgnn/auto) to search the best hyper-parameter automatically.

e.g.:

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config
```

**Note**: If you are interested in some model,
you can refer to the below `models list <https://github.com/BUPT-GAMMA/OpenHGNN#models>`.

OpenHGNN In-depth Usage
=========================

Evaluate a new dataset in a e existing
-----------------------------------

Innovate a HGNN model
-----------------------

Apply a existing model to a new scenario
------------------------------------------


