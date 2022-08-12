Quick Start
==========================

Quick Start From API
--------------------

Run experiments:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Running an existing baseline model on an existing benchmark

.. code-block:: python

    from openhgnn import Experiment
    experiment = Experiment(model='RGCN', dataset='acm4GTN', task='node_classification', gpu=-1, lr=0.05, hidden_dim=64,
                            max_epoch=30, num_layers=3)
    experiment.run()

Hyper-parameter optimization
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Run an experiment with optuna

.. code-block:: python

    from openhgnn import Experiment
    def search_space(trial):
        return {
            "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64]),
            "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
            'num_layers': trial.suggest_int('num_layers', 2, 3)
        }
    experiment = Experiment(model='RGCN', dataset='acm4GTN', task='node_classification', gpu=-1,
                            hpo_search_space=search_space, hpo_trials=20)
    experiment.run()

OpenHGNN will determine hyperparameters in the following order:

- If ``hpo_search_space`` is not None, search for the best hyperparameter by optuna. This is controlled by :func:`func_search` in ``./openhgnn/auto/hpo.py``. Please refer `here <https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/auto>`_ for more details.
- Otherwise, if ``--use_best_config`` is enabled, load the best hyperparameters built within OpenHGNN. The configurations are in ``./openhgnn/utils/best_config.py``.
- Otherwise, load the hyperparameters in ``./openhgnn/config.ini``.


Quick Start From Source
-----------------------

Besides using API, you can also directly run the source code of OpenHGNN.

.. code:: bash

    >>> python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config
    usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config]

Optional arguments are as follows:

    - ``--model MODEL``, ``-m MODEL`` name of models.  Please refer to the `model list <https://github.com/BUPT-GAMMA/OpenHGNN#models>`_ for supported models and their names.
    - ``--task TASK``, ``-t TASK`` name of task.  Please refer to `tasks <https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/tasks#Supported task>`_  for the list of tasks.
    - ``--dataset DATASET``, ``-d DATASET`` name of datasets.  Please refer to `datasets <https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset#Dataset>`_ for the list of datasets.
    - ``--gpu GPU``, ``-g GPU``	controls which GPU you will use. If you do not have GPU, set ``-g -1``.
    - ``--use_best_config`` use the best config OpenHGNN has found in the dataset with the model. If you want to set the different hyper-parameter, modify the `config.ini <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/config.ini>`_ file manually.
    - ``--load_from_pretrained`` will load the model from a default checkpoint.

e.g.:

.. code:: bash

    python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config
