Basic Usage
==========================

Run experiments
------------------
Running an existing baseline model on an existing benchmark api_dataset_

.. code:: bash

    >>>python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config
    usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config]

*optional arguments*:
    - --model MODEL,	-m MODEL	name of models
    - --task TASK,	-t TASK	name of task
    - --dataset DATASET,	-d DATASET	name of datasets
    - --gpu GPU, -g GPU	controls which gpu you will use. If you do not have gpu, set -g -1.
    - --use_best_config	use_best_config means you can use the best config in the dataset with the model. If you want to set the different hyper-parameter, modify the [openhgnn.config.ini](./openhgnn/config.ini) manually.
    - --use_hpo Besides use_best_config, we give a hyper-parameter [example](./openhgnn/auto) to search the best hyper-parameter automatically.

e.g.:

.. code:: bash

    python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config


**Note**: If you are interested in some model,
you can refer to the below `models list <https://github.com/BUPT-GAMMA/OpenHGNN#models>`_.

Evaluate a new dataset in an existing model
==============================================

Apply a new model in a
==============================================
In this section, we will create a model,
which is a very simple graph embedding algorithm.

First we import necessary library like numpy, scipy, networkx, sklearn,
we also import API like ‘BaseModel’ and ‘register_model’ from cogl/models/ to build our new model:


How to build a new model
--------------------------
**First step**

We should create a class your_model that inherits
the :ref:`Base Model <api-model>` .
and register the model with @register_model(str).

.. code-block:: python

    from openhgnn.models import BaseModel, register_model
    @register_model('demo_model')
    class Demo(BaseModel):
        ...

**Second step**

We must implement the classmethod build_model_from_args , other functions like __init__(), forward() and so on.

.. code-block:: python

    ...
    class Demo(BaseModel):
        @classmethod
        def build_model_from_args(cls, args, hg):
            ...
            return cls(...)
        def __init__():
            return
        def forward(hg, h_dict=None):
            return h_dict

**Finally**

We should fill the dict SUPPORTED_MODELS in `models/init.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/models/__init__.py>`_

Apply a existing model to a new scenario
==============================================


