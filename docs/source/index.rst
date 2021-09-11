.. OpenHGNN documentation master file, created by
   sphinx-quickstart on Thu Aug 26 09:31:18 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenHGNN's documentation!
====================================

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/index
   install/usage

.. toctree::
   :maxdepth: 2
   :caption: Advanced Materials
   :hidden:
   :titlesonly:
   :glob:

   pipeline/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:
   :glob:

   api/openhgnn.dataset
   api/openhgnn.models
   api/openhgnn.tasks
   api/openhgnn.trainerflow
   api/dataset
   api/task
   api/layer
   api/model
   api/trainerflow


This is an open-source toolkit for Heterogeneous Graph Neural Network(OpenHGNN) based on `DGL [Deep Graph Library] <https://github.com/dmlc/dgl>`_.
We integrate SOTA models of heterogeneous graph.

Key Features
--------------

* Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and datasets using `optuna <https://optuna.org/>`_ which is a hyperparameter optimization framework.
* Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
* Efficiency: The backend dgl provides efficient APIs.

Getting Started
---------------

Requirements and Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Python <https://www.python.org/>`_ >= 3.6
* `PyTorch <https://pytorch.org/get-started/locally/>`_ >= 1.7.1
* `DGL <https://github.com/dmlc/dgl>`_ >= 0.7

Running an existing baseline model on an existing benchmark `dataset <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/dataset/#Dataset>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   >>> python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config

-g controls which gpu you will use. If you do not have gpu, set -g -1.

--use_best_config means you can use the best config in the dataset with the model. If you want to set the different hyper-parameter, modify the `openhgnn.config.ini <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/config.ini>`_ manually.

--use_hpo Besides use_best_config, we give a hyper-parameter `example <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/auto>`_ to search the best hyper-parameter automatically.

e.g.:

.. code:: bash

   >>> python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config

It is under development, and we release it in a nightly build version. For now, we just give some new models, such as HetGNN, NSHE, GTN, MAGNN, RSHN.
