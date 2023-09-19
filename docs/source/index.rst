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

   get_started/index
   get_started/quick_start

.. toctree::
   :maxdepth: 2
   :caption: Advanced Materials
   :hidden:
   :glob:

   advanced_materials/index
   advanced_materials/developer_guide

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:
   :glob:

   api/dataset
   api/task
   api/layer
   api/model
   api/trainerflow


This is an open-source toolkit for Heterogeneous Graph Neural Network(OpenHGNN) based on `DGL [Deep Graph Library] <https://github.com/dmlc/dgl>`_ and `PyTorch <https://pytorch.org/>`_.
We integrate SOTA models of heterogeneous graph.

Key Features
--------------

* Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and datasets using `optuna <https://optuna.org/>`_ which is a hyperparameter optimization framework.
* Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
* Efficiency: The backend dgl provides efficient APIs.

Contract Us
-------------
If you have any question, please submit issues or contact us: tyzhao@bupt.edu.cn
