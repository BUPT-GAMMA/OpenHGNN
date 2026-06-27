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
   get_started/model_overview
   get_started/task_overview
   get_started/reproduce_model

.. toctree::
   :maxdepth: 1
   :caption: Release Notes
   :hidden:
   :glob:

   release/index

.. toctree::
   :maxdepth: 1
   :caption: Models
   :hidden:
   :glob:

   models/index

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


OpenHGNN is an open-source toolkit for heterogeneous graph neural networks
based on `DGL [Deep Graph Library] <https://github.com/dmlc/dgl>`_ and
`PyTorch <https://pytorch.org/>`_. It provides reusable models, datasets,
tasks, trainer flows, and experiment entry points for heterogeneous graph
research and applications.

Key Features
--------------

* Easy-to-use experiment entry points through ``Experiment(...)`` and
  ``main.py``.
* Extensible model, dataset, task, and trainerflow registries for new research
  contributions.
* DGL-first graph construction, sampling, and message passing for heterogeneous
  graph workloads.
* OpenHGNN v0.9 adds 10 model contributions and expands task coverage with
  ``node_regression``.
* Registered models, tasks, datasets, and trainer flows can be inspected from
  the command line.

Quick checks
--------------

.. code:: bash

   openhgnn env
   openhgnn list models
   openhgnn list tasks
   openhgnn validate-registry

Contact Us
-------------
If you have any question, please submit issues or contact us: tyzhao@bupt.edu.cn
