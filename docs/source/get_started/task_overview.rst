Task Overview
=============

OpenHGNN separates task logic from model and trainerflow logic. A task defines
the graph, labels, split, loss, and evaluation interface used by an experiment.

To inspect registered tasks in the current environment:

.. code:: bash

   openhgnn list tasks
   openhgnn validate-registry --format json

Common Tasks
------------

.. list-table::
   :header-rows: 1

   * - Task name
     - Purpose
     - Typical output
     - Example model
   * - ``node_classification``
     - Predict discrete labels for nodes.
     - Micro-F1, Macro-F1, accuracy
     - ``RGCN``, ``HGEN``, ``HERO``
   * - ``link_prediction``
     - Predict edges or relations.
     - AUC, AP, MRR, Hits@K
     - ``HTGformer``, ``SEHTGNN``
   * - ``recommendation``
     - Rank or predict user-item interactions.
     - Recall, NDCG, MRR
     - ``HCMGNN``
   * - ``node_regression``
     - Predict continuous-valued node targets.
     - MAE, RMSE, MSE
     - ``SEHTGNN``, ``HTGformer``

Node Regression
---------------

OpenHGNN v0.9 adds ``node_regression`` for datasets where node labels are
continuous values instead of class IDs. The task follows the same public entry
points as other tasks:

.. code:: python

   from openhgnn import Experiment

   experiment = Experiment(
       model='SEHTGNN',
       dataset='sehtgnn_covid',
       task='node_regression',
       gpu=-1,
       max_epoch=1,
   )
   experiment.run()

Equivalent command-line usage:

.. code:: bash

   python main.py -m SEHTGNN -d sehtgnn_covid -t node_regression -g -1 --use_best_config

Input and output contract
~~~~~~~~~~~~~~~~~~~~~~~~~

``node_regression`` expects a DGL graph, train/validation/test splits, and
continuous labels returned by the dataset's ``get_labels()`` method. General
node-regression datasets use the split indices returned by ``get_split()``.
Model-specific temporal or sampled datasets may provide split-specific label
graphs, but the trainerflow should still expose the public OpenHGNN task name
``node_regression``.

Loss and metrics
~~~~~~~~~~~~~~~~

The general task uses ``torch.nn.MSELoss`` as its loss function. The default
evaluation metric is ``rmse``. Supported metric names are:

.. list-table::
   :header-rows: 1

   * - Metric
     - Meaning
     - When to use
   * - ``rmse``
     - Root mean squared error
     - Default regression comparison metric.
   * - ``mae``
     - Mean absolute error
     - Robust error reporting and temporal regression datasets.
   * - ``mse``
     - Mean squared error
     - Loss-aligned reporting.

SEHTGNN's node-regression path reports RMSE, MAE, and MSE. HTGformer's COVID
regression path reports MAE through its model-specific trainer.

Supported v0.9 models
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Model
     - Dataset example
     - Command
   * - ``SEHTGNN``
     - ``sehtgnn_covid``
     - ``python main.py -m SEHTGNN -d sehtgnn_covid -t node_regression -g 0 --use_best_config``
   * - ``HTGformer``
     - ``covid4HGformer``
     - ``python main.py -m HTGformer -d covid4HGformer -t node_regression -g 0``

For a full reproduction record, include the environment, seed, command, metric,
and log path described in :doc:`reproduce_model`.

Contribution Notes
------------------

When adding a new task or extending an existing task, keep model-specific logic
out of shared task files unless the behavior is generally reusable. Model-only
special cases should be documented in the model README and isolated in the
model's trainerflow or dataset module.
