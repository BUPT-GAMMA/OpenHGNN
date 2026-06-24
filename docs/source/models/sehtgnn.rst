SEHTGNN
-------

SEHTGNN (Simple and Efficient Heterogeneous Temporal Graph Neural Network) is a
heterogeneous temporal graph model covering link prediction, node
classification, and node regression.

Paper and Code
--------------

* Paper: https://doi.org/10.48550/arXiv.2510.18467
* Upstream code: https://github.com/wangyili00/SE-HTGNN

How to Run
----------

.. code-block:: bash

   python main.py -m SEHTGNN -t node_classification -d sehtgnn_yelp -g 0 --use_best_config
   python main.py -m SEHTGNN -t node_regression -d sehtgnn_covid -g 0 --use_best_config
   python main.py -m SEHTGNN -t link_prediction -d sehtgnn_ogbn -g 0 --use_best_config
   python main.py -m SEHTGNN -t link_prediction -d sehtgnn_aminer -g 0 --use_best_config

Use CPU by setting ``-g -1``.

Datasets and Tasks
------------------

.. list-table::
   :header-rows: 1

   * - Dataset
     - Task
     - Metric
   * - ``sehtgnn_ogbn``
     - link prediction
     - AUC / AP
   * - ``sehtgnn_aminer``
     - link prediction
     - AUC / AP
   * - ``sehtgnn_yelp``
     - node classification
     - Macro-F1 / Recall
   * - ``sehtgnn_covid``
     - node regression
     - MAE / RMSE

Reproduction Result
-------------------

.. list-table::
   :header-rows: 1

   * - Dataset
     - Metric
     - Paper
     - OpenHGNN
   * - OGBN-MAG
     - AUC
     - 92.71
     - 93.08
   * - OGBN-MAG
     - AP
     - 93.13
     - 94.21
   * - Aminer
     - AUC
     - 90.03
     - 90.45
   * - YELP
     - Macro-F1
     - 44.24
     - 42.27
   * - COVID-19
     - MAE
     - 497
     - 499
   * - COVID-19
     - RMSE
     - 1069
     - 1064

Implementation Notes
--------------------

SEHTGNN adds OpenHGNN ``node_regression`` support for continuous-valued node
targets. The task should be registered in ``openhgnn/tasks/__init__.py`` and
covered by end-to-end smoke tests.

Smoke Tests
-----------

The release smoke should cover all three supported task types:

.. code-block:: bash

   python main.py -m SEHTGNN -t node_classification -d sehtgnn_yelp -g -1 --use_best_config
   python main.py -m SEHTGNN -t node_regression -d sehtgnn_covid -g -1 --use_best_config
   python main.py -m SEHTGNN -t link_prediction -d sehtgnn_ogbn -g -1 --use_best_config
