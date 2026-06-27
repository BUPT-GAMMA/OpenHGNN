SEHTGNN
=======

* Paper: Simple and Efficient Heterogeneous Temporal Graph Neural Network,
  NeurIPS 2025
* Registered model name: ``SEHTGNN``
* Tasks: ``node_classification``, ``link_prediction``, ``node_regression``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - Preprocessed temporal graph datasets from the SE-HTGNN paper:
       OGBN-MAG, Aminer, YELP, and COVID-19.
   * - Preprocessing
     - Uses temporal snapshots. COVID-19 contains state and county nodes from
       2020-05-01 to 2021-02-28.
   * - Commands
     - ``python main.py -m SEHTGNN -t node_classification -d sehtgnn_yelp -g 0 --use_best_config``;
       ``python main.py -m SEHTGNN -t node_regression -d sehtgnn_covid -g 0 --use_best_config``;
       ``python main.py -m SEHTGNN -t link_prediction -d sehtgnn_ogbn -g 0 --use_best_config``.
   * - Metrics
     - Link prediction: AUC and AP. Node classification: Macro-F1 and Recall.
       Node regression: MAE and RMSE.
   * - Expected result
     - OGBN-MAG AUC/AP: 93.08/94.21. Aminer AUC/AP: 90.45/90.82.
       YELP Macro-F1/Recall: 42.27/43.11. COVID MAE/RMSE: 499/1064.
   * - Hardware/runtime
     - Reported on GeForce RTX 3090.
   * - Seed
     - Uses config defaults; ``seed`` should be recorded for each run.

Implementation Notes
--------------------

SEHTGNN uses LLM-enhanced type prompts, simplified spatial aggregation, and
temporal fusion with dynamic attention.

