HTGformer
=========

* Paper: HTGformer: Heterogeneous Temporal Graph Transformer, SIGIR 2025
* Registered model name: ``HTGformer``
* Tasks: ``link_prediction``, ``node_classification``, ``node_regression``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - ``ogbn_mag4HGformer``, ``aminer4HGformer``, ``yelp4HGformer``, and
       ``covid4HGformer``. Files are downloaded automatically to
       ``openhgnn/dataset/``.
   * - Preprocessing
     - Uses registered OpenHGNN datasets and model-specific temporal samples.
       The merged result uses w/o_LLM mode with learnable type embeddings.
   * - Commands
     - ``python main.py -m HTGformer -d ogbn_mag4HGformer -t link_prediction -g 0``;
       ``python main.py -m HTGformer -d yelp4HGformer -t node_classification -g 0``;
       ``python main.py -m HTGformer -d covid4HGformer -t node_regression -g 0``.
   * - Metrics
     - Link prediction: AUC and AP. Node classification: Macro-F1 and Recall.
       Node regression: MAE.
   * - Expected result
     - OGBN-MAG AUC/AP: 94.61/93.98. Aminer AUC/AP: 88.41/82.99. YELP
       Macro-F1/Recall: 35.91/40.91. COVID MAE: 511.59.
   * - Hardware/runtime
     - Reported on NVIDIA GeForce RTX 4090.
   * - Seed
     - ``num_repeats = 5`` in ``config.ini``. Record seed for each run.

Implementation Notes
--------------------

The full paper result uses LLaMA3 type encoding. The merged OpenHGNN result is
the w/o_LLM variant, explaining the YELP performance gap noted in the README.

