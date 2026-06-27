HGEN
====

* Paper: Heterogeneous Graph Ensemble Networks, IJCAI 2025
* Registered model name: ``HGEN``
* Task: ``node_classification``
* Trainerflow: ``HGEN_trainer``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - ``dblp4GTN``, ``acm4GTN``, and ``imdb4GTN`` from the OpenHGNN/DGL
       dataset mirror.
   * - Preprocessing
     - Datasets are downloaded automatically on first use. HGEN extracts
       metapath reachable graphs with DGL.
   * - Command
     - ``python main.py -m HGEN -t node_classification -d dblp4GTN -g 0``
   * - Metric
     - Macro-F1 and Micro-F1.
   * - Expected result
     - DBLP: Macro-F1 88.15, Micro-F1 89.09. IMDB: Macro-F1 56.31,
       Micro-F1 57.51. ACM: Macro-F1 89.62, Micro-F1 89.46.
   * - Hardware/runtime
     - GPU is recommended. release validation did not measure full runtime.
   * - Seed
     - ``seed = 0`` in ``config.ini``.

Implementation Notes
--------------------

HGEN uses ``dgl.metapath_reachable_graph`` and ``dgl.nn.GraphConv``. Final
prediction sums logits from multiple metapath-specific GCN encoders.

