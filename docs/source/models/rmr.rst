RMR
===

* Paper: Reserving-Masking-Reconstruction Model for Self-Supervised
  Heterogeneous Graph Representation, KDD 2024
* Registered model name: ``RMR``
* Task: ``node_classification``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - ``acm4RMR``, ``imdb4RMR``, and ``aminer4RMR`` from the OpenHGNN dataset
       mirror.
   * - Preprocessing
     - Dataset-specific RMR loading and split ratios ``1,5,10,20``.
   * - Command
     - ``python main.py -m RMR -t node_classification -d acm4RMR -g 0``
   * - Metric
     - Macro-F1, Micro-F1, and AUC.
   * - Expected result
     - ``acm4RMR`` split 20: Macro-F1 87.48, Micro-F1 87.50, AUC 94.24.
       ``imdb4RMR`` split 20: Macro-F1 48.13, Micro-F1 48.04, AUC 63.63.
       ``aminer4RMR`` split 20: Macro-F1 87.87, Micro-F1 90.56.
   * - Hardware/runtime
     - GPU is recommended. Some paper datasets were not included because of
       missing labels or large image partitions.
   * - Seed
     - ``seed = 0`` in dataset-specific config sections.

Implementation Notes
--------------------

RMR is a self-supervised representation method followed by node-classification
evaluation. Do not commit generated embeddings or runtime outputs.

