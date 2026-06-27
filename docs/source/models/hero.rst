HERO and HERO_homo
==================

* Paper: Self-Supervised Heterogeneous Graph Learning: A Homophily and
  Heterogeneity View, ICLR 2024
* Registered model names: ``HERO`` and ``HERO_homo``
* Task: ``node_classification``

HERO Reproduction
-----------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - ``ACM4HERO``, ``Aminer4HERO``, ``DBLP4HERO``, and ``Yelp4HERO`` from
       Hugging Face-hosted archives.
   * - Preprocessing
     - Dataset classes download raw files, build DGL heterographs, and cache
       processed files.
   * - Command
     - ``python main.py -m HERO -d Yelp4HERO -t node_classification -g 0``
   * - Metric
     - Micro-F1 and Macro-F1.
   * - Expected result
     - ACM: Micro-F1 0.91501, Macro-F1 0.91595. Aminer: 0.72020, 0.66705.
       DBLP: 0.93588, 0.92678. Yelp: 0.92354, 0.92677.
   * - Hardware/runtime
     - GPU-only is recommended by the merged model README.
   * - Seed
     - Dataset-specific config sections in ``config.ini``.

HERO_homo Reproduction
----------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - DGL built-in datasets and OGB datasets, including ``photo4HERO``,
       ``computers4HERO``, ``cs4HERO``, and ``physics4HERO``.
   * - Preprocessing
     - Converts homogeneous graphs into HERO feature views and feature-distance
       matrices, then caches processed files.
   * - Command
     - ``python main.py -m HERO_homo -d photo4HERO -t node_classification -g 0``
   * - Metric
     - Micro-F1 and Macro-F1.
   * - Expected result
     - Photo: Micro-F1 0.93343, Macro-F1 0.92021. Computers: 0.88533,
       0.86264. CS: 0.92754, 0.87665. Physics: 0.95925, 0.94587.
   * - Hardware/runtime
     - GPU is recommended.
   * - Seed
     - Dataset-specific config sections in ``config.ini``.

