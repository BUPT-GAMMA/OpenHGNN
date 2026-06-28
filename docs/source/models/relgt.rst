RelGT
=====

* Paper: Relational Graph Transformer, arXiv 2025
* Registered model name: ``RelGT``
* Task: RelBench supervised prediction through ``RelGT_trainer``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - RelBench datasets. Default OpenHGNN config uses ``rel-f1`` and
       ``driver-top3``.
   * - Preprocessing
     - Downloads RelBench data, materializes token caches, and reuses
       precomputed HDF5 neighbor-token files on later runs.
   * - Command
     - ``python main.py -m RelGT -t link_prediction -d rel-f1 -g 0 --use_best_config``
   * - Metric
     - RelBench task metric for the configured supervised prediction task.
   * - Expected result
     - Full RelBench result is pending an external benchmark run with optional
       dependencies installed. Current release validation covers registry
       wiring and local RelGT components.
   * - Hardware/runtime
     - Depends on RelBench cache generation and optional dependencies. Runtime
       was not measured in release validation.
   * - Seed
     - ``seed = 42`` in ``config.ini``.

Optional Dependencies
---------------------

RelGT requires dependencies that are not part of the core OpenHGNN environment:

.. code:: bash

   pip install relbench sentence-transformers h5py einops

Release Limitation
------------------

RelGT requires RelBench data preparation and cached table features before the
full benchmark can run. Keep it out of the default CI smoke matrix until a
small cached RelBench fixture is available in CI.
