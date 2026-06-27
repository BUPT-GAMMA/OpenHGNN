Model Overview
==============

Use this page to find the model name, task, and dataset name that should be
passed to ``Experiment(...)`` or ``main.py``. To inspect the local installation,
run:

.. code:: bash

   openhgnn list models
   openhgnn validate-registry --format json

For a step-by-step reproduction workflow, including environment setup, config
selection, seed recording, logs, and README requirements, see
:doc:`reproduce_model`.

For the release-level model table, venues, reproduction entries, and remaining
documentation notes, see :doc:`../release/v0.9`.

OpenHGNN v0.9 Model Updates
---------------------------

OpenHGNN v0.9 adds 10 model contributions. ``HERO`` and ``HERO_homo`` are two
registered model names, but they are counted as one model contribution because
they come from the same HERO paper family.

.. list-table::
   :header-rows: 1

   * - Contribution
     - Registered model name
     - Venue
     - Main task
     - Example dataset
   * - HGDL
     - ``HGDL``
     - NeurIPS 2024
     - ``node_classification``
     - ``dblp4HGDL``, ``acm4HGDL``
   * - HGEN
     - ``HGEN``
     - IJCAI 2025
     - ``node_classification``
     - ``dblp4GTN``
   * - HGSketch
     - ``HGSketch``
     - SIGIR 2025
     - graph-level representation
     - model-specific graph datasets
   * - HGOT
     - ``HGOT``
     - ICML 2025
     - ``node_classification``
     - see model README
   * - RMR
     - ``RMR``
     - KDD 2024
     - ``node_classification``
     - ``acm4RMR``, ``aminer4RMR``, ``imdb4RMR``
   * - HERO
     - ``HERO``, ``HERO_homo``
     - ICLR 2024
     - ``node_classification``
     - ``Yelp4HERO``, ``photo4HERO``
   * - SEHTGNN
     - ``SEHTGNN``
     - NeurIPS 2025
     - ``node_classification``, ``link_prediction``, ``node_regression``
     - ``sehtgnn_yelp``, ``sehtgnn_covid``
   * - HTGformer
     - ``HTGformer``
     - SIGIR 2025
     - ``link_prediction``, ``node_classification``, ``node_regression``
     - ``ogbn_mag4HGformer``, ``covid4HGformer``
   * - HCMGNN
     - ``HCMGNN``
     - IJCAI 2024
     - ``recommendation``
     - ``GMD4HCMGNN``
   * - RelGT
     - ``RelGT``
     - arXiv 2025
     - RelBench supervised prediction
     - ``rel-f1``

Example Commands
----------------

.. code:: bash

   python main.py -m HGDL -d dblp4HGDL -t node_classification -g -1 --use_best_config
   python main.py -m SEHTGNN -d sehtgnn_covid -t node_regression -g -1 --use_best_config
   python main.py -m HTGformer -d ogbn_mag4HGformer -t link_prediction -g -1
   python main.py -m HCMGNN -d GMD4HCMGNN -t recommendation -g -1

v0.9 Reproduction Table
-----------------------

The release notes maintain the authoritative v0.9 reproduction table with
model, venue, task, dataset, command, and remaining documentation notes:
:doc:`../release/v0.9`.

Model Documentation
-------------------

During the v0.9 transition, detailed reproduction notes for new models may
still live in the legacy ``openhgnn/output/<MODEL>/README.md`` location. The
release documentation keeps this overview as the stable entry point, and model
README files can be migrated into ``docs/source`` in a later documentation-only
cleanup.
