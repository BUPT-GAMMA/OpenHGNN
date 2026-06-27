HGDL
====

* Paper: Heterogeneous Graph Label Distribution Learning, NeurIPS 2024
* Registered model name: ``HGDL``
* Task: ``node_classification`` for label distribution learning
* Trainerflow: ``HGDL_trainer``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - Upstream HGDL datasets for DBLP and ACM, loaded from
       ``openhgnn/dataset/DBLP/`` and ``openhgnn/dataset/ACM/``.
   * - Preprocessing
     - Ported from the upstream HGDL repository. Builds metapath graphs and
       uses a seeded 40/10/50 train/validation/test split on author nodes.
   * - Command
     - ``python main.py -m HGDL -t node_classification -d dblp4HGDL -g 0 --use_best_config``
   * - Alternative command
     - ``python main.py -m HGDL -t node_classification -d acm4HGDL -g 0 --use_best_config``
   * - Metric
     - KL, Cosine, Canberra, Chebyshev, Clark, Intersection.
   * - Expected result
     - ``dblp4HGDL``: KL 0.0704, Cosine 0.0198, Canberra 2.8117,
       Chebyshev 0.0637, Clark 1.6377, Intersection 0.9356.
       ``acm4HGDL``: KL 0.7514, Cosine 0.1962, Canberra 13.1013,
       Chebyshev 0.3516, Clark 3.4767, Intersection 0.6414.
   * - Hardware/runtime
     - GPU is recommended. release validation did not measure full runtime.
   * - Seed
     - ``seed = 0`` in ``config.ini``.

Implementation Notes
--------------------

HGDL uses ``dgl.nn.GraphConv``, ``edge_softmax``, ``apply_edges``, and
``update_all``. The loss combines KL divergence with an Omega consistency
regularizer.

