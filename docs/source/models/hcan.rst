HCAN
====

HCAN implements the two models proposed in *Effective and Scalable
Heterogeneous Graph Neural Network Framework with Convolution-oriented
Attention* (ICDE 2025): Vanilla HCAN (``HCAN``) and Decoupled HCAN
(``DHCAN``, also registered as ``D-HCAN``).

Model correspondence
--------------------

``HCAN`` follows Algorithm 1. Each layer builds the zero-to-``max_hop``
convolutional tokens, computes relation-aware convolution-oriented attention,
and combines the weighted and counterweight payloads.

``DHCAN`` follows Equations 13--15. It enumerates only type-compatible
relation paths, performs relation-wise mean propagation and non-parametric
attention once without gradients, and caches those channel embeddings. The
trainable channel projection and semantic fusion MLPs then reuse the cache at
every epoch. Call ``model.clear_cache()`` before reusing a model with another
graph or another feature set. ``cache_device`` controls where the precomputed
channels are stored and defaults to ``cpu``. The dedicated node-classification
trainer retains channels only for the target node type to limit host memory.

Quick start
-----------

V-HCAN node classification on HGB ACM:

.. code-block:: bash

   python main.py -m HCAN -t node_classification -d HGBn-ACM -g 0

D-HCAN node classification on OGB-MAG:

.. code-block:: bash

   python main.py -m DHCAN -t node_classification -d ogbn-mag -g 0

The first D-HCAN forward pass includes precomputation. Report preprocessing
time separately from training time per epoch when comparing with Table VIII.
The paper reports five-run mean validation/test accuracy of 56.44/54.81 on
OGB-MAG; reproducing that table requires five complete runs with the official
temporal split. The complete OGB-MAG graph also requires substantial host
memory during parameter-free precomputation.

Configuration
-------------

The default options are in ``openhgnn/config.ini`` under ``[HCAN]`` and
``[DHCAN]``. Both implementations consume the complete heterograph. The
D-HCAN trainer performs its parameter-free graph propagation in full-batch
mode on CPU, then trains cached target-node representations in batches.
``hidden_dim`` must be divisible by ``2 * num_heads`` for V-HCAN because the
weighted and counterweight payloads each produce half of the output, while
D-HCAN uses a single decoupled propagation stage (``num_layers = 1``).

Tests
-----

.. code-block:: bash

   python -m pytest -o addopts= tests/test_hcan_model.py tests/test_hcan_config.py tests/test_node_classification_early_stop.py -q
