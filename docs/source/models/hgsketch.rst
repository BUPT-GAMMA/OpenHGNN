HGSketch
========

* Paper: Heterogeneous Graph Embedding Made More Practical, SIGIR 2025
* Registered model name: ``HGSketch``
* Task scope: graph-level heterogeneous graph representation
* Trainerflow: ``HGSketch_trainer``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - Model-specific graph datasets. A stable OpenHGNN graph-classification
       dataset fixture is still needed for a release end-to-end smoke.
   * - Preprocessing
     - Converts DGL heterogeneous graphs to a homogeneous NetworkX view,
       extracts simplices, builds Hodge Laplacians, and applies hashing.
   * - Command
     - ``python main.py -m HGSketch -t graph_classification -d <graph_dataset> -g -1``
   * - Metric
     - Downstream graph classification metric from the linear classifier.
   * - Expected result
     - Paper-level graph classification result is not yet part of an OpenHGNN
       end-to-end task. Release validation is component-level through
       ``tests/test_hgsketch.py``.
   * - Hardware/runtime
     - CPU-capable component pipeline. Runtime depends on graph size and
       simplex extraction cost.
   * - Seed
     - ``seed = 0`` in ``config.ini``.

Implementation Notes
--------------------

HGSketch is non-parametric and intentionally uses NetworkX/scipy at the
algorithm boundary. OpenHGNN still needs a formal ``graph_classification`` task
or a documented component-test scope for release validation.

Release Limitation
------------------

OpenHGNN does not yet provide a general ``graph_classification`` task registry
entry. HGSketch should therefore remain validated by component tests until a
formal graph-classification task and tiny fixture dataset are added.
