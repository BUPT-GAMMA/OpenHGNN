.. _pipeline-task:

Task
======

The task means downstream task. So it is related to dataset closely.

Included Object:
--------------------

* :ref:`api-dataset`, containing the heterogeneous graph as DGLGraph, as well as node/edge features
  and additional dataset-specific information.

* Evaluation Metric, a callable object which takes in the prediction and the ground truth label, and returns an output.

   * Currently we will implement evaluation metrics for node classification and link prediction.
* Training Loss, a callable object which takes in the prediction and the ground truth as well as any other necessary information and returns an output. It must be differentiable.

   * Ranking loss for self-supervision, or contrastive learning losses, all fall into this category.
* Training-validation-test Split.

Method:
-------

* ..func:``get_graph()``

   * return the DGLGraph
* get_loss_fn()

   * train with a full_batch graph
* evaluate()

   * Evaluate Metric
* get_labels()

   * return labels

Supported task
------------------

* Node classification
* Link prediction
