.. _pipeline-task:

Task
======

The task means downstream task. So it is related to dataset closely.

Included Object:
--------------------

* :ref:`api-dataset`,
containing the heterogeneous graph as DGLGraph, as well as node/edge features and additional dataset-specific information.
* Evaluation Metric, a callable object which takes in the prediction and the ground truth label, and returns an output.

   * Currently we will implement evaluation metrics for node classification and link prediction.
* Training Loss, a callable object which takes in the prediction and the ground truth as well as any other necessary information and returns an output. It must be differentiable.

   * Ranking loss for self-supervision, or contrastive learning losses, all fall into this category.
* Training-validation-test Split.

Method:
-------

* get_graph()

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

How to build a new trainerflow
---------------------------------

**First step**

We should create a class your_task that inherits
the `BaseTask <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/tasks/base_task.py>`_
and register the model with @register_task(str).
We give the task recommendation as an example.

.. code-block:: python

    from openhgnn.tasks import BaseTask, register_task
    @register_task('recommendation')
    class Recommendation(BaseTask):
        ...

*Note*

We decorate the func evaluate() with @abstractmethod. So the evaluate() must be overridden, or the your_task cannot be instantiated.

**Finally**

We should fill the dict SUPPORTED_TASKS in `task/init.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/tasks/__init__.py>`_