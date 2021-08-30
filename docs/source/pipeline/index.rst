A Pipeline of OpenHGNN
============================

We define three components:
:ref:`pipeline-models`,
:ref:`pipeline-task`,
:ref:`pipeline-trainerFlow`

* For a given paper,

   * For unsupervised model, the model and the trainerflow are specific cause the loss calculation is fixed.So given a model name, we can identify the model and trainerflow, user need also specify the task name.
   * For other model, the model is fixed and the trainerflow is changeable according to task. User must specify the model and the task, and the trainerflow will be determined by the task. For example, RGCN on the node classification task will use the entity classification flow and RGCN on the link prediction task will use the dist_mult flow.

.. image:: .._static.pipeline_img.png

.. toctree::
   :maxdepth: 2
   :titlesonly:

   overview
   task
   model
   trainerFlow