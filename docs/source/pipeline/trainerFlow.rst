.. _pipeline-trainerFlow:

TrainerFlow
=============

A trainerflow is an abstraction of a predesigned workflow that trains and evaluate a model on a given dataset for a specific use case. It must contain a unique training mechanism involving loss calculation and a specific sampler(sample something used in loss calculation) .

Once we select the model and the task, the func `get_trainerflow <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/start.py>`_ will help us select the trainerflow. So the customized trainerflow needed be added in this func.

Included Object:
-------------------

* task : :ref:`pipeline-task`
* model : :ref:`pipeline-models` (built through given args.model)
* optimizer : torch.optim.Optimizer
* dataloader(if mini_batch_flag is True) :

   * torch.utils.data.DataLoader
   * `dgl.dataloading <https://docs.dgl.ai/en/latest/api/python/dgl.dataloading.html#>`_

Method:
---------

* train()

   * decorated with @abstractmethod, so it must be overridden.
* _full_train_setp()

   * train with a full_batch graph
* _mini_train_step()

   * train with a mini_batch seed nodes graph
* _test_step()

   * evaluate in training/validation/testing

Supported trainerflow
----------------------

* Node classification flow

   * Supported Model: HAN/MAGNN/GTN...
   * The task: node classification

      * The task.dataset must include the splited[train/valid/test.] mask.
   * The sampler in this flow is supported by dgl.dataloading.
   * The flow is the most common in the GNNs cause most GNNs model are involved in the task semi-supervised node classification. Here the task is to classify the nodes of HIN(Heterogeneous Information Network).
   * Note: we will set the args.out_dim with num_classes if they are not equivalent.

* Dist Mult

   * The same with entity classification except that it is used for link prediction.
   * Supported Model: RGCN/CompGCN/RSHN
   * Supported Task: link prediction
* HetGNN trainerflow
* NSHE trainerflow
