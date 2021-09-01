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

   * Supported Model: HAN/MAGNN/GTN
   * The task: node classification

      * The task.dataset must include the splited[train/valid/test.] mask.
      * The task.dataset will give the value of input dimension to the args.in_dim.
   * The sampler in this flow is supported by dgl.dataloading.
   * The flow is the most common in the GNNs cause most GNNs model are involved in the task semi-supervised node classification. Here the task is to classify the nodes of HIN(Heterogeneous Information Network).
   * Note: we will set the args.out_dim with num_classes if they are not equivalent.
* Entity_classification_flow

   * The same with node classification except that it is used for knowledge graph.
   * Supported Model: RGCN/CompGCN/RSHN
   * Supported Task: node classification
* Dist Mult

   * The same with entity classification except that it is used for link prediction.
   * Supported Model: RGCN/CompGCN/RSHN
   * Supported Task: link prediction
* HetGNN trainerflow
* NSHE trainerflow

How to build a new trainerflow
-------------------------------

**First step**

We should create a class your_trainerflow that inherits
the `BaseFlow <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/trainerflow/base_flow.py>`_
and register the trainerflow with @register_trainer(str).

.. code-block:: python

    from openhgnn.trainerflow import BaseFlow, register_flow
    @register_flow('demo_trainer')
    class DemoTrainer(BaseFlow):
        ...

**Second step**

We decorate the func train() with @abstractmethod. So the train() must be overridden, or the your_trainerflow cannot be instantiated.
Besides train(), the init() and _test_step() should both be implement.
One of the _full_train_step() and _mini_train_step() must be implemented at least.


.. code-block:: python

    ...
    class DemoTrainer(BaseFlow):
        def __init(self, args):
            self.task = build_task(args)
            self.hg = self.task.get_graph().to(self.device)
            self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        def train(self,):
            for epoch in epoch_iter:
                self._full_train_step()
                self._full_test_step()

        def _full_train_step(self, ):
            self.model.train()
            logits = self.model(self.hg)[self.category]
            loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def _full_test_step(self, mode=None, logits=None):
            self.model.eval()
            with torch.no_grad():
                loss = self.loss_fn(logits[mask], self.labels[mask]).item()
                metric = self.task.evaluate(pred, name=self.metric, mask=mask)
                return metric, loss

**Finally**

We should fill the dict SUPPORTED_FLOWS in `trainerflow/init.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/trainerflow/__init__.py>`_
