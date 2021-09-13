Basic Usage
==========================

Run experiments
------------------
Running an existing baseline model on an existing benchmark :ref:`task <api_dataset>`

.. code:: bash

    >>> python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config
    usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config] [--use_hpo]

*optional arguments*:
    - ``--model MODEL``, ``-m MODEL`` name of models
    - ``--task TASK``, ``-t TASK`` name of task
    - ``--dataset DATASET``, ``-d DATASET``	name of datasets
    - ``--gpu GPU``, ``-g GPU``	controls which gpu you will use. If you do not have gpu, set -g -1.
    - ``--use_best_config``	use_best_config means you can use the best config OpeHGNN has found in the dataset with the model.If you want to set the different hyper-parameter, modify the `openhgnn.config.ini <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/config.ini>`_ manually.
    - ``--use_hpo`` Besides use_best_config, we use hyper-parameter optimization from optuna. And refer them to the section below for details.

e.g.:

.. code:: bash

    python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config

.. note::

If you are interested in some model, please refer to the `models list <https://github.com/BUPT-GAMMA/OpenHGNN#models>`_.

Hyper-parameter optimization
-------------------------------
Running an experiment with optuna

.. code:: bash

    >>> python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config --use_hpo

OpenHGNN will determine hyperparameters in the following order:

- If ``--use_hpo`` is enabled, search for the best hyperparameter by optuna. This is controlled by :func:``func_search`` in ./openhgnn/auto/hpo.py.
Please refer `here <https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/auto>`_ for more details.
- Otherwise, if --use_best_config is enabled, load the best hyperparameters built within OpenHGNN. The configurations are in ./openhgnn/utils/best_config.py.
- Otherwise, load the hyperparameters in ./openhgnn/config.ini.

Evaluate a new dataset
=======================
You can specify your own dataset if necessary. In this section we use HGBn-ACM as an example for node classification dataset.

How to build a new dataset
---------------------------

**First step: Process dataset**

We give a `demo <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/debug/HGBn-ACM2dgl.py>`_ to process the HGBn-ACM.
First, download the HGBn-ACM from the `Link <https://www.biendata.xyz/hgb/#/datasets>`_.
After that, we process it as a `dgl.heterograph <https://docs.dgl.ai/en/latest/guide/graph-heterogeneous.html#guide-graph-heterogeneous>`_.

The following code snippet is an example for creating a heterogeneous graph in DGL.

.. code:: python

    >>> import dgl
    >>> import torch as th

    >>> # Create a heterograph with 3 node types and 3 edges types.
    >>> graph_data = {
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
    ... }
    >>> g = dgl.heterograph(graph_data)
    >>> g.ntypes
    ['disease', 'drug', 'gene']
    >>> g.etypes
    ['interacts', 'interacts', 'treats']
    >>> g.canonical_etypes
    [('drug', 'interacts', 'drug'),
     ('drug', 'interacts', 'gene'),
     ('drug', 'treats', 'disease')]

We recommend the feature name set by the ``'h'``.

.. code:: python

    >>> g.nodes['drug'].data['h'] = th.ones(3, 1)

DGL provides :func:`dgl.save_graphs` and :func:`dgl.load_graphs` respectively for saving
heterogeneous graphs in binary format and loading them from binary format.
So we can use `dgl.save_graphs <https://docs.dgl.ai/en/latest/generated/dgl.save_graphs.html#>`_ to store graph into the disk.

.. code:: python

    >>> dgl.save_graphs("demo_graph.bin", g)

**Second step: Add extra information**

We can get a binary format named *demo_graph.bin* after first step, and we should move it into the directory *openhgnn/dataset/*.
But for now, it is not a complete dataset.
We should specify some important information in the `NodeClassificationDataset.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/dataset/NodeClassificationDataset.py#L145>`_

For example, we should set the *category*, *num_classes* and *multi_label* (if necessary) with ``"paper"``, ``3``, ``True``.
More infos, refer to :ref:`Base Node Classification Dataset <api-base-node-dataset>`.

.. code:: python

    if name_dataset == 'demo_graph':
        data_path = './openhgnn/dataset/demo_graph.bin'
        g, _ = load_graphs(data_path)
        g = g[0].long()
        self.category = 'author'
        self.num_classes = 4
        self.multi_label = False

**Third step: optional**

We can use demo_graph as our dataset name to evaluate a existing model.

.. code:: bash

    python main.py -m GTN -d demo_graph -t node_classification -g 0 --use_best_config


If you have another dataset name, you should also modify the `build_dataset <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/dataset/__init__.py>`_.

Apply a new model
====================
In this section, we will create a model named RGAT,
which is not in our openhgnn.models.

How to build a new model
--------------------------
**First step: Register model**

We should create a class your_model that inherits the :ref:`Base Model <api-model>` .
and register the model with @register_model(str).

.. code-block:: python

    from openhgnn.models import BaseModel, register_model
    @register_model('RGAT')
    class RGAT(BaseModel):
        ...


**Second step: Implement functions**

We must implement the classmethod build_model_from_args , other functions like __init__(), forward() and so on.

.. code-block:: python

    ...
    class RGAT(BaseModel):
        @classmethod
        def build_model_from_args(cls, args, hg):
            return cls(in_dim=args.hidden_dim,
                       out_dim=args.hidden_dim,
                       h_dim=args.out_dim,
                       etypes=hg.etypes,
                       num_heads=args.num_heads,
                       dropout=args.dropout)

        def __init__(self, in_dim, out_dim, h_dim, etypes, num_heads, dropout):
            super(RGAT, self).__init__()
            self.rel_names = list(set(etypes))
            self.layers = nn.ModuleList()
            self.layers.append(RGATLayer(
                in_dim, h_dim, num_heads, self.rel_names, activation=F.relu, dropout=dropout))
            self.layers.append(RGATLayer(
                h_dim, out_dim, num_heads, self.rel_names, activation=None))
            return

        def forward(self, hg, h_dict=None):
            if hasattr(hg, 'ntypes'):
                # full graph training,
                for layer in self.layers:
                    h_dict = layer(hg, h_dict)
            else:
                # minibatch training, block
                for layer, block in zip(self.layers, hg):
                    h_dict = layer(block, h_dict)
            return h_dict

Here we do not give the implement the RGATLayer, you can get more from `RGATLayer <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/models/RGAT.py>`_.

.. note::

    In OpenHGNN, we preprocess the feature of dataset outside of model.
    Specifically, we use a linear layer with bias for each node type to map all node features to a shared feature space.
    So the parameter *h_dict* of *forward()* in model is not original, and your model need not feature preprocessing.

**Third step: Fill the dict**

We should fill the dict SUPPORTED_MODELS in `models/init.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/models/__init__.py>`_

Apply a new scenario
======================
In this section, we will apply a recommendation scenario, which involves task and trainerflow.

How to build a new task
---------------------------------
**First step: Register task**

We should create a class our_task that inherits
the :ref:`BaseTask <api-task>`
and register the model with @register_task(str).
We give the task recommendation as an example.

.. code-block:: python

    from openhgnn.tasks import BaseTask, register_task
    @register_task('recommendation')
    class Recommendation(BaseTask):
        ...

**Second step: Implement methods**

We should implement some methods involved with evaluation metric.

.. code-block:: python

    class Recommendation(BaseTask):
        """Recommendation tasks."""
        def __init__(self, args):
            super(Recommendation, self).__init__()
            self.n_dataset = args.dataset
            self.dataset = build_dataset(args.dataset, 'recommendation')
            self.train_hg, self.train_neg_hg, self.val_hg, self.test_hg = self.dataset.get_idx()
            self.evaluator = Evaluator(args.seed)

        def get_loss_fn(self):
            return F.binary_cross_entropy_with_logits

        def evaluate(self, y_true, y_score, name):
            if name == 'ndcg':
                return self.evaluator.ndcg(y_true, y_score)


**Finally**

We should fill the dict SUPPORTED_TASKS in `task/init.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/tasks/__init__.py>`_

How to build a new trainerflow
-------------------------------
**First step: Register trainerflow**

We should create a class your_trainerflow that inherits the :ref:`BaseFlow <api-trainerflow>`
and register the trainerflow with @register_trainer(str).

.. code-block:: python

    from openhgnn.trainerflow import BaseFlow, register_flow
    @register_flow('demo_trainer')
    class DemoTrainer(BaseFlow):
        ...

**Second step: Implement methods**

We decorate the func train() with @abstractmethod. So the train() must be overridden, or the trainerflow cannot be instantiated.
Besides train(), the init() and _test_step() should both be implemented.
One of the _full_train_step() and _mini_train_step() must be implemented at least.


.. code-block:: python

    ...
    class Recommendation(BaseFlow):
         def __init__(self, args=None):
        super(Recommendation, self).__init__(args)
            self.target_link = self.task.dataset.target_link
            self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
            self.evaluator = self.task.get_evaluator(self.metric)

        def train(self,):
            for epoch in epoch_iter:
                self._full_train_step()
                self._full_test_step()

        def _full_train_step(self,):
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
