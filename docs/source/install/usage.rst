Basic Usage
==========================

Run experiments
------------------
Running an existing baseline model on an existing benchmark api_dataset_

.. code:: bash

    >>>python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config
    usage: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config]

*optional arguments*:
    - --model MODEL,	-m MODEL	name of models
    - --task TASK,	-t TASK	name of task
    - --dataset DATASET,	-d DATASET	name of datasets
    - --gpu GPU, -g GPU	controls which gpu you will use. If you do not have gpu, set -g -1.
    - --use_best_config	use_best_config means you can use the best config in the dataset with the model. If you want to set the different hyper-parameter, modify the [openhgnn.config.ini](./openhgnn/config.ini) manually.
    - --use_hpo Besides use_best_config, we give a hyper-parameter [example](./openhgnn/auto) to search the best hyper-parameter automatically.

e.g.:

.. code:: bash

    python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config


**Note**: If you are interested in some model,
you can refer to the below `models list <https://github.com/BUPT-GAMMA/OpenHGNN#models>`_.

Evaluate a new dataset
=======================
When the existing dataset can not meet your needs, you can custom your dataset.
In this section, we will create a new dataset HGBn-ACM, which is used in *node classification* task.

How to build a new dataset
---------------------------

**First step: Process dataset**

We give a `demo <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/debug/HGBn-ACM2dgl.py>`_ to process the HGBn-ACM.
First, download the HGBn-ACM from the `Link <https://www.biendata.xyz/hgb/#/datasets>`_.
After that, we process it as a `dgl.heterograph <https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/dataset#Dataset>`_.

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

We recommend the feature name set by the `"h"`.

.. code:: python

    >>> g.nodes['drug'].data['h'] = th.ones(3, 1)

DGL provides :func:`dgl.save_graphs` and :func:`dgl.load_graphs` respectively for saving
heterogeneous graphs in binary format and loading them from binary format.
So we can use `dgl.load_graphs <https://docs.dgl.ai/en/latest/generated/dgl.load_graphs.html#>`_ to store graph into the local.

.. code:: python

    >>> dgl.save_graphs("demo_graph.bin", g)

**Second step: Add extra information**

We can get a binary format named *demo_graph.bin* after first step, and we should move it into the directory *openhgnn/dataset/*.
But for now, it is not a complete dataset.
We should specify some important information in the `NodeClassificationDataset.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/dataset/NodeClassificationDataset.py#L145>`_

For example, we should set the *category*, *num_classes* and *multi_label*(if necessary) with ``"paper"``, ``3``, ``True``.
More infos, refer to :ref:`Base Node Classification Dataset <api-base-node-dataset>`.

**Third step: optional**

We can use demo_graph as our dataset name to evaluate a existing model.

.. code:: bash

    python main.py -m GTN -d demo_graph -t node_classification -g 0 --use_best_config


If you have another dataset name, you should also modify the `build_dataset <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/dataset/__init__.py>`_.

Apply a new model
====================
In this section, we will create a model,
which is a very simple graph embedding algorithm.

First we import necessary library like numpy, scipy, networkx, sklearn,
we also import API like ‘BaseModel’ and ‘register_model’ from cogl/models/ to build our new model:


How to build a new model
--------------------------
**First step**

We should create a class your_model that inherits
the :ref:`Base Model <api-model>` .
and register the model with @register_model(str).

.. code-block:: python

    from openhgnn.models import BaseModel, register_model
    @register_model('demo_model')
    class Demo(BaseModel):
        ...

**Second step**

We must implement the classmethod build_model_from_args , other functions like __init__(), forward() and so on.

.. code-block:: python

    ...
    class Demo(BaseModel):
        @classmethod
        def build_model_from_args(cls, args, hg):
            ...
            return cls(...)
        def __init__():
            return
        def forward(hg, h_dict=None):
            return h_dict

**Finally**

We should fill the dict SUPPORTED_MODELS in `models/init.py <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/models/__init__.py>`_

Apply a existing model to a new scenario
==============================================


