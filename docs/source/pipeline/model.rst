.. _pipeline-models:

Model
========

A model plays a role of encoder. For input given a hetero-graph or MFG and feature (if available),
the model need to output a dict of node embedding. In general, it should ouput all nodes embedding.
It is allowed that just output the embedding of target nodes which are participated in loss calculation.

It mainly contains two parts: model builder and forward propagation.

Model Builder
--------------

We create a classmethod build_model_from_args for every model.
With that, we can use args and hg to build up a custom model with model-specific hyper-parameters.
So it is necessary to implement the func build_model_from_args in your model.

forward propagation
-----------------------

.. code:: python

   def forward(self, hg=None, feat_dict=None, *args, **wkags):
   '''
        Parameters
        ----------
        g : dgl.heterograph
   					the dgl heterogeneous graph
        feat_dict : dict
            the feature matrix dict of different node types,
            e.g {'author':feat_of_author[number of author nodes * dimension], ...}
        Returns
        -------
        out_dict: dict
        		the same with feat_dict
        '''
   return out_dict

Input: The args hg and feat_dict are not necessary.

* If you just want to train model with full-batch training, you can pass the hg to the model in the init .
* If the feature of dataset is not provided, build a HeteroEmbedLayer.

Output: A dict of embedding.

* output all nodes embedding or just the target node embedding which is involved in loss calculation.

Model Category
-------------------

+------------------------+-----------+-----+-----------------+
| Model		         | Embedding | GNN | Knowledge Graph |
+========================+===========+=====+=================+
| Metapath2vec[KDD 2017] |✔          |     |                 |
+------------------------+-----------+-----+-----------------+️
| RGCN[ESWC 2018]        |           |✔    |   ✔             |
+------------------------+-----------+-----+-----------------+
| HAN[WWW 2019]		 |           |✔    |                 |
+------------------------+-----------+-----+-----------------+
| HetGNN[KDD 2019]	 |✔          |✔    |                 |
+------------------------+-----------+-----+-----------------+
| GTN[NeurIPS 2019]      |           |✔    |                 |
+------------------------+-----------+-----+-----------------+
| RSHN[ICDM 2019]	 |           |     |✔                |
+------------------------+-----------+-----+-----------------+
| MAGNN[WWW 2020]	 |✔          |✔    |                 |
+------------------------+-----------+-----+-----------------+
| NSHE[IJCAI 2020]	 |✔          |✔    |                 |
+------------------------+-----------+-----+-----------------+
| CompGCN[ICLR 2020]     |           |     |✔                |
+------------------------+-----------+-----+-----------------+


How to build a new model
--------------------------
**First step**

We should create a class your_model that inherits
the `BaseModel <https://github.com/BUPT-GAMMA/OpenHGNN/blob/main/openhgnn/models/base_model.py>`_
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