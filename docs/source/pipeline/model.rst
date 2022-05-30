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

An example from section **how to build a new model**

.. code:: python

    class RGAT(BaseModel):
        @classmethod
        def build_model_from_args(cls, args, hg):
            return cls(in_dim=args.hidden_dim,
                       out_dim=args.hidden_dim,
                       h_dim=args.out_dim,
                       etypes=hg.etypes,
                       num_heads=args.num_heads,
                       dropout=args.dropout)

Forward Propagation
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

.. note::

    In OpenHGNN, we preprocess the feature of dataset outside of model.
    Specifically, we use a linear layer with bias for each node type to map all node features to a shared feature space.
    So the parameter *h_dict* of *forward()* in model is not original, and your model need not feature preprocessing.

Output: A dict of embedding.

* output all nodes embedding or just the target node embedding which is involved in loss calculation.

