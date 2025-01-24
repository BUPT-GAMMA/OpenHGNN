# Model

A model plays a role of encoder. For input given a hetero-graph or [MFG](https://docs.dgl.ai/en/latest/guide/minibatch-node.html?highlight=message%20flow%20graphs) and feature (if available), the model need to output a dict of node embedding. In general, it should ouput all nodes embedding. It is allowed that just output the embedding of target nodes which are participated in loss calculation. 

It mainly contains two parts: model builder and forward propagation.

#### Model Builder

We create a classmethod *build_model_from_args* for every model. with that, we can use *args* and *hg* to build up a custom model with model-specific hyper-parameters. So it is necessary to implement the func *build_model_from_args* in your model.

#### forward propagation

```python
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
```

Input: The args *hg* and *feat_dict* are not necessary. 

- If you just want to train model with full-batch training, you can pass the *hg* to the model in the __init__ .
- If the feature of dataset is not provided, build a *[HeteroEmbedLayer](../layers/EmbedLayer.py)*.

Output: A dict of embedding.

- output all nodes embedding or just the target node embedding which is involved in loss calculation.

### Model Category

| Model                                        | Embedding          | GNN                | Knowledge Graph    |
| -------------------------------------------- | ------------------ | ------------------ | ------------------ |
| Metapath2vec[KDD 2017]                       | :heavy_check_mark: |                    |                    |
| [RGCN](./openhgnn/output/RGCN)[ESWC 2018]    |                    | :heavy_check_mark: | :heavy_check_mark: |
| [HAN](./openhgnn/output/HAN)[WWW 2019]       |                    | :heavy_check_mark: |                    |
| [HetGNN](./openhgnn/output/HetGNN)[KDD 2019] | :heavy_check_mark: | :heavy_check_mark: |                    |
| [GTN](./openhgnn/output/GTN)[NeurIPS 2019]   |                    | :heavy_check_mark: |                    |
| [RSHN](./openhgnn/output/RSHN)[ICDM 2019]    |                    |                    | :heavy_check_mark: |
| [MAGNN](./openhgnn/output/MAGNN)[WWW 2020]   | :heavy_check_mark: | :heavy_check_mark: |                    |
| [NSHE](./openhgnn/output/NSHE)[IJCAI 2020]   | :heavy_check_mark: | :heavy_check_mark: |                    |
| CompGCN[ICLR 2020]                           |                    |                    | :heavy_check_mark: |

### How to build a new model

1. Create a class *your_model* that inherits the [*BaseModel*](./base_model.py) and register the model with *@register_model(str)*.
2. Implement the classmethod build_model_from_args.
3. Implement the forward().
4. Fill the dict *SUPPORTED_MODELS* in [models/__init__.py](./__init__.py)

