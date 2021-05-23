# OpenHGNN-DGL
This is an open-source toolkit for Heterogeneous Graph Neural Network(OpenHGNN) based on DGL. We will integrate SOTA models of heterogeneous graph.

### Two Keys

- Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and datasets. TODO: [Feature]AutoML & hyper-parameter search.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.

#### Running an existing baseline model on an existing benchmark dataset

```python
python main.py -m model_name -d dataset_name -t task_name
```

It is under development, and we release it in a nightly build version. For now, we just give some new models, such as HetGNN, NSHE, GTN, MAGNN, RSHN.

### Some models

- [HetGNN](https://github.com/BUPT-GAMMA/OpenHGNN-DGL/tree/main/openhgnn/output/HetGNN)[KDD2019] for node classification and link prediction
- [GTN](https://github.com/BUPT-GAMMA/OpenHGNN-DGL/tree/main/openhgnn/output/GTN)[NeurIPS 2019] for node classification

