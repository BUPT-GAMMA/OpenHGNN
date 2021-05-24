# OpenHGNN-DGL
This is an open-source toolkit for Heterogeneous Graph Neural Network(OpenHGNN) based on DGL. We will integrate SOTA models of heterogeneous graph.

### Two Keys

- Easy-to-Use: OpenHGNN provides easy-to-use interfaces for running experiments with the given models and datasets. TODO: [Feature]AutoML & hyper-parameter search.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
- Efficiency: The backend dgl provides efficient APIs.

#### Why OpenHGNN

##### Relation with dgl

OpenHGNN is a high-level package built on top of DGL; it will not cover efficient implementation of low-level components in C++.

##### Relation with OpenHINE

| Package  | Heterograph structure | Models                            | Efficiency  | Others                                     |
| -------- | --------------------- | --------------------------------- | ----------- | ------------------------------------------ |
| OpenHINE | write by our own      | Most embedding methods            | Inefficient | ---                                        |
| OpenHGNN | API from DGL          | Most graph neural network methods | Efficient   | Better Extensibility, Better Encapsulation |



![image-20210524165402379](/Users/ztany/Library/Application Support/typora-user-images/image-20210524165402379.png)

#### Running an existing baseline model on an existing benchmark dataset

```python
python main.py -m model_name -d dataset_name -t task_name
```

It is under development, and we release it in a nightly build version. For now, we just give some new models, such as HetGNN, NSHE, GTN, MAGNN, RSHN.

### Some models

- [HetGNN](https://github.com/BUPT-GAMMA/OpenHGNN-DGL/tree/main/openhgnn/output/HetGNN)[KDD2019] for node classification and link prediction
- RSHE[ICDM2019] for entity classification
- [GTN](https://github.com/BUPT-GAMMA/OpenHGNN-DGL/tree/main/openhgnn/output/GTN)[NeurIPS 2019] for node classification
- NSHE[IJCAI2020]
- MAGNN[WWW2020]

### Contributors

GAMMA LAB [BUPT]: Tianyu Zhao, Cheng Yang, Chuan Shi

DGL Team: Quan Gan, Jian Zhang

BUPT: Jiahang Li

