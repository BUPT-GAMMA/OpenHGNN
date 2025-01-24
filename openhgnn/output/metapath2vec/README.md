# Metapath2vec[KDD2017]

Paper: [**metapath2vec: Scalable Representation Learning for Heterogeneous Networks**](https://ericdongyx.github.io/metapath2vec/m2v.html)

Code from author: https://www.dropbox.com/s/w3wmo2ru9kpk39n/code_metapath2vec.zip

Code from dgl Team: https://github.com/dmlc/dgl/tree/master/examples/pytorch/metapath2vec

## How to run

```bash
python main.py -m Metapath2vec -t node_classification -d dblp4MAGNN -g 0
```

If you do not have gpu, set -gpu -1.

## Performance

### Node Classification

| Dataset       |  Metapath | Macro-F1 | Micro-F1 |
| ------------- |--------   | -------- | -------- |
| dblp          |  APVPA    | 0.9241   | 0.9288   |


