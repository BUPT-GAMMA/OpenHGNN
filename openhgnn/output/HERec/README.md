# HERec[TKDE2018]

Paper: [**Heterogeneous Information Network Embedding for Recommendation**](https://ieeexplore.ieee.org/abstract/document/8355676)

Code from author: https://github.com/librahu/HERec

## How to run

```bash
python main.py -m HERec -t node_classification -d dblp4MAGNN -g 0
```

If you do not have gpu, set -gpu -1.

## Performance

### Node Classification

| Dataset       |  Metapath | Macro-F1 | Micro-F1 |
| ------------- |--------   | -------- | -------- |
| dblp          |  APVPA    | 0.9277   | 0.9331   |