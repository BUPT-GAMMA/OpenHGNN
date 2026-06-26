# HERO

-   paper: [SELF-SUPERVISED HETEROGENEOUS GRAPH LEARN ING: A HOMOPHILY AND HETEROGENEITY VIEW](https://openreview.net/forum?id=3FJOKjooIj)
-   Code from :(https://github.com/YujieMo/HERO)

# Note: 
HERO_homo only supports homogeneous graphs.  
If you want to run the heterogeneous graph version, please refer to the `HERO` README.

## How to run
Run HERO_homo on a homogeneous dataset, for example:
```bash
python main.py -m HERO_homo -d photo4HERO -t node_classification -g 0
python main.py -m HERO_homo -d computers4HERO -t node_classification -g 0
python main.py -m HERO_homo -d cs4HERO -t node_classification -g 0
python main.py -m HERO_homo -d physics4HERO -t node_classification -g 0
```

## Dataset

Current implementation supports homogeneous graph datasets, including:

- `photo4HERO`
- `computers4HERO`
- `cs4HERO`
- `physics4HERO`
- `cora4HERO`
- `citeseer4HERO`
- `pubmed4HERO`
- `corafull4HERO`
- `wikics4HERO`
- `ogbn-arxiv4HERO`

These datasets are automatically downloaded or loaded through **DGL built-in dataset APIs** and the **OGB dataset interface**.

The dataset class then processes the raw homogeneous graph, constructs the corresponding feature views and feature-distance matrix, and caches them as processed files.

## Performance: NodeClassification

| Dataset   | F1-micro | F1-macro |Target node|
| --------- | -------- | -------- |-----------|
| Photo     | 0.93343  | 0.92021  | photo     |
| Computers | 0.88533  | 0.86264  | computer  |
| CS        | 0.92754  | 0.87665  | author    |
| Physics   | 0.95925  | 0.94587  | author    |

## Hyperparameter Settings

For dataset-specific hyperparameter settings, please refer to `openhgnn/config.ini`.
In the current configuration, the following datasets have manually tuned hyperparameters:

- `photo4HERO`
- `computers4HERO`
- `cs4HERO`
- `physics4HERO`

#### If you have any questions,

Submit an issue or email to  3144537424@qq.com