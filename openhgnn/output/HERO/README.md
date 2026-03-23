# HERO

-   paper: [SELF-SUPERVISED HETEROGENEOUS GRAPH LEARN ING: A HOMOPHILY AND HETEROGENEITY VIEW](https://openreview.net/forum?id=3FJOKjooIj)
-   Code from :(https://github.com/YujieMo/HERO)

# Note: 
The HERO implementation mainly supports heterogeneous graph learning.  
If you want to test the homogeneous graph version, please refer to the `HERO_homo` README.

## How to run
- Clone the Openhgnn-DGL
  ```bash
  python main.py -m HERO -d Yelp4HERO -t node_classification -g 0
  python main.py -m HERO -d ACM4HERO -t node_classification -g 0
  python main.py -m HERO -d DBLP4HERO -t node_classification -g 0
  python main.py -m HERO -d Aminer4HERO -t node_classification -g 0
  ```
for high efficiency, only gpu

## Dataset

Current implementation supports:

- `ACM4HERO`
- `Aminer4HERO`
- `DBLP4HERO`
- `Yelp4HERO`

### Download Links

- ACM4HERO: `https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/ACM.zip`
- Aminer4HERO: `https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/Aminer.zip`
- DBLP4HERO: `https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/DBLP.zip`
- Yelp4HERO: `https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/Yelp.zip`

The dataset class automatically downloads raw files, processes them into `dgl.heterograph`, and caches them as processed files.

## Performance: NodeClassification

| Dataset | Target node | F1-micro | F1-macro |
| ------- | ----------- | -------- | -------- |
| ACM     | paper       | 0.91501  | 0.91595  |
| Aminer  | paper       | 0.72020  | 0.66705  |
| DBLP    | author      | 0.93588  | 0.92678  |
| Yelp    | business    | 0.92354  | 0.92677  |

## For dataset-specific hyperparameter settings, 
please refer to `openhgnn/config.ini`.

#### If you have any questions,

Submit an issue or email to  3144537424@qq.com