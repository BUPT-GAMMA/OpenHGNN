# Datasets and Interfaces for Benchmarking Heterogeneous Graph Neural Networks

## Introduction
To accommodate the urgent requirement of emerging fields and the advance of Heterogeneous Graph Neural Networks (HGNNs), we build a new benchmark for two new fields: risky product detection (ICDM) and takeout recommendation (MTWM). Besides that, we establish benchmark interfaces with over 40 heterogeneous graph datasets from other fields and providea powerful and novel toolkit to research the charactertistics of graph datasets. All of the above is publicly available jusy by several codes.

## Get Started
### Python environment:
```bash
conda create --name hgbi python=3.7
```
### Install packages
```bash
pip install -r requirement.txt 
#install source code
cd OpenHGNN
pip install .
```

### Import related package
from openhgnn.utils import hgbi
from openhgnn.utils.tsne_g import draw_tsne
from openhgnn.utils.visualize import plot_degree_dist
from openhgnn.utils.meta_path_analyse import meta_path_heterophily

### Example 1: Load risk product detection dataset (RPDD) and takeout recommendation dataset (TRD)
```
import utils.hgbi as hgbi
ds_node = hgbi.build_dataset(
    name = 'RPDD',task = 'node_classification')
ds_link = hgbi.build_dataset(
    name = 'TRD',task = 'link_prediction')
```

### Example 2: Load other heterogeneous graph dataset
You can also load other graph dataset from other fields:
```
ds_node = hgbi.build_dataset(
    name = 'acm4NSHE',task = 'node_classification')
print(ds_node.g)

ds_link = hgbi.build_dataset(
    name = 'ohgbl-yelp2',task = 'link_prediction')
print(ds_link.g)
```

### Example 3: Analysis and visualization
```

dataset = hgbi.build_dataset(
        name = 'dblp4GTN',task = 'node_classification')

plot_degree_dist(dataset.g,'./degree.png')
draw_tsne(dataset,'./sne.png')
meta_path_nums, heterophily, edge_radio  = meta_path_heterophily (g, meta_paths_dict=dataset.meta_paths_dict, strength=2)

```
For more details, please refer to "demo_experiment.py"

## Summary of total available heterogeneous graph datasets
### Node classification
| Dataset        | Ntype | Node       | Etype | Edge        | Avg Attri | Label | Model     | Original (default: Macro/Micro-F1%)     | Reproduced (Macro/Micro_F1%) |
| :------------- | :---- | :--------- | :---- | :---------- | :-------- | :---- | :-------- | :---------------- | :-------------- |
| acm4NSHE       | 3     | 11,246     | 4     | 34,852      | 128       | 3     | NSHE      | 83\.27/84.12      | 84\.78/84.95    |
| acm4HeCo       | 3     | 11,246     | 4     | 34,852      | 3,043     | 3     | HeCo      | 89\.04/88.71      | 88\.66/88.35    |
| acm4NARS       | 3     | 21,488     | 4     | 34,864      | 720       | 3     | NARS      | 92\.9 (Accuracy)  | 91\.35/91.44    |
| acm4HetGNN     | 3     | 49,708     | 5     | 202,067     | 387       | 4     | HetGNN    | 97\.8/97.9        | 97\.01/97.05    |
| acm4GTN        | 3     | 8,994      | 4     | 25,922      | 1,902     | 3     | GTN       | 92\.68 (F1 score) | 92\.03/92       |
| dblp4MAGNN     | 4     | 26,128     | 6     | 239,566     | 5,601     | 4     | SimpleHGN | 93\.89/94.35      | 86\.79/86.75    |
| imdb4MAGNN     | 3     | 11,616     | 4     | 34,212      | 3,468     | 3     | MAGNN     | 60\.43/60.63      | 62\.85/62.78    |
| imdb4GTN       | 3     | 12,772     | 4     | 37,288      | 1,256     | 4     | GTN       | 60\.92 (F1 score) | 56\.97/58.61    |
| yelp4HeGAN     | 5     | 3,913      | 8     | 77,360      | 64        | 3     | HeGAN     | 85\.24/80.31      | 71\.51/79.16    |
| HGBn-ACM       | 4     | 10,942     | 8     | 547,872     | 1,902     | 3     | SimpleHGN | 93\.2/93.12       | 66\.64/88.4     |
| HGBn-DBLP      | 4     | 26,128     | 6     | 239,566     | 1,538     | 4     | SimpleHGN | 93\.77/94.35      | 86\.31/87.24    |
| ohgbn-Freebase | 8     | 12,164,758 | 36    | 62,982,566  | N/A       | 8     | RGCN      | N/A       | 53\.07/69.33    |
| ohgbn-yelp2    | 4     | 82,465     | 4     | 30,542,675  | N/A       | 16    | RGCN      | 5\.10/23.24       | 5\.04/40.44     |
| ohgbn-acm      | 3     | 8,994      | 2     | 25,922      | 1,902     | 3     | fastGTN   | N/A               | 92\.92/92.85    |
| ohgbn-imdb     | 3     | 12,772     | 4     | 37,288      | 1,256     | 3     | RGCN      | N/A               | 57\.57/63.66    |
| dblp4GTN       | 3     | 18,405     | 4     | 67,946      | 334       | 4     | fastGTN   | 94\.18 (F1 score)  | 90\.39/91.39    |
| aifb           | 7     | 7,262      | 104   | 48,810      | N/A       | 4     | RGCN      | 95\.83 (Accuracy) | 96\.92/97.22                   |
| mutag          | 5     | 27,163     | 50    | 148,100     | N/A       | 2     | RGCN      | 73\.23 (Accuracy) | 66\.40/70.59                   |
| bgs            | 27    | 94,806     | 122   | 672,884     | N/A       | 2     | RGCN      | 83\.10 (Accuracy) | 88\.26/89.66                   |
| am             | 7     | 1,885,136  | 108   | 5,668,682   | N/A       | 11    | RGCN      | 89\.29 (Accuracy) | 89\.41/89.90                   |
| RPDD           | 7     | 13,806,619 | 7     | 157,814,864 | 256       | 2     | RGCN      | N/A               | 90\.46/98.02                   |

### Link prediction
| Dataset        | Ntype | Node       | Etype | Edge       | Avg Attri | Label | Model   | Paper  | AUC\_ROC    |
| :------------- | :---- | :--------- | :---- | :--------- | :-------- | :---- | :------ | :----- | :---------- |
| amazon4SLICE   | 1     | 10,099     | 2     | 170,783    | 1,156     | 2     | RGCN    | N/A    | 74\.6(avg)  |
| HGBl-ACM       | 4     | 10,942     | 8     | 547,872    | 1,902     | 1     | HDE     | N/A    | 87\.41      |
| HGBl-DBLP      | 4     | 26,128     | 6     | 239,566    | 1,538     | 1     | HDE     | N/A    | 98\.36      |
| HGBl-IMDB      | 4     | 21,420     | 6     | 86,642     | 3,390     | 1     | HDE     | N/A    | 91\.51      |
| HGBl-amazon    | 1     | 10,099     | 2     | 148,659    | 1,156     | 2     | GATNE-T | N/A    | 80\.83(avg) |
| HGBl-LastFM    | 3     | 20,612     | 6     | 283,042    | N/A       | 1     | RGCN    | 81\.9 | 76\.46      |
| HGBl-PubMed    | 4     | 63,109     | 20    | 489,972    | 200       | 1     | RGCN    | 88\.32 | 89\.3       |
| ohgbl-yelp1    | 4     | 2,353,365  | 4     | 10,417,742 | N/A       | 1     | CompGCN | N/A    | 61\.21      |
| ohgbl-yelp2    | 4     | 82,465     | 4     | 31,206,253 | N/A       | 1     | RGCN    | N/A    | 65\.6       |
| ohgbl-Freebase | 8     | 12,164,755 | 36    | 63,906,230 | N/A       | 1     | RGCN    | 50\.18 | 58\.75      |
| DoubanMovie    | 6     | 37,595     | 12    | 3,429,852  | N/A       | 1     | RGCN    | N/A    | 91\.55      |
| TRD            | 3     | 408,849    | 4     | 18,931,400 | N/A       | 1     | RGCN    | N/A    | 92\.69      |

