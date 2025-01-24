# HGCL

-paper: [Heterogeneous Graph Contrastive Learning
for Recommendation
](https://arxiv.org/pdf/2303.00995.pdf)

-code from author: [HGCL](https://github.com/HKUDS/HGCL)

## How to run
- Clone the Openhgnn-DGL
  ```bash
  python main.py -m HGCL -t recommendation -d Epinions -g 0
  ```

for high efficiency, only gpu

## Performance: Recommendation

-   Device: GPU, **GeForce GTX 1080Ti**
-   Dataset:Epinions,CiaoDVD,Yelp


| Recommendation |                HR                 |               NDCG                |
|:--------------:|:---------------------------------:|:---------------------------------:|
|    Epinions    | paper: 83.67%    OpenHGNN: 82.15% | paper: 64.13%    OpenHGNN: 62.45% |
|    CiaoDVD     | paper: 73.76%    OpenHGNN: 72.93% | paper: 52.61%    OpenHGNN: 50.72% |
|      Yelp      | paper: 87.12%    OpenHGNN: 86.26% | paper: 63.10%    OpenHGNN: 60.58% |

## More

#### Contributor

Siyuan Wen[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  [wsy0718@bupt.edu.cn](mailto:wsy0718@bupt.edu.cn).


