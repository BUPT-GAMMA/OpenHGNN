# InGram[ICML]

Paper:InGram: [Inductive Knowledge Graph Embedding via Relation Graphs](https://proceedings.mlr.press/v202/lee23c/lee23c.pdf)


#### How to run

Clone the Openhgnn-DGL

```bash
python main.py -m Ingram -d NL-100 -t Ingram -g 0
```

If you do not have gpu, set -gpu -1.

Candidate dataset: NL-100

#### Performance

| InGram[OpenHGNN] | MR                        | MRR                         | H@10                        | H@1                          |
|------------------|---------------------------|-----------------------------|-----------------------------|------------------------------|
| NL-100           | Paper:92.6  openhgnn:97.0 | Paper:0.309  openhgnn:0.295 | Paper:0.506  openhgnn:0.494 | Paper: 0.212 openhgnn: 0.193 |




### TrainerFlow: Ingram


#### model

Ingram

### Dataset

NL-100


#### Contirbutor

Zikai Zhou[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to 460813395@qq.com.