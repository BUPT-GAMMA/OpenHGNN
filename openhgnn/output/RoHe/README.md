# RoHe[AAAI2022]

Paper: [**Robust Heterogeneous Graph Neural Networks against Adversarial Attacks**](http://shichuan.org/doc/132.pdf)

Code from author: https://github.com/BUPT-GAMMA/RoHe

We use the code from author and integrate the model into our package. 

### How to run:

```bash
python main.py -m RoHe -t node_classification -d acm_han_raw -g 0
```

If you do not have gpu, set -gpu -1.

### Performance:

| model    | Micro-F1[raw data]           | Macro-F1[raw data] | Micro-F1[attacked data] | Macro-F1[attacked data] |
| -------- | ---------------------------- | ------------------ | ----- | ---- |
| Raw-HAN  | 0.910     | 0.908              | **0.457** | **0.209**  |
| RoHe-HAN |     0.940               | 0.941             | **0.929** | **0.930** |

### TrainerFlow: 

trainflow/RoHe_trainer.py

### Model:

models/RoHe.py

### More

#### Contirbutor

Dashuai Yue[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to [jayyue@bupt.edu.cn](mailto:jayyue@bupt.edu.cn).
