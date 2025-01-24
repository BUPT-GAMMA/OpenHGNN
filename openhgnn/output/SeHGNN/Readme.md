# SeHGNN[AAAI 23]

- paper: [Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN)](http://arxiv.org/abs/2207.02547)
- Code from author: [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN)

## How to run

- Clone the Openhgnn-DGL

```
python main.py -m SeHGNN -d ogbn-mag -g 0 
```

If you do not have gpu, set -gpu -1.

## Performance on ogbn-mag
| Node classification | Validation accuracy | Test accuracy |
| ------------------- | -------- | -------- |
| paper               | 58.70±0.08    | 56.71±0.14    |
| OpenHGNN            | 58.6584   | 56.6251   |

## Hyper-Parameter

You can modify the parameters in openhgnn/config.ini

## More

#### Contributor

Yue Yu(2023)[GAMMA LAB]

#### If you have any questions,

Submit an issue or email to  [loadingyy12138@gmail.com](mailto:loadingyy12138@gmail.com).



