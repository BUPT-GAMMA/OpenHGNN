## 训练与推理脚本

### 代码目录结构

- link_prediction_train.py 链路预测训练脚本
- link_prediction_inference.py 链路预测预测脚本
- node_classification_train.py 节点分类训练脚本
- node_classification_inference.py 节点分类预测脚本
- my_config.ini 配置文件

### 节点分类任务

#### 训练

```
cd examples/inference
python node_classification_train.py -m RGCN -g -1 --mini-batch-flag
```

*命令行参数*:

``--model -m ``    模型名 支持模型包括 RGCN HAN ieHGCN CompGCN RHGNN HPN NARS MHNF RSHN HGAT HetSANN SimpleHGN HGT MAGNN GTN
fastGTN

``--gpu -g``    控制你使用哪一个GPU，如果没有GPU，设定 -g -1

``--mini-batch-flag``  使用minibatch训练 支持模型包括 RGCN HAN ieHGCN CompGCN RHGNN

#### 预测

```
python node_classification_inference.py -m RGCN -g -1 --mini-batch-flag
```

预测的输出为目标类型节点的下标和预测值

### 链路预测任务

#### 训练

```
cd examples/inference
python link_prediction_train.py -m RGCN -g -1 --mini-batch-flag
```

*命令行参数*:

``--model -m ``    模型名 支持模型包括 RGCN

``--gpu -g``    控制你使用哪一个GPU，如果没有GPU，设定 -g -1

``--mini-batch-flag``  使用minibatch训练 支持模型包括 RGCN

#### 预测

```
python link_prediction_inference.py -m RGCN -g -1 --mini-batch-flag
```

### 模型参数

参见 my_config.ini

### 图表示学习任务

#### 训练

```
cd examples/inference
python link_prediction_train.py -m RGCN -g -1 --mini-batch-flag
```

*命令行参数*:

``--model -m ``    模型名 支持模型包括 RGCN

``--gpu -g``    控制你使用哪一个GPU，如果没有GPU，设定 -g -1

``--mini-batch-flag``  使用minibatch训练 支持模型包括 RGCN

