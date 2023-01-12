## 使用OpenHGNN构建应用

### 使用说明

首先将config复制到该目录下

```
cp openhgnn/config.ini examples/applications/my_config.ini 
```

然后根据应用场景选择任务类型进行训练和推理

### 节点分类任务

#### 训练

```
cd examples/applications/node_classification
python train.py -m RGCN -g -1 --mini-batch-flag
```

*命令行参数*:

``--model -m ``    模型名 支持模型包括 RGCN HAN ieHGCN CompGCN RHGNN HPN NARS MHNF RSHN HGAT HetSANN SimpleHGN HGT MAGNN GTN
fastGTN

``--gpu -g``    控制你使用哪一个GPU，如果没有GPU，设定 -g -1

``--mini-batch-flag``  使用minibatch训练 支持模型包括 RGCN HAN ieHGCN CompGCN RHGNN

#### 预测

```
python inference.py -m RGCN -g -1 --mini-batch-flag
```

预测的输出为目标类型节点的下标和预测值

### 链路预测任务

#### 训练

```
cd examples/applications/link_prediction
python train.py -m RGCN -g -1 --mini-batch-flag
```

*命令行参数*:

``--model -m ``    模型名 支持模型包括 RGCN HAN

``--gpu -g``    控制你使用哪一个GPU，如果没有GPU，设定 -g -1

``--mini-batch-flag``  使用minibatch训练 支持模型包括 RGCN HAN

#### 预测

```
python inference.py -m RGCN -g -1 --mini-batch-flag
```

预测的输出为目标边的首尾节点标号和边的打分

### 模型参数

参见配置文件

### 图表示学习任务

#### 训练

```
cd examples/applications/graph_representation_learning
python train.py -m Metapath2vec -mp APA -g -1 
```

*命令行参数*:

``--model -m ``    模型名 支持模型包括 Metapath2vec HERec

``--gpu -g``    控制你使用哪一个GPU，如果没有GPU，设定 -g -1

``--meta-path-key -mp``   指定进行随机游走时使用的metapath

## RGCN
### 模型简述
RGCN把图卷积操作拓展到多种边关系上，即对每一种边关系的邻居表征乘上各自边关系对应的参数矩阵。其中为了应对关系种类过多导致参数量增加的情况，可以把每一种关系的参数矩阵表示成固定数量个基矩阵的组合。

## HAN
### 模型简述
HAN指定多条元路径，并且根据元路径生成生成元路径邻居，例如根据作者-论文-作者这一元路径可以获得作者节点的所有元路径可达的作者邻居，在多个元路径抽取出的子图上应用图注意机制(GATConv)，最终把每个子图上的节点表征通过语义级别的注意力机制进行融合。

## CompGCN
### 模型简述
CompGCN联合学习了边类型和节点的嵌入。

## Metapath2vec

### 模型简述
在图上按照一个固定的元路径(meta path)例如作者-论文-作者进行随机游走，并基于同一路径上的相近节点相似度进行建模，应用SkipGram算法生成所有类型的节点嵌入。

## HERec

### 模型简述
与Metapath2vec类似，在图上按照一个固定的元路径(meta path)例如作者-论文-作者进行随机游走，区别在于HERec基于同一路径上相同类型而不是所有类型的节点相似度进行建模，应用SkipGram算法生成该类型节点的节点嵌入。

## MAGNN 

### 模型概述
MAGNN首先应用特定的线性变换将异构节点的属性特征投影到一个潜在的向量空间中, 然后使用注意力机制对每个元路径进行元路径内部聚合, 最后再使用注意力机制进行元路径之间聚合, 将多个元路径中获得的潜在向量融合到最终节点嵌入中. 通过多个元路径的聚合, 我们的模型可以学习异构图中的综合语义信息.

## MHNF
### 模型概述
MHNF首先针对每一条meta path逐跳进行消息聚合, 即第k层聚合目标节点k跳邻居传递的消息. 然后引入注意力机制对不同meta path的信息聚合, 更新目标节点的节点嵌入

## RSHN
### 模型概述
RSHN首先创建了一个粗线图神经网络(CL-GNN)，挖掘基于粗化的线图的不同类型边的潜在关联的以边为中心的关系结构特征。然后，使用异构图神经网络(H-GNN)来利用来自异构图中相邻节点和节点间传播的边缘的隐式消息。

## HGT
### 模型概述
HGT借用Transformer的思想，使用注意力机制完成消息传递。在计算注意力权重系数的时候，利用了节点类型和边类型的信息，从而得到Query，Key和Value，进而计算得到消息的值。但是在聚合的时候，和RGCN不同，它的聚合是不考虑边的类型，类似于同质图中的聚合，使用的程序设计接口也有所不同，最后聚合得到新的结果。

## SimpleHGN
### 模型概述
SimpleHGN主要思想是设计一个简单的，能应用在异质图上高效模型。同样使用了注意力机制，在计算注意力权重系数的时候，利用了边类型的信息，结合源节点的特征得到了消息。聚合的时候和HGT类似，不考虑边的类型，完成聚合得到新的结果。

## HetSANN
### 模型概述
HetSANN主要思想是充分利用节点和边类型的信息，应用在异质图上的模型。HetSANN利用了注意力机制，但是在完成特征映射的时候利用的是边类型的信息，随后利用映射好的特征，计算注意力权重系数，得到消息。聚合的时候和HGT类似，不考虑边的类型，完成聚合可以得到新的特征。

## ieHGCN
### 模型概述
ie-HGCN主要思想和RGCN类似，将图卷积操作应用在不同的关系类型上，从而得到不同关系类型下计算的结果。在聚合的过程中，利用了Transformer中的注意力机制，借助计算消息时得到的映射后的特征，计算Query，Key和Value，从而得到不同关系的注意力权重系数，进而完成最终的聚合。