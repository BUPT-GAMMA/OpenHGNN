# 异质图神经网络开源工具包

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BUPT-GAMMA/OpenHGNN)
[![Documentation Status](https://readthedocs.org/projects/openhgnn-zh-cn/badge/?version=latest)](https://openhgnn.readthedocs.io/zh_CN/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/BUPT-GAMMA/OpenHGNN)
[![visitors](https://visitor-badge.glitch.me/badge?page_id=BUPT-GAMMA.OpenHGNN)](https://github.com/BUPT-GAMMA/OpenHGNN)
[![Total lines](https://img.shields.io/tokei/lines/github/BUPT-GAMMA/OpenHGNN?color=red)](https://github.com/BUPT-GAMMA/OpenHGNN)

[**启智社区（中文版）**](https://git.openi.org.cn/GAMMALab/OpenHGNN)｜ [**Github Community (English)**](https://github.com/BUPT-GAMMA/OpenHGNN) ｜[**Space4HGNN [SIGIR2022]**](./space4hgnn) ｜[**Benchmark&Leaderboard**](./openhgnn/dataset/ohgb.md) | [**Slack Channel**](https://app.slack.com/client/TDM5126J1/C03J6GND001)

OpenHGNN是一个基于 [DGL [Deep Graph Library]](https://github.com/dmlc/dgl) 和 [PyTorch](https://pytorch.org/) 的开源异质图神经网络工具包，集成了异质图神经网络的前沿模型。

## 新闻


<details>
<summary>
2024-07-23 开源0.7版本
</summary>
<br/>

我们开源了0.7版本
- 新增模型和数据集
- 新增图提示学习框架
- 新增DGL的数据处理框架Graphbolt
- 新增GNN消息聚合方式：dgl.sparse
- 新增分布式训练流程

</details>





<details>
<summary>
2023-07-17 开源0.5版本
</summary>
<br/>

我们开源了0.5版本。

- 新增模型和数据集
- 四个新的前沿图学习任务：预训练、推荐、图攻防、异常事件检测
- TensorBoard可视化功能
- 维护和测试模块

</details>
<details>

<summary>
2023-02-24 优秀孵化奖
</summary>
<br/>

OpenHGNN荣获启智社区优秀孵化项⽬奖！详细链接：https://mp.weixin.qq.com/s/PpbwEdP0-8wG9dsvRvRDaA

</details>

<details>

<summary>
2023-02-21 中国电子学会科技进步一等奖
</summary>
<br/>

算法库支撑了北邮牵头，蚂蚁集团、中国移动、海致科技等参与的“大规模复杂异质图数据智能分析技术与规模化应用”项目。该项目获得了2022年电子学会科技进步一等奖。

</details>
<details>

<summary>
2023-01-13 开源0.4版本
</summary>
<br/>

我们开源了0.4版本。

- 新增模型
- 提供构建应用的流程
- 更多支持采样训练的模型
- 更新千万节点规模图的评测

</details>

<details>
<summary>
2022-08-02 论文接收
</summary>
<br/>

我们的论文 <i>OpenHGNN: An Open Source Toolkit for Heterogeneous Graph Neural Network</i> 在CIKM2022 short paper track接收。

</details>

<details>
<summary>
2022-06-27 开源0.3版本
</summary>
<br/>

我们开源了0.3版本。

- 新增模型
- 支持API调用
- 简化定制数据集和模型流程
- 异质图信息可视化工具

</details>

<details>
<summary>
2022-02-28 开源0.2版本
</summary>
<br/>

我们开源了0.2版本。

- 新增模型
- 异质图神经网络的设计空间：[Space4HGNN [SIGIR2022]](./space4hgnn)
- 基准数据集以及排行榜：[Benchmark&Leaderboard](./openhgnn/dataset/ohgb.md)
</details>

<details>
<summary>
2022-01-07 加入启智社区
</summary>
<br/>
启智社区用户可以享受到如下功能：

- 全新的中文文档
- 免费的计算资源—— [云脑使用教程](https://git.openi.org.cn/GAMMALab/OpenHGNN/src/branch/main/yunnao_tutorial.md)
- OpenHGNN最新功能
  - 新增模型：【KDD2017】Metapath2vec、【TKDE2018】HERec、【KDD2021】HeCo、【KDD2021】SimpleHGN、【TKDE2021】HPN、【ICDM2021】HDE、fastGTN
  - 新增日志功能
  - 新增美团外卖数据集
  </details>
  
## 关键特性

- 易用：OpenHGNN提供了了易用的接口在给定的模型和数据集上运行实验，且集成了 [optuna](https://optuna.org/) 进行超参数优化。
- 可扩展：用户可以定义定制化的任务/模型/数据集来对新的场景应用新的模型。
- 高效：底层的DGL框架提供了提供了高效的API。

## 开始使用

#### 环境要求

- Python  >= 3.6

- [PyTorch](https://pytorch.org/get-started/)  >= 2.3.0

- [DGL](https://github.com/dmlc/dgl) >= 2.2.1

- CPU 或者 NVIDIA GPU, Linux, Python3

**1. Python 环境 (可选):** 推荐使用 Conda 包管理

```bash
conda create -n openhgnn python=3.6
source activate openhgnn
```

**2. 安装Pytorch:** 参考 [PyTorch安装文档](https://pytorch.org/get-started/) 根据你的操作系统和CUDA版本选择合适的安装命令。例如：

```bash
pip install torch torchvision torchaudio
```

**3. 安装DGL:** 参考 [DGL安装文档](https://www.dgl.ai/pages/start.html) 根据你的操作系统和CUDA版本选择合适的安装命令。例如：

```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

**4. 安装 openhgnn:** 

- 从pypi安装
```bash
pip install openhgnn
```

- 从源码安装
```bash
git clone https://github.com/BUPT-GAMMA/OpenHGNN
# If you encounter a network error, try git clone from openi as following.
# git clone https://git.openi.org.cn/GAMMALab/OpenHGNN.git
cd OpenHGNN
pip install .
```



**5. 安装 gdbi(可选):** 

- 安装gdbi
```bash
pip install git+https://github.com/xy-Ji/gdbi.git
```

- 安装图数据库
```bash
pip install neo4j==5.16.0
pip install nebula3-python==3.4.0
```


#### 在已有的评测上运行已有的基线模型 [数据集](./openhgnn/dataset/#Dataset)

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config --load_from_pretrained 
```

使用方法: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config][--use_database]

*可选参数*:

``-h, --help``    展示帮助信息并退出

``--model -m ``    模型名

``--task -t``    任务名

``--dataset -d``    数据集名

``--gpu -g``    控制你使用哪一个GPU，如果没有GPU，设定 -g -1。

``--use_best_config``    use_best_config 意味着你使用该模型在该数据集下最优的配置，如果你想要设定不同的超参数,请手动修改 [配置文件](./openhgnn/config.ini)。使用最佳配置会覆盖配置文件中的参数。

``--load_from_pretrained`` 从默认检查点加载模型。

``--use_database`` 从数据库加载数据集


``---mini_batch_flag`` 使用mini_batch方式训练HGNN

``---graphbolt`` 使用graphbolt框架的mini_batch训练流程

``---use_distributed`` 使用分布式方式训练HGNN



示例: 

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config

python main.py -m RGCN -d imdb4GTN -t node_classification -g 0 --mini_batch_flag --graphbolt

```

**提示**: 如果你对某个模型感兴趣,你可以参考下列的模型列表。

请参考 [文档](https://openhgnn.readthedocs.io/en/latest/index.html) 了解更多的基础和进阶的使用方法。


#### 使用TensorBoard可视化训练结果
```bash
tensorboard --logdir=./openhgnn/output/{model_name}/
```
示例：
```bash
tensorboard --logdir=./openhgnn/output/RGCN/
```

**提示**:需要先运行一次你想要可视化的模型，才能用以上命令可视化结果。

#### 使用gdbi访问数据库中的标准图数据
以neo4j数据库和imdb数据集为例
- 构造图数据集的csv文件(节点级:A.csv，连接级:A_P.csv)
- 导入csv文件到图数据库中
```bash
LOAD CSV WITH HEADERS FROM "file:///data.csv" AS row
CREATE (:graphname_labelname {ID: row.ID, ... });
```
- 在config.py文件中添加访问图数据库所需的用户信息
```python
self.graph_address = [graph_address]
self.user_name = [user_name]
self.password = [password]
```

- 示例: 

```bash
python main.py -m MAGNN -d imdb4MAGNN -t node_classification -g 0 --use_best_config --use_database
```



## [模型](./openhgnn/models/#Model)

### 特定任务下支持的模型

表格中的链接给出了模型的基本使用方法.

| 模型                                                     | 节点分类           | 链路预测           | 推荐               |
| -------------------------------------------------------- | ------------------ | ------------------ | ------------------ |
| [TransE](./openhgnn/output/TransE)[NIPS 2013]            |                    | :heavy_check_mark: |                    |
| [TransH](./openhgnn/output/TransH)[AAAI 2014]            |                    | :heavy_check_mark: |                    |
| [TransR](./openhgnn/output/TransR)[AAAI 2015]            |                    | :heavy_check_mark: |                    |
| [TransD](./openhgnn/output/TransD)[ACL 2015]             |                    | :heavy_check_mark: |                    |
| [Metapath2vec](./openhgnn/output/metapath2vec)[KDD 2017] | :heavy_check_mark: |                    |                    |
| [RGCN](./openhgnn/output/RGCN)[ESWC 2018]                | :heavy_check_mark: | :heavy_check_mark: |                    |
| [HERec](./openhgnn/output/HERec)[TKDE 2018]              | :heavy_check_mark: |                    |                    |
| [HAN](./openhgnn/output/HAN)[WWW 2019]                   | :heavy_check_mark: | :heavy_check_mark: |                    |
| [KGCN](./openhgnn/output/KGCN)[WWW 2019]                 |                    |                    | :heavy_check_mark: |
| [HetGNN](./openhgnn/output/HetGNN)[KDD 2019]             | :heavy_check_mark: | :heavy_check_mark: |                    |
| [HeGAN](./openhgnn/output/HeGAN)[KDD 2019]               | :heavy_check_mark: |                    |                    |
| HGAT[EMNLP 2019]                                         |                    |                    |                    |
| [GTN](./openhgnn/output/GTN)[NeurIPS 2019] & fastGTN     | :heavy_check_mark: |                    |                    |
| [RSHN](./openhgnn/output/RSHN)[ICDM 2019]                | :heavy_check_mark: | :heavy_check_mark: |                    |
| [GATNE-T](./openhgnn/output/GATNE-T)[KDD 2019]           |                    | :heavy_check_mark: |                    |
| [DMGI](./openhgnn/output/DMGI)[AAAI 2020]                | :heavy_check_mark: |                    |                    |
| [MAGNN](./openhgnn/output/MAGNN)[WWW 2020]               | :heavy_check_mark: |                    |                    |
| [HGT](./openhgnn/output/HGT)[WWW 2020]                   |                    |                    |                    |
| [CompGCN](./openhgnn/output/CompGCN)[ICLR 2020]          | :heavy_check_mark: | :heavy_check_mark: |                    |
| [NSHE](./openhgnn/output/NSHE)[IJCAI 2020]               | :heavy_check_mark: |                    |                    |
| [NARS](./openhgnn/output/NARS)[arxiv]                    | :heavy_check_mark: |                    |                    |
| [MHNF](./openhgnn/output/MHNF)[arxiv]                    | :heavy_check_mark: |                    |                    |
| [HGSL](./openhgnn/output/HGSL)[AAAI 2021]                | :heavy_check_mark: |                    |                    |
| [HGNN-AC](./openhgnn/output/HGNN_AC)[WWW 2021]           | :heavy_check_mark: |                    |                    |
| [HeCo](./openhgnn/output/HeCo)[KDD 2021]                 | :heavy_check_mark: |                    |                    |
| [SimpleHGN](./openhgnn/output/HGT)[KDD 2021]             | :heavy_check_mark: |                    |                    |
| [HPN](./openhgnn/output/HPN)[TKDE 2021]                  | :heavy_check_mark: | :heavy_check_mark: |                    |
| [RHGNN](./openhgnn/output/RHGNN)[arxiv]                  | :heavy_check_mark: |                    |                    |
| [HDE](./openhgnn/output/HDE)[ICDM 2021]                  |                    | :heavy_check_mark: |                    |
| [HetSANN](./openhgnn/output/HGT)[AAAI 2020]              | :heavy_check_mark: |                    |                    |
| [ieHGCN](./openhgnn/output/HGT)[TKDE 2021]               | :heavy_check_mark: |                    |                    |
| [KTN](./openhgnn/output/KTN)[NIPS 2022]                  | :heavy_check_mark: |                    |                    |

### 候选模型

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification[EMNLP 2019]
- [Heterogeneous Information Network Embedding with Adversarial Disentangler[TKDE 2021]](https://ieeexplore.ieee.org/document/9483653)

## 贡献者

OpenHGNN团队[北邮 GAMMA 实验室]、DGL 团队和鹏城实验室。

[贡献者名单](./CONTRIBUTING.md)

## 引用OpenHGNN

欢迎在您的工作中用如下的方式引用OpenHGNN:

```
@inproceedings{han2022openhgnn,
  title={OpenHGNN: An Open Source Toolkit for Heterogeneous Graph Neural Network},
  author={Hui Han, Tianyu Zhao, Cheng Yang, Hongyi Zhang, Yaoqi Liu, Xiao Wang, Chuan Shi},
  booktitle={CIKM},
  year={2022}
}
```
