# 异质图神经网络开源工具包 

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BUPT-GAMMA/OpenHGNN)
[![Documentation Status](https://readthedocs.org/projects/openhgnn-zh-cn/badge/?version=latest)](https://openhgnn.readthedocs.io/zh_CN/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/BUPT-GAMMA/OpenHGNN)

 [**Github Community (English)**](./README.md)

OpenHGNN是一个基于 [DGL [Deep Graph Library]](https://github.com/dmlc/dgl) 和 [PyTorch](https://pytorch.org/) 的开源异质图神经网络工具包, 集成了异质图神经网络的前沿模型.

## 新闻

我们于启智社区开源了v0.1.1中文版本。

启智社区用户可以享受到如下功能：

- 全新的中文文档
- 免费的计算资源, 云脑使用教程
- OpenHGNN最新功能

## 关键特性

- 易用: OpenHGNN提供了了易用的接口在给定的模型和数据集上运行实验, 且集成了 [optuna](https://optuna.org/) 进行超参数优化.
- 可扩展: 用户可以定义定制化的任务/模型/数据集来对新的场景应用新的模型.
- 高效: 底层的DGL框架提供了提供了高效的API.

## 开始使用

#### 环境要求

- Python  >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/)  >= 1.7.1
- [DGL](https://github.com/dmlc/dgl) >= 0.7.0

- CPU 或者 NVIDIA GPU, Linux, Python3

**1. Python 环境 (可选):** 推荐使用 Conda 包管理

```bash
conda create -n openhgnn python=3.7
source activate openhgnn
```

**2. Pytorch:** 安装Pytorch, 参考[PyTorch安装文档](https://pytorch.org/get-started/locally/).

```bash
# CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. DGL:** 安装 DGL, 参考[DGL安装文档](https://www.dgl.ai/pages/start.html).

```bash
# CUDA versions: cpu, cu101, cu102, cu110, cu111
pip install --pre dgl-cu101 -f https://data.dgl.ai/wheels-test/repo.html
```

**4. 下载OpenHGNN, 安装依赖:**

```bash
git clone https://github.com/BUPT-GAMMA/OpenHGNN
cd OpenHGNN
pip install -r requirements.txt
```

#### 在已有的评测上运行已有的基线模型 [数据集](./openhgnn/dataset/#Dataset)

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --use_best_config --load_from_pretrained
```

使用方法: main.py [-h] [--model MODEL] [--task TASK] [--dataset DATASET]
               [--gpu GPU] [--use_best_config]

*可选参数*:

``-h, --help``	展示帮助信息并退出

``--model -m ``	模型名

``--task -t``	任务名

``--dataset -d``    数据集名

``--gpu -g``	控制你使用哪一个GPU, 如果没有GPU, 设定 -g -1.

``--use_best_config``	use_best_config 意味着你使用该模型在该数据集下最优的配置, 如果你想要设定不同的超参数,请手动修改 [配置文件](./openhgnn/config.ini) . 使用最佳配置会覆盖配置文件中的参数。

``--use_hpo`` 除了 use_best_config, 我们还提供了一个超参数的 [样例](./openhgnn/auto) 来自动查找最佳超参数.

``--load_from_pretrained`` 从默认检查点加载模型.

示例: 

```bash
python main.py -m GTN -d imdb4GTN -t node_classification -g 0 --use_best_config
```

**提示**: 如果你对某个模型感兴趣,你可以参考下列的模型列表.

请参考 [文档](https://openhgnn.readthedocs.io/en/latest/index.html) 了解更多的基础和进阶的使用方法.

## [模型](./openhgnn/models/#Model)

### 特定任务下支持的模型

表格中的链接给出了模型的基本使用方法.

| 模型                                                      | 节点分类             | 链路预测             | 推荐               |
| -------------------------------------------------------- | ------------------- | ------------------ | ------------------ |
| [Metapath2vec](./openhgnn/output/metapath2vec)[KDD 2017] | :heavy_check_mark:  |                    |                    |
| [RGCN](./openhgnn/output/RGCN)[ESWC 2018]                | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [HERec](./openhgnn/output/HERec)[TKDE 2018]              | :heavy_check_mark:  |                    |                    |
| [HAN](./openhgnn/output/HAN)[WWW 2019]                   | :heavy_check_mark:  |                    |                    |
| [KGCN](./openhgnn/output/KGCN)[WWW 2019]                 |                     |                    | :heavy_check_mark: |
| [HetGNN](./openhgnn/output/HetGNN)[KDD 2019]             | :heavy_check_mark:  | :heavy_check_mark: |                    |
| HGAT[EMNLP 2019]                                         |                     |                    |                    |
| [GTN](./openhgnn/output/GTN)[NeurIPS 2019]               | :heavy_check_mark:  |                    |                    |
| [RSHN](./openhgnn/output/RSHN)[ICDM 2019]                | :heavy_check_mark:  |                    |                    |
| [DMGI](./openhgnn/output/DMGI)[AAAI 2020]                | :heavy_check_mark:  |                    |                    |
| [MAGNN](./openhgnn/output/MAGNN)[WWW 2020]               | :heavy_check_mark:  |                    |                    |
| HGT[WWW 2020]                                            |                     |                    |                    |
| [CompGCN](./openhgnn/output/CompGCN)[ICLR 2020]          | :heavy_check_mark:  | :heavy_check_mark: |                    |
| [NSHE](./openhgnn/output/NSHE)[IJCAI 2020]               | :heavy_check_mark:  |                    |                    |
| [NARS](./openhgnn/output/NARS)[arxiv]                    | :heavy_check_mark:  |                    |                    |
| [MHNF](./openhgnn/output/MHNF)[arxiv]                    | :heavy_check_mark:  |                    |                    |
| [HGSL](./openhgnn/output/HGSL)[AAAI 2021]                | :heavy_check_mark:  |                    |                    |
| [HGNN-AC](./openhgnn/output/HGNN_AC)[WWW 2021]           | :heavy_check_mark:  |                    |                    |
| [HeCo](./openhgnn/output/HeCo)[KDD 2021]                 | :heavy_check_mark:  |                    |                    |
| SimpleHGN[KDD 2021]                                      |                     |                    |                    |
| [HPN](./openhgnn/output/HPN)[TKDE 2021]                  | :heavy_check_mark:  |                    |                    |
| [RHGNN](./openhgnn/output/RHGNN)[arxiv]                  | :heavy_check_mark:  |                    |                    |
| [HDE](./openhgnn/output/HDE)[ICDM 2021]                  |                     | :heavy_check_mark: |                    |
|                                                          |                     |                    |                    |

### 候选模型

- Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification[EMNLP 2019]
- [Heterogeneous Information Network Embedding with Adversarial Disentangler[TKDE 2021]](https://ieeexplore.ieee.org/document/9483653)

## 贡献者

OpenHGNN 团队 [北邮 GAMMA 实验室] 和 DGL 团队.

[贡献者名单](./CONTRIBUTING.md).

