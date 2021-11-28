# Design Space for Heterogeneous Graph Neural Network

Following [GraphGym](https://github.com/snap-stanford/GraphGym), we release a platform Space4HGNN for designing and evaluating Heterogeneous Graph Neural Networks (HGNN). It is implemented with PyTorch and DGL, using the OpenHGNN package.

## How to run

### Install

The installation process is same with OpenHGNN [Get Started](https://github.com/BUPT-GAMMA/OpenHGNN#get-started).

### Run a single experiment

#### Generate designs randomly

```bash
python space4hgnn/generate_yaml.py --gnn_type gcnconv --times 1 --key has_bn --configfile test
```

``--aggr -a``, specify the gnn type, and gcnconv, gatconv, sageconv, ginconv are optional.

``--times -t``, the ID of yaml file to control different random sampling.

``--key -k``, specify  a design dimension.

``--configfile -c``, specify a directory name to store configure yaml file.

#### Run

```bash
python space4hgnn.py -m general_HGNN -u metapath -t node_classification -d HGBn-ACM -g 0 -r 5 -a gcnconv -s 1 -k has_bn -v True -c test -p HGB
```

``--model -m ``  name of models

``--subgraph_extraction -u`` subgraph extraction methods

``--task -t`` name of task

``--dataset -t`` name of dataset

``--gpu -g`` controls which gpu you will use. If you do not have gpu, set -g -1.

``--repeat -r`` times to repeat, default 5

``--gnn_type -a `` gun type. 

``--times -t`` same with generating random designs

``--key -k`` a design dimension

``--value -v`` the value of ``key`` design dimension

``--configfile -c  `` load the yaml file which is in the directory configfile

``--predictfile -p`` The file path to store predict files.

e.g.: 

We implement three model families in Space4HGNN, Homogenezation model family, Relation model family, Meta-path model family.

For Homogenization model family, we can omit the ``--subgraph_extraction``

```bash
python space4hgnn.py -m homo_GNN -t node_classification -d HGBn-ACM -g 0 -r 5 -a gcnconv -s 1 -k has_bn -v True -c test -p HGB
```

For Relation model family, ``--model`` is general_HGNN and ``--subgraph_extraction`` is relation

```bash
python space4hgnn.py -m general_HGNN -u relation -t node_classification -d HGBn-ACM -g 0 -r 5 -a gcnconv -s 1 -k has_bn -v True -c test -p HGB
```

For Meta-path model family, ``--model`` is general_HGNN and ``--subgraph_extraction`` is meta-path

```bash
python space4hgnn.py -m general_HGNN -u relation -t node_classification -d HGBn-ACM -g 0 -r 5 -a gcnconv -s 1 -k has_bn -v True -c test -p HGB
```

**Note**: If you are interested in some model, you can refer to the below models list.

Refer to the [docs](https://openhgnn.readthedocs.io/en/latest/index.html) to get more basic and depth usage.

