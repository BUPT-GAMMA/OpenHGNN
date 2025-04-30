# RMR

Paper:  [Reserving-Masking-Reconstruction Model for Self-Supervised Heterogeneous Graph Representation](https://dl.acm.org/doi/pdf/10.1145/3637528.3671719 )

The open source code by the author : https://github.com/DuanhaoranCC/RMR

## How to run

First of all ,you need to clone the Openhgnn-DGL.Then open the Openhgnn-DGL，and make sure to open the terminal in this folder and ensure that the Python global environment or the virtual environment launched by conda supports it.

my env:

```
absl-py                 2.2.2
alembic                 1.15.2
annotated-types         0.7.0
asttokens               3.0.0
certifi                 2025.1.31
charset-normalizer      3.4.1
colorama                0.4.6
colorlog                6.9.0
comm                    0.2.2
debugpy                 1.8.5
decorator               5.2.1
dgl                     2.2.1+cu118
exceptiongroup          1.2.2
executing               2.2.0
filelock                3.13.1
fsspec                  2024.6.1
greenlet                3.1.1
grpcio                  1.71.0
idna                    3.10
igraph                  0.11.8
importlib_metadata      8.6.1
intel-openmp            2021.4.0
ipykernel               6.29.5
ipython                 8.18.1
isodate                 0.7.2
jedi                    0.19.2
Jinja2                  3.1.4
joblib                  1.4.2
jupyter_client          8.6.2
jupyter_core            5.7.2
littleutils             0.2.4
lmdb                    1.6.2
Mako                    1.3.10
Markdown                3.8
MarkupSafe              2.1.5
matplotlib-inline       0.1.7
mkl                     2021.4.0
mpmath                  1.3.0
nest-asyncio            1.6.0
networkx                3.2.1
numpy                   1.26.3
ogb                     1.3.6
openhgnn                0.7.0
optuna                  4.3.0
ordered-set             4.1.0
outdated                0.2.2
packaging               24.2
pandas                  2.2.3
parso                   0.8.4
pillow                  11.0.0
pip                     25.0
platformdirs            4.3.2
prompt_toolkit          3.0.50
protobuf                6.30.2
psutil                  7.0.0
pure_eval               0.2.3
pydantic                2.11.3
pydantic_core           2.33.1
Pygments                2.19.1
pyparsing               3.2.3
python-dateutil         2.9.0.post0
pytz                    2025.2
pywin32                 306
PyYAML                  6.0.2
pyzmq                   26.2.0
rdflib                  7.1.4
requests                2.32.3
scikit-learn            1.6.1
scipy                   1.13.1
setuptools              75.8.0
six                     1.17.0
SQLAlchemy              2.0.40
stack-data              0.6.3
sympy                   1.13.1
tbb                     2021.11.0
tensorboard             2.19.0
tensorboard-data-server 0.7.2
texttable               1.7.0
threadpoolctl           3.6.0
torch                   2.3.0+cu118
torchaudio              2.3.0+cu118
torchdata               0.9.0
torchvision             0.18.0+cu118
tornado                 6.4.1
tqdm                    4.67.1
traitlets               5.14.3
typing_extensions       4.12.2
typing-inspection       0.4.0
tzdata                  2025.2
urllib3                 2.4.0
wcwidth                 0.2.13
Werkzeug                3.1.3
wheel                   0.45.1
zipp                    3.21.0
```

Special note, as my computer system is **Windows 11**, there are currently no official updates based on the latest Windows system for `DGL`. Features such as `GraphBolt` are restricted from use in my virtual environment

To test this model，you can both **use the instruction** or **create a new test.py file** to try diy coding.

### Using the  instruction

Ensure you have activated your virtual environment,than run the instruction:

```
python main.py -m RMR -t node_classification -d acm4RMR -g 0 
```

if you do not have gpu, you can set `-g -1`

**Candidate dataset:** 

​	`aminer4RMR`/`imdb4RMR`



### create a new test.py file

you should create the test.py in the dir of `Openhgnn-DGL` directly, and than enter the following content:

```
from openhgnn import Experiment
experiment = Experiment(model='RMR', dataset='acm4RMR', task='node_classification',graphbolt=False ,use_distributed=False,gpu = 0)
# experiment = Experiment(model='RMR', dataset='imdb4RMR', task='node_classification',graphbolt=False ,use_distributed=False,gpu = 0)
# experiment = Experiment(model='RMR', dataset='aminer4RMR', task='node_classification',graphbolt=False ,use_distributed=False,gpu = 0)
experiment.run()
```



## Performance

The paper involves a total of 5 datasets, one of which the author did not have an open-source version with added labels, so it was not supported here. Another dataset has a single image partition of over 700 MB, which is too large. It is estimated that eval+train would take several hours to train it,so it was also not supported here  Currently, three datasets have been reproduced.

This model focus on  the node classification task:

| Datasets | Metric | Split | Paper | Openhgnn |
| -------- | ------ | ----- | ----- | -------- |
| imdb4RMR | Ma-F1  | 1     | 33.23 | 36.97    |
|          |        | 5     | 41.95 | 47.24    |
|          |        | 10    | 46.86 | 46.96    |
|          |        | 20    | 50.75 | 48.13    |
|          | Mi-F1  | 1     | 39.47 | 41.28    |
|          |        | 5     | 41.77 | 47.20    |
|          |        | 10    | 46.95 | 47.01    |
|          |        | 20    | 50.89 | 48.04    |
|          | AUC    | 1     | 56.19 | 60.29    |
|          |        | 5     | 59.33 | 64.36    |
|          |        | 10    | 65.58 | 63.32    |
|          |        | 20    | 68.97 | 63.63    |



| Datasets | Metric | Split | Paper | Openhgnn |
| -------- | ------ | ----- | ----- | -------- |
| acm4RMR  | Ma-F1  | 1     | 56.53 | 48.12    |
|          |        | 5     | 86.46 | 86.64    |
|          |        | 10    | 87.14 | 85.66    |
|          |        | 20    | 87.63 | 87.48    |
|          | Mi-F1  | 1     | 60.39 | 48.28    |
|          |        | 5     | 86.27 | 86.45    |
|          |        | 10    | 86.82 | 85.21    |
|          |        | 20    | 87.58 | 87.50    |
|          | AUC    | 1     | 76.70 | 73.84    |
|          |        | 5     | 95.03 | 94.58    |
|          |        | 10    | 95.71 | 93.38    |
|          |        | 20    | 95.65 | 94.24    |



| Datasets   | Metric | Split | Paper | Openhgnn |
| ---------- | ------ | ----- | ----- | -------- |
| aminer4RMR | Ma-F1  | 1     | 55.94 | 50.19    |
|            |        | 5     | 75.68 | 73.8     |
|            |        | 10    | 82.50 | 84.01    |
|            |        | 20    | 90.81 | 87.87    |
|            | Mi-F1  | 1     | 53.02 | 50.83    |
|            |        | 5     | 79.75 | 79.43    |
|            |        | 10    | 87.35 | 87.95    |
|            |        | 20    | 93.01 | 90.56    |





## Dataset

Supported dataset: acm4RMR, imdb4RMR, aminer4RMR

You can download the dataset by

```
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/acm4RMR.zip
wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/imdb4RMR.zip
Wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/aminer4RMR.zip
```



## More

#### Contirbutor

Haolan Yang[GAMMA LAB]



If you have any questions,

Submit an issue or email to  2021212359@bupt.edu.cn.