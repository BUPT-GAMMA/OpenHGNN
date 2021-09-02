import yaml
from random import choice
from random import randint
from random import random
from random import uniform
import os
hidden_dim = choice([64, 128])
layers_pre_mp = choice([1,2,3])
layers_post_mp = choice([1,2,3])
layers_gnn = choice([1,2,3])
stage_type = choice(['stack'])
activation = choice(['relu'])
has_bn = choice([True, False])
gnn_type = choice(['gcnconv'])
has_l2norm = choice([True, False])
mini_batch_flag = choice([True, False])
macro_func = choice(['attention'])
num_layers = choice([1,2,3])
num_heads = choice([4,6,8])
dropout = random()
lr = uniform(0.0001,0.1)
weight_decay = uniform(0.000001, 0.001)
patience = randint(10,100)
max_epoch = randint(100,500)
slope = uniform(0.001,0.1)

aproject = {'node_classification':
                {
                    'HGBn-ACM':
                    {
                        'homo_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },

                        'relation_HGNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },

                        'mp_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                                'macro_func':macro_func,
                            },

                        'GCN':
                            {
                                'hidden_dim':hidden_dim,
                                'num_layers':num_layers,
                                'activation':activation,
                                'dropout':dropout,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },

                        'GAT':
                            {
                                'hidden_dim':hidden_dim,
                                'num_layers':num_layers,
                                'dropout':dropout,
                                'slope':slope,
                                'num_heads':num_heads,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },

                    },

                    'HGBn-DBLP':
                    {
                        'homo_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },
                        'mp_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                                'macro_func':macro_func,
                            },
                        'GCN':
                            {
                                'hidden_dim':hidden_dim,
                                'num_layers':num_layers,
                                'activation':activation,
                                'dropout':dropout,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },
                    },

                    'HGBn-Freebase':
                    {
                        'homo_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },
                        'mp_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                                'macro_func':macro_func,
                            },
                    },

                    'HGBn-IMDB':
                        {
                            'homo_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },
                        'mp_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                                'macro_func':macro_func,
                            },
                        },

                },


            'link_prediction':
                {
                    'HGBl-amazon':
                        {
                            'homo_GNN':
                            {
                                'hidden_dim':hidden_dim,
                                'layers_pre_mp':layers_pre_mp,
                                'layers_post_mp':layers_post_mp,
                                'layers_gnn':layers_gnn,
                                'stage_type':stage_type,
                                'activation':activation,
                                'dropout':dropout,
                                'has_bn':has_bn,
                                'gnn_type':gnn_type,
                                'has_l2norm':has_l2norm,
                                'lr':lr,
                                'weight_decay':weight_decay,
                                'patience':patience,
                                'max_epoch':max_epoch,
                                'mini_batch_flag':mini_batch_flag,
                            },
                        },
                }


            }

fileNamePath = os.path.split(os.path.realpath(__file__))[0]
yamlPath = os.path.join(fileNamePath, 'config2.yaml')

f = open(yamlPath,'w')
print(yaml.dump(aproject,f))