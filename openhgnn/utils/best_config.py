BEST_CONFIGS = {
    "node_classification": {
        'RGCN': {
            'general': {
                'seed': 0, 'patience': 40, 'validation': True, 'evaluation_metric': 'acc'
            },
            'aifb': {
                'lr': 0.005, 'weight_decay': 0, 'max_epoch': 100,
                'hidden_dim': 32, 'n_bases': -1, 'num_layers': 2, 'batch_size': 126, 'dropout': 0,
                'mini_batch_flag': False, 'validation': True
            },
            'mutag': {
                'lr': 0.005, 'weight_decay': 0.0005, 'max_epoch': 50,
                'hidden_dim': 32, 'n_bases': 30, 'num_layers': 2, 'batch_size': 50, 'fanout': 4, 'dropout': 0.5,
                'mini_batch_flag': False, 'validation': True
            },
            'bgs': {
                'lr': 0.005, 'weight_decay': 0.0005, 'max_epoch': 50,
                'hidden_dim': 16, 'n_bases': 40, 'num_layers': 3, 'batch_size': 126, 'fanout': 4, 'dropout': 0.1,
                'mini_batch_flag': True, 'validation': True
            },
            'am': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 50,
                'hidden_dim': 16, 'n_bases': 40, 'num_layers': 2, 'batch_size': 126, 'fanout': 4, 'dropout': 0,
                'mini_batch_flag': True, 'validation': True
            },
            'ogbn-mag': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 10,
                'hidden_dim': 16, 'n_bases': -1, 'num_layers': 2, 'batch_size': 126, 'fanout': 4, 'dropout': 0,
                'mini_batch_flag': True, 'validation': True
            },
            'HGBn-ACM': {
                'evaluation_metric': 'f1'
            },
            'HGBn-DBLP': {
                'num_layers': 2, 'hidden_dim': 16, 'dropout': 0.5
            },
            'HGBn-Freebase': {
                'num_layers': 5, 'hidden_dim': 16, 'dropout': 0.5
            },
            'HGBn-IMDB': {
            },
            'ohgbn-Freebase': {
                'lr': 0.005, 'weight_decay': 0.0001, 'hidden_dim': 32, 'n_bases': 40, 'num_layers': 2, 'batch_size': 256,
                'fanout': 4, 'dropout': 0.2, 'mini_batch_flag': True,
            },
            'ohgbn-acm': {
                'lr': 0.01, 'hidden_dim': 64, 'dropout': 0.4, 'num_layers': 2, 'evaluation_metric': 'f1'
            },
            'ohgbn-imdb': {
                'lr': 0.01, 'hidden_dim': 128, 'dropout': 0.2, 'num_layers': 4, 'evaluation_metric': 'f1'
            },
            'ohgbn-yelp2': {'lr': 0.001, 'hidden_dim': 64, 'dropout': 0.3696343130196993, 'num_layers': 3,
                            'use_self_loop': True, 'n_bases': 3}
        },
        'CompGCN': {
            'general': {
                'seed': 0, 'patience': 100, 'validation': True, 'evaluation_metric': 'acc'
            },
            'aifb': {
                'lr': 0.01, 'weight_decay': 0.0001, 'max_epoch': 100,
                'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.2, 'comp_fn': 'sub', 'batch_size': 128,
                'mini_batch_flag': False, 'validation': True
            },
            'mutag': {
                'lr': 0.01, 'weight_decay': 0.0001, 'max_epoch': 100,
                'hidden_dim': 32, 'num_layers': 3, 'dropout': 0.5, 'comp_fn': 'sub',
                'mini_batch_flag': False, 'validation': True
            },
            'bgs': {

            },
            'am': {
                'lr': 0.001, 'weight_decay': 0.0001, 'max_epoch': 100, 'fanout': 4,
                'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.2, 'comp_fn': 'sub', 'batch_size': 128,
                'mini_batch_flag': True, 'validation': True
            },
            'ohgbn-Freebase': {
                'mini_batch_flag': True, 'batch_size': 128,
            },
            'ohgbn-acm': {'lr': 0.005, 'hidden_dim': 128, 'dropout': 0.16, 'num_layers': 2,
                          'comp_fn': 'sub', 'evaluation_metric': 'f1'},
            'ohgbn-imdb': {'lr': 0.001, 'hidden_dim': 64, 'num_layers': 3,
                           'comp_fn': 'sub', 'evaluation_metric': 'f1'}
        },
        'GTN': {
            'general': {'lr': 0.005, 'weight_decay': 0.001, 'hidden_dim': 128, 'max_epoch': 100, 'patience': 20,
                        'norm_emd_flag': True, 'mini_batch_flag': False},
            'acm4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'imdb4GTN': {
                'lr': 0.01, 'hidden_dim': 128, 'num_layers': 2, 'num_channels': 8, 'adaptive_lr_flag': True,
                'dropout': 0.1
            },
            'dblp4GTN': {
                'num_layers': 3, 'num_channels': 2, 'adaptive_lr_flag': False,
            },
            'ohgbn-acm': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'ohgbn-imdb': {
                'lr': 0.001, 'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            }
        },
        'fastGTN': {
            'general': {'lr': 0.005, 'weight_decay': 0.001, 'hidden_dim': 64, 'max_epoch': 100, 'patience': 20,
                        'norm_emd_flag': True, 'mini_batch_flag': False},
            'acm4GTN': {
                'lr': 0.01, 'hidden_dim': 32, 'num_layers': 3, 'num_channels': 8, 'adaptive_lr_flag': False,
                'dropout': 0.01,
            },
            'imdb4GTN': {
                'lr': 0.01, 'num_layers': 3, 'dropout': 0.3, 'num_channels': 4, 'hidden_dim': 128,
                'adaptive_lr_flag': False,
            },
            'dblp4GTN': {'lr': 0.001, 'hidden_dim': 128, 'num_layers': 3, 'num_channels': 8, 'adaptive_lr_flag': False,
                         'dropout': 0.5},
        },
        'MHNF': {
            'general': {'lr': 0.005, 'weight_decay': 0.001, 'hidden_dim': 64, 'max_epoch': 50, 'patience': 10,
                        'norm_emd_flag': True, 'mini_batch_flag': False},
            'acm4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'imdb4GTN': {
                'lr': 0.001, 'hidden_dim': 128, 'num_layers': 4, 'num_channels': 4
            },
            'dblp4GTN': {
                'num_layers': 3, 'num_channels': 2, 'adaptive_lr_flag': False,
            },
            'ohgbn-acm': {
                'lr': 0.005, 'hidden_dim': 64, 'weight_decay': 0.005, 'num_layers': 2, 'num_channels': 3,
                'norm_emd_flag': False, 'identity': False, 'adaptive_lr_flag': True,
            },
            'ohgbn-imdb': {
                'lr': 0.1, 'hidden_dim': 32, 'weight_decay': 0.01, 'num_layers': 4, 'num_channels': 2,
                'identity': False, 'adaptive_lr_flag': True,
            },
        },
        'HAN': {
            'general': {},
            'acm_han_raw': {
                'lr': 0.005, 'num_heads': [8], 'hidden_dim': 128, 'dropout': 0.6, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100
            },
            'acm4GTN': {
                'lr': 0.01, 'hidden_dim': 128, 'dropout': 0.46, 'num_heads': [1]
            },
            'imdb4GTN': {
                'lr': 0.01, 'hidden_dim': 64, 'dropout': 0.24, 'num_heads': [2, 2]
            },
            'HGBn-ACM': {
            },
            'ohgbn-acm': {'num_heads': [8, 8]},
            'ohgbn-imdb': {'lr': 0.005, 'hidden_dim': 32, 'dropout': 0.5}

        },
        'HPN': {
            'general': {},
            'ohgbn-acm': {
                'lr': 0.01, 'hidden_dim': 64, 'dropout': 0.2, 'k_layer': 3, 'alpha': 0.2, 'edge_drop': 0
            },
            'ohgbn-imdb': {
                'lr': 0.005, 'hidden_dim': 128, 'dropout': 0.25
            }
        },
        'NARS': {
            'general': {},
            'HGBn-ACM': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 3, 'hidden_dim': 64, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'evaluation_metric': 'acc', 'ff_layer': 2,
            },
            'HGBn-DBLP': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 3, 'hidden_dim': 512, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'evaluation_metric': 'acc', 'ff_layer': 2,
            },
            'HGBn-Freebase': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 2, 'hidden_dim': 64, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'evaluation_metric': 'acc', 'ff_layer': 2,
            },
            'HGBn-IMDB': {
                'lr': 0.001, 'num_heads': [8], 'num_hops': 2, 'hidden_dim': 64, 'dropout': 0.5, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100, 'evaluation_metric': 'acc', 'ff_layer': 2,
            },
            'ohgbn-acm': {'lr': 0.001, 'hidden_dim': 32, 'dropout': 0.4, 'num_heads': 2, 'num_hops': 2,
                          'evaluation_metric': 'f1'},
            'ohgbn-imdb': {'lr': 0.01, 'hidden_dim': 128, 'dropout': 0.4, 'num_heads': 1, 'num_hops': 2,
                           'evaluation_metric': 'f1'}

        },
        'HetGNN': {
            'general': {'max_epoch': 500, 'patience': 20, 'mini_batch_flag': True},
            'academic4HetGNN': {
                'lr': 0.001, 'weight_decay': 0.00001, 'dim': 128, 'batch_size': 64, 'window_size': 5,
                'batches_per_epoch': 50, 'rw_length': 50, 'rw_walks': 10, 'rwr_prob': 0.5,
            }
        },
        'MAGNN': {
            'general': {'lr': 0.005, 'weight_decay': 0.001, 'dropout': 0.5,
                        'h_dim': 64, 'encoder_type': 'RotateE', 'inter_attn_feats': 128, 'num_heads': 8,
                        'num_layers': 2},
            'imdb4MAGNN': {
                'max_epoch': 100, 'patience': 30, 'out_dim': 3, 'mini_batch_flag': False
            },
            'dblp4MAGNN': {
                'max_epoch': 10, 'patience': 5, 'mini_batch_flag': True, 'batch_size': 8, 'num_samples': 5,
                'num_workers': 24, 'out_dim': 4
            }
        },
        'NSHE': {
            'general': {},
            'acm4NSHE': {'weight_decay': 0.001, 'num_e_neg': 1, 'num_ns_neg': 4,
                         'max_epoch': 500, 'patience': 10,
                         }
        },
        'HeCo': {
            'general': {
                'hidden_dim': 64, 'max_epoch': 10000, 'mini_batch_flag': False
            },
            'acm4HeCo': {'seed': 84, 'eva_lr': 0.05, 'eva_wd': 0, 'patience': 5, 'learning_rate': 0.0008,
                         'weight_decay': 0, 'tau': 0.8, 'feat_drop': 0.3, 'attn_drop': 0.5,
                         'sample_rate': {'author': 7, "subject": 3}, 'lam': 0.5
                         }
        },
        'DMGI': {
            'general': {
                'seed': 0, 'sc': 3, 'learning_rate': 0.0005
            },
            'acm_han_raw': {'weight_decay': 0.0001, 'dropout': 0.5, 'reg_coef': 0.001,
                            'patience': 20, 'sup_coef': 0.1, 'hid_unit': 64,
                            'isSemi': False, 'isBias': False, 'isAttn': False
                            },
            'imdb4GTN': {'weight_decay': 0.0001, 'dropout': 0.5, 'reg_coef': 0.1,
                         'patience': 50, 'sup_coef': 0.1, 'hid_unit': 64,
                         'isSemi': False, 'isBias': False, 'isAttn': False
                         }
        },
        'RSHN': {
            'general': {},
            'aifb': {
                'in_dim': 16, 'hidden_dim': 16, 'num_node_layer': 2,
                'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000,
            },
            'mutag': {
                'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000,
                'dropout': 0.2,
            },
            'bgs': {
                'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000
            },
            'am': {'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000
                   },
            'ohgbn-acm': {'lr': 0.001, 'hidden_dim': 128, 'dropout': 0.4, 'num_channels': 2, 'num_node_layer': 2,
                          'num_edge_layer': 2},
            'ohgbn-imdb': {'lr': 0.01, 'hidden_dim': 128, 'dropout': 0.09, 'num_node_layer': 2, 'num_edge_layer': 2
                           }
        },
        'HGSL': {
            'general': {},
            'acm4GTN': {
                'undirected_relations': 'author-paper,paper-subject', 'gnn_dropout': 0, 'fs_eps': 0.8,
                'fp_eps': 0.2, 'mp_eps': 0.6,
            }
        },
        'HGT': {
            'general': {},
            'ohgbn-acm': {
                'lr': 0.001, 'hidden_dim': 128, 'dropout': 0.1, 'num_layers': 2, 'num_heads': 1
            },
            'ohgbn-imdb': {'lr': 0.001, 'hidden_dim': 128, 'dropout': 0.3, 'num_layers': 2, 'num_heads': 4},

        },
        'SimpleHGN': {
            'general': {},
            'ohgbn-acm': {
                'lr': 0.0005, 'hidden_dim': 64, 'num_layers': 3, 'edge_dim': 32, 'num_heads': 2, 'num_edge': 4,
                'attn_drop_rate': 0.1, 'feats_drop_rate': 0.5, 'beta': 0.05, 'residual': False, 'slope': 0.01,
                'max_epoch': 200, 'patience': 50
            },
            'ohgbn-imdb': {'lr': 0.001, 'hidden_dim': 32, 'num_layers': 2, 'edge_dim': 32, 'num_heads': 8,
                           'num_edge': 8,
                           'attn_drop_rate': 0.34, 'feats_drop_rate': 0.4, 'beta': 0.01,
                           'residual': False, 'slope': 0.1, 'max_epoch': 100, 'patience': 50},

        }

    },

    "link_prediction": {
        'NARS': {
            'general': {'num_hops': 3},
        },
        'HetGNN': {
            'general': {'max_epoch': 500, 'patience': 10, 'mini_batch_flag': True},
            'academic4HetGNN': {
                'lr': 0.01, 'weight_decay': 0.0001, 'dim': 128, 'batch_size': 64, 'window_size': 5,
                'batches_per_epoch': 50, 'rw_length': 50, 'rw_walks': 10, 'rwr_prob': 0.5,
            }
        },
        'HAN': {
            'general': {

            },
            'HGBl-PubMed': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 400,
                'hidden_dim': 64, 'n_bases': 40, 'num_layers': 4, 'batch_size': 126, 'fanout': 4, 'dropout': 0.5,
                'validation': True, 'feat': 2, 'seed': 4,
            },
        },
        'CompGCN': {
            'general': {
            },
            'ohgbl-yelp1': {
                'mini_batch_flag': True, 'num_layers': 2,
            },
            'ohgbl-yelp2': {
                'mini_batch_flag': True, 'num_layers': 2, 'batch_size': 1024
            },
            'ohgbl-MTWM': {
                'weight_decay': 0.00001, 'max_epoch': 50, 'patience': 50,
                'mini_batch_flag': True, 'num_layers': 2,
            }
        },
        'RGCN': {
            'general': {
            },
            'HGBl-amazon': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 400,
                'hidden_dim': 64, 'n_bases': 40, 'num_layers': 4, 'batch_size': 126, 'fanout': 4, 'dropout': 0.5,
                'validation': True, 'feat': 2, 'seed': 0,
            },
            'HGBl-PubMed': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 400,
                'hidden_dim': 64, 'n_bases': 40, 'num_layers': 4, 'batch_size': 126, 'fanout': 4, 'dropout': 0.5,
                'validation': True, 'feat': 2, 'seed': 4,
            },
            'HGBl-LastFM': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 400,
                'hidden_dim': 64, 'n_bases': 40, 'num_layers': 4, 'batch_size': 126, 'fanout': 4, 'dropout': 0.5,
                'validation': True, 'feat': 2, 'seed': 3,
            },
            'FB15k-237': {
                'lr': 0.01, 'weight_decay': 0.0005, 'max_epoch': 100,
                'hidden_dim': 16, 'n_bases': 40, 'num_layers': 2, 'batch_size': 126, 'fanout': 4, 'dropout': 0,
                'validation': True
            },
            'ohgbl-MTWM': {
                'mini_batch_flag': True, 'num_layers': 2, 'batch_size': 12800
            },
            'ohgbl-yelp1': {
                'mini_batch_flag': True, 'num_layers': 2, 'batch_size': 102400
            },
            'ohgbl-yelp2': {
                'mini_batch_flag': True, 'num_layers': 2,
            }
        },
        'TransE': {
            'general':{
            },
            'FB15k':{
                'lr':1, 'weight_decay': 0.0001, 'hidden_dim': 400, 'neg_size': 13, 'margin': 4, 'batch_size': 100, 'valid_percent':0.01, 'test_percent': 0.1
            },
            'wn18':{
                # 'lr': 0.71, 'weight_decay': 0.0001, 'hidden_dim': 400, 'neg_size': 97, 'margin': 32.8, 'batch_size': 100, 'dis_norm': 1, 'valid_percent':1, 'test_percent': 1
                'lr': 1, 'weight_decay': 0.0001, 'hidden_dim': 400, 'neg_size': 100, 'margin': 50, 'batch_size': 100, 'dis_norm': 2, 'patience': 3, 'valid_percent':0.05, 'test_percent': 0.05
            }
        },
        'TransH': {
            'general':{
            },
            'FB15k':{
                'lr': 1, 'weight_decay': 0.0001, 'hidden_dim': 400, 'neg_size': 15, 'margin': 5, 'batch_size': 100, 'valid_percent':0.01, 'test_percent': 0.1
            },
            'wn18':{
                'lr': 0.71, 'weight_decay': 0.0001, 'hidden_dim': 200, 'neg_size': 100, 'margin': 32.8, 'batch_size': 100, 'patience': 5, 'valid_percent':1, 'test_percent': 1
            }
        },
        'TransD': {
            'general':{
            },
            'FB15k':{
                'lr': 0.6, 'weight_decay': 0.0001, 'ent_dim': 400, 'rel_dim': 400, 'neg_size': 20, 'margin': 5, 'batch_size': 100, 'patience': 3, 'valid_percent':0.01, 'test_percent': 0.1, 
            },
            'wn18':{
                'lr': 0.5, 'weight_decay': 0.0001, 'ent_dim': 400, 'rel_dim': 400, 'neg_size': 98, 'margin': 200, 'batch_size': 100, 'patience':5, 'valid_percent':1, 'test_percent': 1
            }
        }

    },
    "recommendation": {
        'KGCN': {
            "general": {},
            'LastFM4KGCN': {
                'in_dim': 16, 'hidden_dim': 16, 'n_relation': 60,
                'batch_size': 128, 'lr': 0.002
            },
        }
    },
}
