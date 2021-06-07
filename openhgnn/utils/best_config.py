BEST_CONFIGS = {
    "node_classification": {
        "han": {
            'general': {},
            'acm_han': {"lr": 0.005, "max_epoch": 1000},
            'acm_han_raw': {}
        },
        'hgt': {
            'general': {},
            'acm1': {'hidden_dim': 128, 'num_heads': 4, 'n_layers': 2}
        },
        'GTN': {
            'general': {'lr': 0.005, 'weight_decay': 0.001, 'hidden_dim': 64, 'max_epoch': 50, 'patience': 10,
                        'norm_emd_flag': True, 'mini_batch_flag': False},
            'acm4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'imdb4GTN': {
                'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True,
            },
            'dblp4GTN': {
                'num_layers': 3, 'num_channels': 2, 'adaptive_lr_flag': False,
            },
        },
        'HAN': {
            'general': {},
            'acm_han_raw': {
                'lr': 0.005, 'num_heads': [8], 'hidden_dim': 128, 'dropout': 0.6, 'weight_decay': 0.001,
                'num_epochs': 200, 'patience': 100
            },
            'acm4GTN': {
                'lr': 0.001, 'weight_decay': 0.00001, 'num_heads': [8, 8],
            },
            'imdb4GTN': {
                'lr': 0.001, 'weight_decay': 0.00001, 'num_heads': [2, 2, 2]
            }

        },
        'HetGNN': {
            'general': {'max_epoch': 500, 'patience': 20, 'mini_batch_flag': True},
            'academic4HetGNN': {
                'lr': 0.001, 'weight_decay': 0.00001, 'dim': 128, 'batch_size': 64, 'window_size': 5,
                'batches_per_epoch': 50, 'rw_length': 50, 'rw_walks': 10, 'rwr_prob': 0.5,
            }
        },
        'MAGNN': {
            'general': {'mini_batch_flag': False, 'max_epoch': 500, 'patience': 30},
            'imdb': {
                'lr': 0.005, 'weight_decay': 0.001, 'dropout': 0.5, 'hidden_dim': 64, 'encoder_type': 'RotateE',
                'inter_attn_feats': 128, 'num_heads': 8, 'num_layers': 2,
            },
        },
        'NSHE': {
            'general': {},
            'acm4SNHE': {'lr': 0.005, 'weight_decay': 0.001, 'num_e_neg': 1, 'num_ns_neg': 4,
                         'max_epoch': 500, 'patience': 10,
                         }
        },
        'RSHN':{
            'general': {},
            'aifb': {'hidden_dim': 8, 'num_node_layer': 2, 'num_edge_layer': 1, 'rw_len': 4, 'batch_size': 5000,
                     'dropout': 0.6
                     },
            'mutag': {'hidden_dim': 8, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000
                },
            'bgs': {'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000
                      },
            'am': {'hidden_dim': 16, 'num_node_layer': 2, 'num_edge_layer': 2, 'rw_len': 4, 'batch_size': 5000
                      },
        }

    },
    "entity_classification": {
        "deepwalk": {
            "general": {},
        },
    },
    "link_prediction": {
        'general': {'max_epoch': 500, 'patience': 10, 'mini_batch_flag': True},
        'academic4HetGNN': {
            'lr': 0.01, 'weight_decay': 0.0001, 'dim': 128, 'batch_size': 64, 'window_size': 5,
            'batches_per_epoch': 50, 'rw_length': 50, 'rw_walks': 10, 'rwr_prob': 0.5,
        }
    }
}
