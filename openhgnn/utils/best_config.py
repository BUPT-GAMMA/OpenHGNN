BEST_CONFIGS = {
    "node_classification": {
        "han": {
            'general': {},
            'acm_han': {"lr": 0.005, "max_epoch": 1000},
            'acm_han_raw': {}
        },
        'hgt':{
            'general': {},
            'acm1': {'hidden_dim': 128, 'num_heads':4, 'n_layers':2}
        },
        'GTN':{
            'general': {},
            'acm4GTN': {
                'hidden_dim': 64, 'num_layers': 2, 'num_channels': 2, 'adaptive_lr_flag': True, 'norm_emd_flag': True
            },
            'imdb4GTN':{
                'hidden_dim': 64, 'num_layers': 3, 'num_channels': 2, 'adaptive_lr_flag': True, 'norm_emd_flag': True
            },
            'dblp4GTN': {
                'hidden_dim': 64, 'num_layers': 3, 'num_channels': 2, 'adaptive_lr_flag': False, 'norm_emd_flag': True
            },
        }

    },
    "entity_classification": {
        "deepwalk": {
            "general": {},
        },
    },

}
