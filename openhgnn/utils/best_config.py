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
                'num_layers': 2, 'num_heads': 2, 'adaptive_lr_flag': True,
            }
        }

    },
    "entity_classification": {
        "deepwalk": {
            "general": {},
        },
    },

}
