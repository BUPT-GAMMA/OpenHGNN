BEST_CONFIGS = {
    "node_classification": {
        "han": {
            'general': {},
            'acm_han': {"lr": 0.005, "max_epoch": 1000},
            'acm_han_raw': {}
        },
        'hgt':{
            'general':{},
            'acm1': {'hidden_dim': 128, 'num_heads':4, 'n_layers':2}
        }

    },
    "entity_classification": {
        "deepwalk": {
            "general": {},
        },
    },

}
