def func_search(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
        "hidden_size": trial.suggest_categorical("hidden_size", [8, 16, 32]),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        'num_layers': trial.suggest_int('num_layers', 2, 3)
    }