from openhgnn import Experiment


def search_space(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64]),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        'num_layers': trial.suggest_int('num_layers', 2, 3)
    }


if __name__ == '__main__':
    experiment = Experiment(model='RGCN', dataset='acm4GTN', task='node_classification', gpu=-1,
                            hpo_search_space=search_space, hpo_trials=20)
    experiment.run()
