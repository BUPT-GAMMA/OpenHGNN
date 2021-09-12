from ..trainerflow import build_flow
import optuna


def func_search(trial):
    return {
        "lr": trial.suggest_categorical("lr", [1e-3, 5e-3, 1e-2]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64]),
        "num_heads": trial.suggest_categorical("num_heads",[1, 2, 4]),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        'n_layers': trial.suggest_int('n_layers', 2, 3)
    }


def hpo_experiment(args, trainerflow, **kwargs):

    tool = AutoML(args, trainerflow, n_trials=10, func_search=func_search)
    result = tool.run()
    print("\nFinal results:\n")
    print(result)
    return result


class AutoML(object):
    """
    Args:
        func_search: function to obtain hyper-parameters to search
    """

    def __init__(self, args, trainerflow, n_trials=3, **kwargs):
        self.args = args
        self.trainerflow = trainerflow
        #self.seed = kwargs.pop("seed") if "seed" in kwargs else [1]
        assert "func_search" in kwargs
        self.func_search = kwargs["func_search"]
        self.metric = kwargs["metric"] if "metric" in kwargs else None
        self.n_trials = n_trials
        self.best_result = None
        self.best_params = None
        self.default_params = kwargs

    def _objective(self, trials):
        args = self.args
        cur_params = self.func_search(trials)
        for key, value in cur_params.items():
            args.__setattr__(key, value)
        flow = build_flow(args, self.trainerflow)
        result = flow.train()['Test_score']
        if isinstance(result, tuple):
            result = (result[0] + result[1]) / 2
        if self.best_result is None or result > self.best_result:
            self.best_result = result
            self.best_params = cur_params
        return result

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        print(study.best_params)
        return self.best_result
