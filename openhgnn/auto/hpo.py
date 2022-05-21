from ..trainerflow import build_flow
import optuna
from ..utils import set_random_seed

def func_search(trial):
    return {
        "lr": trial.suggest_float("lr", 0.3, 0.7),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 0.0001]),
        # "hidden_dim": trial.suggest_categorical("hidden_dim", [200, 400]),
        # "ent_dim": trial.suggest_categorical("ent_dim", [200, 400]),
        # "rel_dim": trial.suggest_categorical("rel_dim", [200, 400]),
        "neg_size": trial.suggest_int("neg_size", 70, 130),
        "margin": trial.suggest_float("margin", 150, 250),
        # "valid_percent": trial.suggest_categorical("valid_percent", [0.4]),
        # "test_percent": trial.suggest_categorical("test_percent", [1]),
    }


def hpo_experiment(args, trainerflow, **kwargs):
    logger = args.logger
    tool = AutoML(args, trainerflow, n_trials=15, func_search=func_search, logger=logger)
    result = tool.run()
    logger.info("[Hyper-parameter optimization] Final results:{}".format(result))
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
        self.logger = kwargs['logger']
        self.n_trials = n_trials
        self.best_score = None
        self.best_params = None
        self.default_params = kwargs

    def _objective(self, trials):
        args = self.args
        cur_params = self.func_search(trials)
        for key, value in cur_params.items():
            args.__setattr__(key, value)
        # Set random seed each time, or the initialization of the weight will be different.
        set_random_seed(args.seed)
        flow = build_flow(args, self.trainerflow)
        result = flow.train()['metric']['test']
        if isinstance(result, tuple):
            score = (result[0] + result[1]) / 2
        elif isinstance(result, dict):
            score = -result['MR']
            # for _, v in result.items():
            #     score += v
            # score /= len(result)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_params = cur_params
        return score

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        self.logger.info("[Hyper-parameter optimization] Best parameter: {}".format(self.best_params))
        return self.best_score
