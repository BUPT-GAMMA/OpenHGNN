from ..trainerflow import build_flow
import optuna
from ..utils import set_random_seed


def hpo_experiment(args, trainerflow, **kwargs):
    logger = args.logger
    tool = AutoML(args, trainerflow, n_trials=args.hpo_trials, search_space=args.hpo_search_space, logger=logger)
    result = tool.run()
    logger.info("[Hyper-parameter optimization] Final results:{}".format(result))
    return result


class AutoML(object):
    """
    Args:
        search_space: function to obtain hyper-parameters to search
    """

    def __init__(self, args, trainerflow, n_trials=3, **kwargs):
        self.args = args
        self.trainerflow = trainerflow
        # self.seed = kwargs.pop("seed") if "seed" in kwargs else [1]
        assert "search_space" in kwargs
        self.search_space = kwargs["search_space"]
        self.metric = kwargs["metric"] if "metric" in kwargs else None
        self.logger = kwargs['logger']
        self.n_trials = n_trials
        self.best_score = None
        self.best_params = None
        self.default_params = kwargs

    def _objective(self, trials):
        args = self.args
        cur_params = self.search_space(trials)
        for key, value in cur_params.items():
            args.__setattr__(key, value)
        # Set random seed each time, or the initialization of the weight will be different.
        set_random_seed(args.seed)
        flow = build_flow(args, self.trainerflow)
        result = flow.train()['metric']['test']
        if isinstance(result, tuple):
            score = (result[0] + result[1]) / 2
        elif isinstance(result, dict):
            score = 0
            for _, v in result.items():
                score += v
            score /= len(result)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_params = cur_params
        return score

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        self.logger.info("[Hyper-parameter optimization] Best parameter: {}".format(self.best_params))
        return self.best_score
