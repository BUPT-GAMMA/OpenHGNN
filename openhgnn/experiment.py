import os.path
from .config import Config
from .utils import set_random_seed, set_best_config, Logger
from .trainerflow import build_flow
from .auto import hpo_experiment
import torch
import warnings

__all__ = ['Experiment']


class Experiment(object):
    r"""Experiment.

    Parameters
    ----------
    model : str or nn.Module
        Name of the model or a hetergenous gnn model provided by the user.
    dataset : str or DGLDataset
        Name of the model or a DGLDataset provided by the user.
    use_best_config: bool
        Whether to load the best config of specific models and datasets. Default: False
    load_from_pretrained : bool
        Whether to load the model from the checkpoint. Default: False
    hpo_search_space :
        Search space for hyperparameters.
    hpo_trials : int
        Number of trials for hyperparameter search.
    Examples
    --------
    >>> experiment = Experiment(model='RGCN', dataset='imdb4GTN', task='node_classification', gpu=-1)
    >>> experiment.run()
    """

    default_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    specific_trainerflow = {
        'HetGNN': 'hetgnntrainer',
        'HGNN_AC': 'node_classification_ac',
        'NSHE': 'nshetrainer',
        'HeCo': 'HeCo_trainer',
        'DMGI': 'DMGI_trainer',
        'KGCN': 'kgcntrainer',
        'Metapath2vec': 'mp2vec_trainer',
        'HERec': 'herec_trainer',
        'SLiCE': 'slicetrainer',
        'HeGAN': 'HeGAN_trainer',
        'HDE': 'hde_trainer',
        'SIAN': 'SIAN_trainer',
        'GATNE-T': 'GATNE_trainer',
        'TransE': 'TransX_trainer',
        'TransH': 'TransX_trainer',
        'TransR': 'TransX_trainer',
        'TransD': 'TransX_trainer',
        'RedGNN': 'RedGNN_trainer',
        'RedGNNT': 'RedGNNT_trainer',
        'GIE': 'TransX_trainer',
        'HAN': {
            'node_classification': 'han_nc_trainer',
            'link_prediction': 'han_lp_trainer',
        },
        'RoHe': 'RoHe_trainer',
        'Mg2vec': 'mg2vec_trainer',
        'DHNE': 'DHNE_trainer',
        'DiffMG': 'DiffMG_trainer',
        'MeiREC': 'MeiREC_trainer',
        'KGAT': 'KGAT_trainer',
        'SHGP': 'SHGP_trainer',
        'HGCL': 'hgcltrainer',
        'lightGCN': 'lightGCN_trainer',
        'HMPNN':'KTN_trainer',
        'SeHGNN': 'SeHGNN_trainer',
        'Grail': 'Grail_trainer',
        'ComPILE': 'ComPILE_trainer',
        'AdapropT':'AdapropT_trainer',
        'AdapropI':'AdapropI_trainer',
        'LTE':'LTE_trainer',
        'SACN':'SACN_trainer',
        'ExpressGNN': 'ExpressGNN_trainer',
        'NBF':'NBF_trainer',
        'Ingram': 'Ingram_trainer',
        'DisenKGAT': 'DisenKGAT_trainer',
######################          add trainer_flow  here。 【model name】：【register name】
        'BPHGNN':'BPHGNN_trainer',        
        'HGPrompt':'HGPrompt_trainer',
        'HGMAE':'HGMAE_trainer',
        'HGA':'hga_trainer',
        'RHINE':'rhine_trainer',
        'FedHGNN':'FED_REC_trainer',
        "MHGCN":{
            "node_classification":"MHGCN_NC_Trainer",
            "link_prediction":"MHGCN_LP_Trainer"
        },
##########################


    }
    
    immutable_params = ['model', 'dataset', 'task']

    def __init__(self, model, dataset, task,
                 gpu: int=-1,
                 use_best_config: bool = False,
                 load_from_pretrained: bool = False,
                 hpo_search_space=None,
                 hpo_trials: int = 100,
                 output_dir: str = "./openhgnn/output",
                 conf_path: str = default_conf_path,
                 use_database:bool = False,
                 **kwargs):
        self.config = Config(file_path=conf_path, model=model, dataset=dataset, task=task, gpu=gpu)
        self.config.model = model
        self.config.dataset = dataset
        self.config.task = task
        self.config.use_distributed = kwargs['use_distributed']
        kwargs.pop('use_distributed')
        if self.config.use_distributed:
            self.config.gpu = [i for i in range(torch.cuda.device_count())]
        else:
            self.config.gpu = gpu
        self.config.use_best_config = use_best_config
        self.config.use_database = use_database
        # self.config.use_hpo = use_hpo
        self.config.load_from_pretrained = load_from_pretrained
        self.config.output_dir = os.path.join(output_dir, self.config.model_name)
        # self.config.seed = seed
        self.config.hpo_search_space = hpo_search_space
        self.config.hpo_trials = hpo_trials

        if not getattr(self.config, 'seed', False):
            self.config.seed = 0
        if use_best_config:
            self.config = set_best_config(self.config)
        self.set_params(**kwargs)
        print(self)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            assert key not in self.immutable_params
            self.config.__setattr__(key, value)

    def distributed_run(self, proc_id):
        num_gpus=len(self.config.gpu)
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:12391",
            world_size=num_gpus,
            rank=proc_id,
        )
        self.config.gpu = self.config.gpu[proc_id]

        if self.config.gpu < 0:
            self.config.device = torch.device('cpu')
        elif self.config.gpu >= 0:
            if not torch.cuda.is_available( ):
                self.config.device = torch.device('cpu')
                warnings.warn("cuda is unavailable, the program will use cpu instead. please set 'gpu' to -1.")
            else:
                self.config.device = torch.device('cuda', int(self.config.gpu))

        # test add profiler
        if self.config.mini_batch_flag == False:
            if hasattr(self.config, 'max_epoch'):
                self.config.max_epoch = self.config.max_epoch // num_gpus

        if hasattr(self.config, 'line_profiler_func'):
            from line_profiler import LineProfiler
            prof = LineProfiler
            for func in self.config.line_profiler_func:
                prof = prof(func)
            prof.enable_by_count()

        self.config.logger = Logger(self.config)
        set_random_seed(self.config.seed)
        trainerflow = self.specific_trainerflow.get(self.config.model, self.config.task)
        if type(trainerflow) is not str:
            trainerflow = trainerflow.get(self.config.task)
        if self.config.hpo_search_space is not None:
            # hyper-parameter search
            hpo_experiment(self.config, trainerflow)
        else:
            flow = build_flow(self.config, trainerflow)
            result = flow.train()
            if hasattr(self.config, 'line_profiler_func'):
                prof.print_stats()
            return

    def run(self):
        """ run the experiment """

        # 'line_profiler_func' is for internal use in profiling code execution time, here is an example to use it:
        # from openhgnn import Experiment
        # from openhgnn.trainerflow import NodeClassification
        # Experiment(model='RGCN', dataset='acm4GTN', task='node_classification', gpu=-1, max_epoch=1, line_profiler_func=[NodeClassification.train, ]).run()

        if self.config.use_distributed:
            num_gpus = len(self.config.gpu)
            import torch.multiprocessing as mp
            mp.spawn(self.distributed_run, nprocs=num_gpus)
            return

        if hasattr(self.config, 'line_profiler_func'):
            from line_profiler import LineProfiler
            prof = LineProfiler
            for func in self.config.line_profiler_func:
                prof = prof(func)
            prof.enable_by_count()

        self.config.logger = Logger(self.config)
        set_random_seed(self.config.seed)
        trainerflow = self.specific_trainerflow.get(self.config.model, self.config.task)
        if type(trainerflow) is not str:
            trainerflow = trainerflow.get(self.config.task)
        if self.config.hpo_search_space is not None:
            # hyper-parameter search
            hpo_experiment(self.config, trainerflow)
        else:
            flow = build_flow(self.config, trainerflow)
            result = flow.train()
            if hasattr(self.config, 'line_profiler_func'):
                prof.print_stats()
            return result

    def __repr__(self):
        basic_info = '------------------------------------------------------------------------------\n' \
                     ' Basic setup of this experiment: \n' \
                     '     model: {}    \n' \
                     '     dataset: {}   \n' \
                     '     task: {}. \n' \
                     ' This experiment has following parameters. You can use set_params to edit them.\n' \
                     ' Use print(experiment) to print this information again.\n' \
                     '------------------------------------------------------------------------------\n'. \
            format(self.config.model_name, self.config.dataset_name, self.config.task)
        params_info = ''
        for attr in dir(self.config):
            if '__' not in attr and attr not in self.immutable_params:
                params_info += '{}: {}\n'.format(attr, getattr(self.config, attr))
        return basic_info + params_info
