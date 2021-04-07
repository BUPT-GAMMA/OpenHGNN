import configparser
import os
import numpy as np
import torch as th


class Config(object):
    def __init__(self, file_path, model, dataset, gpu):
        conf = configparser.ConfigParser()
        data_path = os.getcwd()
        if gpu == -1:
            self.device = th.device('cpu')
        elif gpu >= 0:
            if th.cuda.is_available():
                self.device = th.device('cuda', int(gpu))
            else:
                print("cuda is not available, please set 'gpu' -1")

        try:
            conf.read(file_path)
        except:
            print("failed!")
        # training dataset path
        self.model = model
        self.dataset = dataset
        self.path = {'output_modelfold': './output/models/',
                     'input_fold': './dataset/'+self.dataset+'/',
                     'temp_fold': './output/temp/'+self.model+'/'}

        if model == "NSHE":
            self.dim_size = {}
            self.dim_size['emd'] = conf.getint("NSHE", "emd_dim")
            self.dim_size['context'] = conf.getint("NSHE", "context_dim")
            self.dim_size['project'] = conf.getint("NSHE", "project_dim")

            self.lr = conf.getfloat("NSHE", "learning_rate")
            self.beta = conf.getfloat("NSHE", "beta")
            self.seed = conf.getint("NSHE", "seed")
            np.random.seed(self.seed)
            self.max_epoch = conf.getint("NSHE", "max_epoch")
            self.num_e_neg = conf.getint("NSHE", "num_e_neg")
            self.num_ns_neg = conf.getint("NSHE", "num_ns_neg")
            self.norm_emd_flag = conf.get("NSHE", "norm_emd_flag")
        elif model == "GTN":
            self.lr = conf.getfloat("GTN", "learning_rate")
            self.weight_decay = conf.getfloat("GTN", "weight_decay")
            self.seed = conf.getint("GTN", "seed")
            # np.random.seed(self.seed)

            self.emd_size = conf.getint("GTN", "emd_dim")
            self.num_channels = conf.getint("GTN", "num_channels")
            self.num_layers = conf.getint("GTN", "num_layers")
            self.max_epoch = conf.getint("GTN", "max_epoch")

            self.norm_emd_flag = conf.get("GTN", "norm_emd_flag")
            self.adaptive_lr = conf.get("GTN", "adaptive_lr_flag")
            self.sparse_flag = conf.get("GTN", "sparse")
        elif model == "RSHN":
            self.lr = conf.getfloat("RSHN", "learning_rate")
            self.weight_decay = conf.getfloat("RSHN", "weight_decay")
            self.dropout = conf.getfloat("RSHN", "dropout")

            self.seed = conf.getint("RSHN", "seed")
            self.dim = conf.getint("RSHN", "dim")
            self.max_epoch = conf.getint("RSHN", "max_epoch")
            self.rw_len = conf.getint("RSHN", "rw_len")
            self.batch_size = conf.getint("RSHN", "batch_size")
            self.num_node_layer = conf.getint("RSHN", "num_node_layer")
            self.num_edge_layer = conf.getint("RSHN", "num_edge_layer")

            self.validation = conf.getboolean("RSHN", "validation")
        elif model == 'RGCN':
            self.lr = conf.getfloat("RGCN", "learning_rate")
            self.dropout = conf.getfloat("RGCN", "dropout")

            self.n_hidden = conf.getint("RGCN", "n_hidden")
            self.n_bases = conf.getint("RGCN", "n_bases")
            self.n_layers = conf.getint("RGCN", "n_layers")
            self.max_epoch = conf.getint("RGCN", "max_epoch")
            self.weight_decay = conf.getint("RGCN", "weight_decay")

            self.validation = conf.getboolean("RGCN", "validation")
            self.use_self_loop = conf.getboolean("RGCN", "use_self_loop")
        elif model == 'CompGCN':
            self.lr = conf.getfloat("CompGCN", "learning_rate")

            self.weight_decay = conf.getfloat("CompGCN", "weight_decay")
            self.dropout = conf.getfloat("CompGCN", "dropout")

            self.n_hidden = conf.getint("CompGCN", "n_hidden")
            self.n_layers = conf.getint("CompGCN", "n_layers")
            self.max_epoch = conf.getint("CompGCN", "max_epoch")
            self.seed = conf.getint("CompGCN", "seed")
            self.patience = conf.getint("CompGCN", "patience")

            self.comp_fn = conf.get("CompGCN", "comp_fn")
            self.task = 'node_classification'
            self.validation = conf.getboolean("CompGCN", "validation")
            pass
        elif model == 'HetGNN':
            self.lr = conf.getfloat("HetGNN", "learning_rate")

            self.weight_decay = conf.getfloat("HetGNN", "weight_decay")
            #self.dropout = conf.getfloat("CompGCN", "dropout")
            self.max_epoch = conf.getint("HetGNN", "max_epoch")
            self.dim = conf.getint("HetGNN", "dim")
            self.batch_size = conf.getint("HetGNN", "batch_size")
            self.window_size = conf.getint("HetGNN", "window_size")
            self.num_workers = conf.getint("HetGNN", "num_workers")
            self.batches_per_epoch = conf.getint("HetGNN", "batches_per_epoch")

            self.rw_length = conf.getint("HetGNN", "rw_length")
            self.rw_walks = conf.getint("HetGNN", "rw_walks")
            self.rwr_prob = conf.getfloat("HetGNN", "rwr_prob")

            pass
