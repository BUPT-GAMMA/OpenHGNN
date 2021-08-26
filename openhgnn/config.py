import configparser
import os
import numpy as np
import torch as th


class Config(object):
    def __init__(self, file_path, model, dataset, task, gpu):
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
        self.task = task
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
            self.weight_decay = conf.getfloat("NSHE", "weight_decay")
            self.beta = conf.getfloat("NSHE", "beta")
            self.seed = conf.getint("NSHE", "seed")
            np.random.seed(self.seed)
            self.max_epoch = conf.getint("NSHE", "max_epoch")
            self.patience = conf.getint("NSHE", "patience")
            self.num_e_neg = conf.getint("NSHE", "num_e_neg")
            self.num_ns_neg = conf.getint("NSHE", "num_ns_neg")
            self.norm_emd_flag = conf.get("NSHE", "norm_emd_flag")
            self.mini_batch_flag = conf.getboolean("NSHE", "mini_batch_flag")

        elif model == "GTN":
            self.lr = conf.getfloat("GTN", "learning_rate")
            self.weight_decay = conf.getfloat("GTN", "weight_decay")
            self.seed = conf.getint("GTN", "seed")
            # np.random.seed(self.seed)

            self.hidden_dim = conf.getint("GTN", "hidden_dim")
            self.out_dim = conf.getint("GTN", "out_dim")
            self.num_channels = conf.getint("GTN", "num_channels")
            self.num_layers = conf.getint("GTN", "num_layers")
            self.max_epoch = conf.getint("GTN", "max_epoch")
            self.patience = conf.getint("GTN", "patience")

            self.identity = conf.getboolean("GTN", "identity")
            self.norm_emd_flag = conf.getboolean("GTN", "norm_emd_flag")
            self.adaptive_lr_flag = conf.getboolean("GTN", "adaptive_lr_flag")
            self.mini_batch_flag = conf.getboolean("GTN", "mini_batch_flag")
        elif model == "RSHN":
            self.lr = conf.getfloat("RSHN", "learning_rate")
            self.weight_decay = conf.getfloat("RSHN", "weight_decay")
            self.dropout = conf.getfloat("RSHN", "dropout")

            self.seed = conf.getint("RSHN", "seed")
            self.hidden_dim = conf.getint("RSHN", "hidden_dim")
            self.max_epoch = conf.getint("RSHN", "max_epoch")
            self.rw_len = conf.getint("RSHN", "rw_len")
            self.batch_size = conf.getint("RSHN", "batch_size")
            self.num_node_layer = conf.getint("RSHN", "num_node_layer")
            self.num_edge_layer = conf.getint("RSHN", "num_edge_layer")
            self.patience = conf.getint("RSHN", "patience")
            self.validation = conf.getboolean("RSHN", "validation")
            self.mini_batch_flag = conf.getboolean("RSHN", "mini_batch_flag")

        elif model == 'RGCN':
            self.lr = conf.getfloat("RGCN", "learning_rate")
            self.dropout = conf.getfloat("RGCN", "dropout")

            self.hidden_dim = conf.getint("RGCN", "hidden_dim")
            self.out_dim = conf.getint("RGCN", "out_dim")

            self.n_bases = conf.getint("RGCN", "n_bases")
            self.n_layers = conf.getint("RGCN", "n_layers")
            self.max_epoch = conf.getint("RGCN", "max_epoch")
            self.weight_decay = conf.getfloat("RGCN", "weight_decay")
            self.seed = conf.getint("RGCN", "seed")
            self.fanout = conf.getint("RGCN", "fanout")
            self.patience = conf.getint("RGCN", "patience")
            self.batch_size = conf.getint("RGCN", "batch_size")
            self.validation = conf.getboolean("RGCN", "validation")
            self.mini_batch_flag = conf.getboolean("RGCN", "mini_batch_flag")
            self.use_self_loop = conf.getboolean("RGCN", "use_self_loop")

        elif model == 'CompGCN':
            self.lr = conf.getfloat("CompGCN", "learning_rate")

            self.weight_decay = conf.getfloat("CompGCN", "weight_decay")
            self.dropout = conf.getfloat("CompGCN", "dropout")

            self.h_dim = conf.getint("CompGCN", "h_dim")
            self.out_dim = conf.getint("CompGCN", "out_dim")
            self.n_layers = conf.getint("CompGCN", "n_layers")
            self.max_epoch = conf.getint("CompGCN", "max_epoch")
            self.seed = conf.getint("CompGCN", "seed")
            self.patience = conf.getint("CompGCN", "patience")

            self.comp_fn = conf.get("CompGCN", "comp_fn")
            self.mini_batch_flag = conf.getboolean("CompGCN", "mini_batch_flag")
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
            self.seed = conf.getint("HetGNN", "seed")
            self.patience = conf.getint("HetGNN", "patience")
            self.rw_length = conf.getint("HetGNN", "rw_length")
            self.rw_walks = conf.getint("HetGNN", "rw_walks")
            self.rwr_prob = conf.getfloat("HetGNN", "rwr_prob")
            self.mini_batch_flag = conf.getboolean("HetGNN", "mini_batch_flag")
            pass
        elif model == 'Metapath2vec':
            self.lr = conf.getfloat("Metapath2vec", "learning_rate")
            self.weight_decay = conf.getfloat("Metapath2vec", "weight_decay")

            #self.dropout = conf.getfloat("CompGCN", "dropout")
            self.max_epoch = conf.getint("Metapath2vec", "max_epoch")
            self.dim = conf.getint("Metapath2vec", "dim")
            self.batch_size = conf.getint("Metapath2vec", "batch_size")
            self.window_size = conf.getint("Metapath2vec", "window_size")
            self.num_workers = conf.getint("Metapath2vec", "num_workers")
            self.batches_per_epoch = conf.getint("Metapath2vec", "batches_per_epoch")
            self.seed = conf.getint("Metapath2vec", "seed")
            self.patience = conf.getint("Metapath2vec", "patience")
            self.rw_length = conf.getint("Metapath2vec", "rw_length")
            self.rw_walks = conf.getint("Metapath2vec", "rw_walks")
            self.rwr_prob = conf.getfloat("Metapath2vec", "rwr_prob")
            self.mini_batch_flag = conf.getboolean("Metapath2vec", "mini_batch_flag")

        elif model == 'HAN':
            self.lr = conf.getfloat("HAN", "learning_rate")
            self.weight_decay = conf.getfloat("HAN", "weight_decay")
            self.seed = conf.getint("HAN", "seed")
            self.dropout = conf.getfloat("HAN", "dropout")

            self.hidden_dim = conf.getint('HAN', 'hidden_dim')
            self.out_dim = conf.getint('HAN', 'out_dim')
            num_heads = conf.get('HAN', 'num_heads').split('-')
            self.num_heads = [int(i) for i in num_heads]
            self.patience = conf.getint('HAN', 'patience')
            self.max_epoch = conf.getint('HAN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HAN", "mini_batch_flag")

        elif model == 'MAGNN':
            self.lr = conf.getfloat("MAGNN", "learning_rate")
            self.weight_decay = conf.getfloat("MAGNN", "weight_decay")
            self.seed = conf.getint("MAGNN", "seed")
            self.dropout = conf.getfloat("MAGNN", "dropout")

            self.inter_attn_feats = conf.getint("MAGNN", "inter_attn_feats")
            self.hidden_dim = conf.getint('MAGNN', 'hidden_dim')
            self.out_dim = conf.getint('MAGNN', 'out_dim')
            self.num_heads = conf.getint('MAGNN', 'num_heads')
            self.num_layers = conf.getint("MAGNN", "num_layers")

            self.patience = conf.getint('MAGNN', 'patience')
            self.max_epoch = conf.getint('MAGNN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("MAGNN", "mini_batch_flag")
            self.encoder_type = conf.get('MAGNN', 'encoder_type')
        elif model == 'MAGNN_AC':
            self.lr = conf.getfloat("MAGNN_AC", "learning_rate")
            self.weight_decay = conf.getfloat("MAGNN_AC", "weight_decay")
            self.seed = conf.getint("MAGNN_AC", "seed")
            self.dropout = conf.getfloat("MAGNN_AC", "dropout")
            
            self.feats_drop_rate = conf.getfloat("MAGNN_AC", "feats_drop_rate")
            self.attn_vec_dim = conf.getint("MAGNN_AC", "attn_vec_dim")
            self.feats_opt = conf.get("MAGNN_AC", "feats_opt")
            self.loss_lambda = conf.getfloat("MAGNN_AC", "loss_lambda")
            self.src_node_type = conf.getint("MAGNN_AC", "src_node_type")

            self.inter_attn_feats = conf.getint("MAGNN_AC", "inter_attn_feats")
            self.hidden_dim = conf.getint('MAGNN_AC', 'hidden_dim')
            self.out_dim = conf.getint('MAGNN_AC', 'out_dim')
            self.num_heads = conf.getint('MAGNN_AC', 'num_heads')
            self.num_layers = conf.getint("MAGNN_AC", "num_layers")

            self.patience = conf.getint('MAGNN_AC', 'patience')
            self.max_epoch = conf.getint('MAGNN_AC', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("MAGNN_AC", "mini_batch_flag")
            self.encoder_type = conf.get('MAGNN_AC', 'encoder_type')
        elif model == 'HGT':
            self.lr = conf.getfloat("HGT", "learning_rate")
            self.weight_decay = conf.getfloat("HGT", "weight_decay")
            self.seed = conf.getint("HGT", "seed")
            self.dropout = conf.getfloat("HGT", "dropout")

            self.batch_size = conf.getint("HGT", "batch_size")
            self.hidden_dim = conf.getint('HGT', 'hidden_dim')
            self.out_dim = conf.getint('HGT', 'out_dim')
            self.num_heads = conf.getint('HGT', 'num_heads')
            self.patience = conf.getint('HGT', 'patience')
            self.max_epoch = conf.getint('HGT', 'max_epoch')
            self.num_workers = conf.getint("HGT", "num_workers")
            self.mini_batch_flag = conf.getboolean("HGT", "mini_batch_flag")
            self.n_layers = conf.getint("HGT", "n_layers")
            self.num_heads = conf.getint("HGT", "num_heads")

    def __repr__(self):
        return 'Model:' + self.model + '\nTask:' + self.task + '\nDataset:' + self.dataset
