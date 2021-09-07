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
        self.seed = 0
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

        elif model == "MHNF":
            self.lr = conf.getfloat("MHNF", "learning_rate")
            self.weight_decay = conf.getfloat("MHNF", "weight_decay")
            self.seed = conf.getint("MHNF", "seed")
            # np.random.seed(self.seed)

            self.hidden_dim = conf.getint("MHNF", "hidden_dim")
            self.out_dim = conf.getint("MHNF", "out_dim")
            self.num_channels = conf.getint("MHNF", "num_channels")
            self.num_layers = conf.getint("MHNF", "num_layers")
            self.max_epoch = conf.getint("MHNF", "max_epoch")
            self.patience = conf.getint("MHNF", "patience")

            self.identity = conf.getboolean("MHNF", "identity")
            self.norm_emd_flag = conf.getboolean("MHNF", "norm_emd_flag")
            self.adaptive_lr_flag = conf.getboolean("MHNF", "adaptive_lr_flag")
            self.mini_batch_flag = conf.getboolean("MHNF", "mini_batch_flag")

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

            self.in_dim = conf.getint("RGCN", "in_dim")
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

            self.in_dim = conf.getint("CompGCN", "in_dim")
            self.hidden_dim = conf.getint("CompGCN", "hidden_dim")
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

        elif model == 'NARS':
            self.lr = conf.getfloat("NARS", "learning_rate")
            self.weight_decay = conf.getfloat("NARS", "weight_decay")
            self.seed = conf.getint("NARS", "seed")
            self.dropout = conf.getfloat("NARS", "dropout")
            self.patience = conf.getint('HAN', 'patience')
            self.hidden_dim = conf.getint('NARS', 'hidden_dim')
            self.out_dim = conf.getint('NARS', 'out_dim')
            num_heads = conf.get('NARS', 'num_heads').split('-')
            self.num_heads = [int(i) for i in num_heads]

            self.max_epoch = conf.getint('NARS', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("NARS", "mini_batch_flag")
            self.R = conf.getint('NARS', 'R')
            self.cpu_preprocess = conf.getboolean("NARS", "cpu_preprocess")
            self.input_dropout = conf.getboolean("NARS", "input_dropout")

            self.ff_layer = conf.getint('NARS', 'ff_layer')


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
        
        elif model == 'RHGNN':
            self.lr = conf.getfloat("RHGNN", "learning_rate")
            self.num_heads = conf.getint("RHGNN", "num_heads")
            self.hidden_dim = conf.getint("RHGNN", "hidden_dim")
            self.relation_hidden_units = conf.getint("RHGNN", "relation_hidden_units")
            self.drop_out = conf.getfloat("RHGNN", "drop_out")
            self.n_layers = conf.getint("RHGNN", "n_layers")
            self.residual = conf.getboolean("RHGNN", "residual")
            self.batch_size = conf.getint("RHGNN", "batch_size")
            self.node_neighbors_min_num = conf.getint("RHGNN", "node_neighbors_min_num")
            #self.optimizer = conf.get
            self.weight_decay = conf.getfloat("RHGNN", "weight_decay")
            self.max_epoch = conf.getint("RHGNN", "max_epoch")
            self.patience = conf.getint("RHGNN", "patience")
            self.mini_batch_flag = conf.getboolean("RHGNN", "mini_batch_flag")
            self.negative_slope = conf.getfloat("RHGNN", "negative_slope")
            self.norm = conf.getboolean("RHGNN", "norm")
            self.dropout = conf.getfloat("RHGNN", "dropout")
            self.n_heads = conf.getint("RHGNN", "n_heads")
            self.category = conf.get("RHGNN", "category")
            self.out_dim = conf.getint("RHGNN", "out_dim")
        
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

        elif model == 'DMGI':
            self.lr = conf.getfloat("DMGI", "learning_rate")
            self.l2_coef = conf.getfloat("DMGI", "l2_coef")
            self.sc = conf.getint("DMGI", "sc")
            self.seed = conf.getint("DMGI", "seed")
            self.sup_coef = conf.getfloat("DMGI",'sup_coef')
            self.reg_coef = conf.getfloat("DMGI", "reg_coef")
            self.dropout = conf.getfloat("DMGI", "dropout")
            self.hid_unit = conf.getint('DMGI', 'hid_unit')
            self.num_heads = conf.getint('DMGI', 'num_heads')
            self.patience = conf.getint('DMGI', 'patience')
            self.max_epoch = conf.getint('DMGI', 'max_epoch')

            self.isSemi = conf.getboolean("DMGI", "isSemi")
            self.isBias = conf.getboolean("DMGI", "isBias")
            self.isAttn = conf.getboolean("DMGI", "isAttn")
        elif model == 'HPN':
            self.lr = conf.getfloat("HPN", "learning_rate")
            self.weight_decay = conf.getfloat("HPN", "weight_decay")
            self.seed = conf.getint("HPN", "seed")
            self.dropout = conf.getfloat("HPN", "dropout")
            self.out_embedsize = conf.getint("HPN", "out_embedsize")
            self.hidden_dim = conf.getint('HPN', 'hidden_dim')
            self.k_layer = conf.getint("HPN", "k_layer")
            self.alpha = conf.getfloat("HPN", "alpha")
            self.edge_drop = conf.getfloat("HPN", "edge_drop")
            self.patience = conf.getint('HPN', 'patience')
            self.max_epoch = conf.getint('HPN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HPN", "mini_batch_flag")

    def __repr__(self):
        return 'Model:' + self.model + '\nTask:' + self.task + '\nDataset:' + self.dataset
