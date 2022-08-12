import configparser
import numpy as np
import torch as th
from .utils.activation import act_dict


class Config(object):
    def __init__(self, file_path, model, dataset, task, gpu):
        conf = configparser.ConfigParser()
        if gpu == -1:
            self.device = th.device('cpu')
        elif gpu >= 0:
            if th.cuda.is_available():
                self.device = th.device('cuda', int(gpu))
            else:
                raise ValueError("cuda is not available, please set 'gpu' -1")

        try:
            conf.read(file_path)
        except:
            print("failed!")
        # training dataset path
        self.seed = 0
        self.patience = 1
        self.max_epoch = 1
        self.task = task
        self.model = model
        self.dataset = dataset
        if isinstance(dataset, str):
            self.dataset_name = dataset
        else:
            self.dataset_name = type(self.dataset).__name__
        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = type(self.model).__name__
        self.optimizer = 'Adam'
        # custom model
        if isinstance(model, th.nn.Module):
            self.lr = conf.getfloat("General", "learning_rate")
            self.dropout = conf.getfloat("General", "dropout")
            self.max_epoch = conf.getint("General", "max_epoch")
            self.weight_decay = conf.getfloat("General", "weight_decay")
            self.hidden_dim = conf.getint("General", "hidden_dim")
            self.seed = conf.getint("General", "seed")
            self.patience = conf.getint("General", "patience")
            self.mini_batch_flag = conf.getboolean("General", "mini_batch_flag")
        elif self.model_name == "NSHE":
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

        elif self.model_name in ["GTN", "fastGTN"]:
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

        elif self.model_name == "MHNF":
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

        elif self.model_name == "RSHN":
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

        elif self.model_name == 'RGCN':
            self.lr = conf.getfloat("RGCN", "learning_rate")
            self.dropout = conf.getfloat("RGCN", "dropout")

            self.in_dim = conf.getint("RGCN", "in_dim")
            self.hidden_dim = conf.getint("RGCN", "hidden_dim")

            self.n_bases = conf.getint("RGCN", "n_bases")
            self.num_layers = conf.getint("RGCN", "num_layers")
            self.max_epoch = conf.getint("RGCN", "max_epoch")
            self.weight_decay = conf.getfloat("RGCN", "weight_decay")
            self.seed = conf.getint("RGCN", "seed")
            self.fanout = conf.getint("RGCN", "fanout")
            self.patience = conf.getint("RGCN", "patience")
            self.batch_size = conf.getint("RGCN", "batch_size")
            self.validation = conf.getboolean("RGCN", "validation")
            self.mini_batch_flag = conf.getboolean("RGCN", "mini_batch_flag")
            self.use_self_loop = conf.getboolean("RGCN", "use_self_loop")

        elif self.model_name == 'CompGCN':
            self.lr = conf.getfloat("CompGCN", "learning_rate")

            self.weight_decay = conf.getfloat("CompGCN", "weight_decay")
            self.dropout = conf.getfloat("CompGCN", "dropout")

            self.in_dim = conf.getint("CompGCN", "in_dim")
            self.hidden_dim = conf.getint("CompGCN", "hidden_dim")
            self.out_dim = conf.getint("CompGCN", "out_dim")
            self.num_layers = conf.getint("CompGCN", "num_layers")
            self.max_epoch = conf.getint("CompGCN", "max_epoch")
            self.seed = conf.getint("CompGCN", "seed")
            self.patience = conf.getint("CompGCN", "patience")

            self.comp_fn = conf.get("CompGCN", "comp_fn")
            self.mini_batch_flag = conf.getboolean("CompGCN", "mini_batch_flag")
            self.validation = conf.getboolean("CompGCN", "validation")
            pass
        elif self.model_name == 'HetGNN':
            self.lr = conf.getfloat("HetGNN", "learning_rate")
            self.weight_decay = conf.getfloat("HetGNN", "weight_decay")

            # self.dropout = conf.getfloat("CompGCN", "dropout")
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
        elif self.model_name == 'Metapath2vec':
            self.lr = conf.getfloat("Metapath2vec", "learning_rate")
            self.max_epoch = conf.getint("Metapath2vec", "max_epoch")
            self.dim = conf.getint("Metapath2vec", "dim")
            self.batch_size = conf.getint("Metapath2vec", "batch_size")
            self.window_size = conf.getint("Metapath2vec", "window_size")
            self.num_workers = conf.getint("Metapath2vec", "num_workers")
            self.neg_size = conf.getint("Metapath2vec", "neg_size")
            self.rw_length = conf.getint("Metapath2vec", "rw_length")
            self.rw_walks = conf.getint("Metapath2vec", "rw_walks")
            self.meta_path_key = conf.get("Metapath2vec", "meta_path_key")

        elif self.model_name == 'HERec':
            self.lr = conf.getfloat("HERec", "learning_rate")
            self.max_epoch = conf.getint("HERec", "max_epoch")
            self.dim = conf.getint("HERec", "dim")
            self.batch_size = conf.getint("HERec", "batch_size")
            self.window_size = conf.getint("HERec", "window_size")
            self.num_workers = conf.getint("HERec", "num_workers")
            self.neg_size = conf.getint("HERec", "neg_size")
            self.rw_length = conf.getint("HERec", "rw_length")
            self.rw_walks = conf.getint("HERec", "rw_walks")
            self.meta_path_key = conf.get("HERec", "meta_path_key")

        elif self.model_name == 'HAN':
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

        elif self.model_name == 'NARS':
            self.lr = conf.getfloat("NARS", "learning_rate")
            self.weight_decay = conf.getfloat("NARS", "weight_decay")
            self.seed = conf.getint("NARS", "seed")
            self.dropout = conf.getfloat("NARS", "dropout")
            self.patience = conf.getint('HAN', 'patience')
            self.hidden_dim = conf.getint('NARS', 'hidden_dim')
            self.out_dim = conf.getint('NARS', 'out_dim')
            num_heads = conf.get('NARS', 'num_heads').split('-')
            self.num_heads = [int(i) for i in num_heads]
            self.num_hops = conf.getint('NARS', 'num_hops')

            self.max_epoch = conf.getint('NARS', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("NARS", "mini_batch_flag")
            self.R = conf.getint('NARS', 'R')
            self.cpu_preprocess = conf.getboolean("NARS", "cpu_preprocess")
            self.input_dropout = conf.getboolean("NARS", "input_dropout")

            self.ff_layer = conf.getint('NARS', 'ff_layer')

        elif self.model_name == 'MAGNN':
            self.lr = conf.getfloat("MAGNN", "learning_rate")
            self.weight_decay = conf.getfloat("MAGNN", "weight_decay")
            self.seed = conf.getint("MAGNN", "seed")
            self.dropout = conf.getfloat("MAGNN", "dropout")

            self.inter_attn_feats = conf.getint("MAGNN", "inter_attn_feats")
            self.h_dim = conf.getint('MAGNN', 'h_dim')
            self.out_dim = conf.getint('MAGNN', 'out_dim')
            self.num_heads = conf.getint('MAGNN', 'num_heads')
            self.num_layers = conf.getint("MAGNN", "num_layers")

            self.patience = conf.getint('MAGNN', 'patience')
            self.max_epoch = conf.getint('MAGNN', 'max_epoch')
            self.encoder_type = conf.get('MAGNN', 'encoder_type')
            self.mini_batch_flag = conf.getboolean("MAGNN", "mini_batch_flag")
            if self.mini_batch_flag:
                self.batch_size = conf.getint("MAGNN", "batch_size")
                self.num_samples = conf.getint("MAGNN", "num_samples")
            self.hidden_dim = self.h_dim * self.num_heads

        elif self.model_name == 'RHGNN':
            self.lr = conf.getfloat("RHGNN", "learning_rate")
            self.num_heads = conf.getint("RHGNN", "num_heads")
            self.hidden_dim = conf.getint("RHGNN", "hidden_dim")
            self.relation_hidden_units = conf.getint("RHGNN", "relation_hidden_units")
            self.drop_out = conf.getfloat("RHGNN", "drop_out")
            self.num_layers = conf.getint("RHGNN", "num_layers")
            self.residual = conf.getboolean("RHGNN", "residual")
            self.batch_size = conf.getint("RHGNN", "batch_size")
            self.node_neighbors_min_num = conf.getint("RHGNN", "node_neighbors_min_num")
            # self.optimizer = conf.get
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

        elif self.model_name == 'HGNN_AC':
            self.feats_drop_rate = conf.getfloat("HGNN_AC", "feats_drop_rate")
            self.attn_vec_dim = conf.getint("HGNN_AC", "attn_vec_dim")
            self.feats_opt = conf.get("HGNN_AC", "feats_opt")
            self.loss_lambda = conf.getfloat("HGNN_AC", "loss_lambda")
            self.src_node_type = conf.getint("HGNN_AC", "src_node_type")
            self.HIN = conf.get("HGNN_AC", "HIN")
            if self.HIN == "MAGNN":
                self.lr = conf.getfloat("MAGNN", "learning_rate")
                self.weight_decay = conf.getfloat("MAGNN", "weight_decay")
                self.seed = conf.getint("MAGNN", "seed")
                self.dropout = conf.getfloat("MAGNN", "dropout")

                self.inter_attn_feats = conf.getint("MAGNN", "inter_attn_feats")
                self.h_dim = conf.getint('MAGNN', 'h_dim')
                self.out_dim = conf.getint('MAGNN', 'out_dim')
                self.num_heads = conf.getint('MAGNN', 'num_heads')
                self.num_layers = conf.getint("MAGNN", "num_layers")

                self.patience = conf.getint('MAGNN', 'patience')
                self.max_epoch = conf.getint('MAGNN', 'max_epoch')
                self.mini_batch_flag = conf.getboolean("MAGNN", "mini_batch_flag")
                self.encoder_type = conf.get('MAGNN', 'encoder_type')
                self.hidden_dim = self.h_dim * self.num_heads
            elif self.HIN == "GTN":
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
                self.dropout = conf.getfloat("HGNN_AC", "dropout")
                self.num_heads = conf.getint('HGNN_AC', 'num_heads')
            elif self.HIN == "MHNF":
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
                self.dropout = 0.2
                self.num_heads = 8

        elif self.model_name == 'HGT':
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
            self.norm = conf.getboolean("HGT", "norm")
            self.num_layers = conf.getint("HGT", "num_layers")
            self.num_heads = conf.getint("HGT", "num_heads")
        elif self.model_name == 'HeCo':
            self.lr = conf.getfloat("HeCo", "learning_rate")
            self.weight_decay = conf.getfloat("HeCo", "weight_decay")
            self.seed = conf.getint("HeCo", "seed")

            self.hidden_dim = conf.getint('HeCo', 'hidden_dim')
            self.patience = conf.getint('HeCo', 'patience')
            self.max_epoch = conf.getint('HeCo', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HeCo", "mini_batch_flag")

            self.feat_drop = conf.getfloat("HeCo", "feat_drop")
            self.attn_drop = conf.getfloat("HeCo", "attn_drop")
            self.eva_lr = conf.getfloat("HeCo", "eva_lr")
            self.eva_wd = conf.getfloat("HeCo", "eva_wd")
            sample_rate = conf.get('HeCo', 'sample_rate').split('_')
            # self.sample_rate = [int(i) for i in sample_rate]
            self.sample_rate = {}
            for i in sample_rate:
                one = i.split('-')
                self.sample_rate[one[0]] = int(one[1])
            self.tau = conf.getfloat("HeCo", "tau")
            self.lam = conf.getfloat("HeCo", "lam")

        elif self.model_name == 'DMGI':
            self.lr = conf.getfloat("DMGI", "learning_rate")
            self.weight_decay = conf.getfloat("DMGI", "weight_decay")
            self.sc = conf.getint("DMGI", "sc")
            self.seed = conf.getint("DMGI", "seed")
            self.sup_coef = conf.getfloat("DMGI", 'sup_coef')
            self.reg_coef = conf.getfloat("DMGI", "reg_coef")
            self.dropout = conf.getfloat("DMGI", "dropout")
            self.hidden_dim = conf.getint('DMGI', 'hidden_dim')
            self.num_heads = conf.getint('DMGI', 'num_heads')
            self.patience = conf.getint('DMGI', 'patience')
            self.max_epoch = conf.getint('DMGI', 'max_epoch')
            self.isSemi = conf.getboolean("DMGI", "isSemi")
            self.isBias = conf.getboolean("DMGI", "isBias")
            self.isAttn = conf.getboolean("DMGI", "isAttn")

        elif self.model_name == 'SLiCE':
            self.data_name = conf.get('SLiCE', 'data_name')
            self.num_walks_per_node = conf.getint('SLiCE', 'num_walks_per_node')
            self.beam_width = conf.getint('SLiCE', 'beam_width')
            self.max_length = conf.getint('SLiCE', 'max_length')
            self.walk_type = conf.get("SLiCE", 'walk_type')
            self.batch_size = conf.getint('SLiCE', 'batch_size')
            self.outdir = conf.get('SLiCE', 'outdir')
            self.n_pred = conf.getint('SLiCE', 'n_pred')
            self.max_pred = conf.getint('SLiCE', 'max_pred')
            self.lr = conf.getfloat('SLiCE', 'lr')
            self.n_epochs = conf.getint('SLiCE', 'n_epochs')
            self.get_bert_encoder_embeddings = conf.getboolean('SLiCE', 'get_bert_encoder_embeddings')
            self.checkpoint = conf.getint('SLiCE', 'checkpoint')
            self.path_option = conf.get("SLiCE", 'path_option')
            self.ft_batch_size = conf.getint('SLiCE', 'ft_batch_size')
            # self.embed_dir=conf.get('SLiCE','embed_dir')
            self.d_model = conf.getint('SLiCE', 'd_model')
            self.ft_d_ff = conf.getint('SLiCE', 'ft_d_ff')
            self.ft_layer = conf.get('SLiCE', 'ft_layer')
            self.ft_drop_rate = conf.getfloat('SLiCE', 'ft_drop_rate')
            self.ft_input_option = conf.get('SLiCE', 'ft_input_option')
            self.num_layers = conf.getint('SLiCE', 'num_layers')
            self.ft_lr = conf.getfloat('SLiCE', 'ft_lr')
            self.ft_n_epochs = conf.getint('SLiCE', 'ft_n_epochs')
            self.ft_checkpoint = conf.getint('SLiCE', 'ft_checkpoint')
            self.pretrained_embeddings = conf.get('SLiCE', 'pretrained_embeddings')
        elif self.model_name == 'HPN':
            self.lr = conf.getfloat("HPN", "learning_rate")
            self.weight_decay = conf.getfloat("HPN", "weight_decay")
            self.seed = conf.getint("HPN", "seed")
            self.dropout = conf.getfloat("HPN", "dropout")
            self.hidden_dim = conf.getint('HPN', 'hidden_dim')
            self.k_layer = conf.getint("HPN", "k_layer")
            self.alpha = conf.getfloat("HPN", "alpha")
            self.edge_drop = conf.getfloat("HPN", "edge_drop")
            self.patience = conf.getint('HPN', 'patience')
            self.max_epoch = conf.getint('HPN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HPN", "mini_batch_flag")
        elif self.model_name == 'KGCN':
            self.weight_decay = conf.getfloat("KGCN", "weight_decay")
            self.batch_size = conf.getint("KGCN", "batch_size")
            self.in_dim = conf.getint('KGCN', 'in_dim')
            self.out_dim = conf.getint('KGCN', 'out_dim')
            self.lr = conf.getfloat("KGCN", "lr")
            self.n_neighbor = conf.getint("KGCN", "n_neighbor")
            self.n_relation = conf.getint("KGCN", "n_relation")
            self.aggregate = conf.get("KGCN", "aggregate")
            self.n_item = conf.getint("KGCN", "n_relation")
            self.n_user = conf.getint("KGCN", "n_user")
            self.epoch_iter = conf.getint("KGCN", "epoch_iter")

        elif self.model_name == 'general_HGNN':
            self.lr = conf.getfloat("general_HGNN", "lr")
            self.weight_decay = conf.getfloat("general_HGNN", "weight_decay")
            self.dropout = conf.getfloat("general_HGNN", "dropout")

            self.hidden_dim = conf.getint('general_HGNN', 'hidden_dim')
            self.num_heads = conf.getint('general_HGNN', 'num_heads')
            self.patience = conf.getint('general_HGNN', 'patience')
            self.max_epoch = conf.getint('general_HGNN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("general_HGNN", "mini_batch_flag")
            self.layers_gnn = conf.getint("general_HGNN", "layers_gnn")
            self.layers_pre_mp = conf.getint("general_HGNN", "layers_pre_mp")
            self.layers_post_mp = conf.getint("general_HGNN", "layers_post_mp")
            self.stage_type = conf.get('general_HGNN', 'stage_type')
            self.gnn_type = conf.get('general_HGNN', 'gnn_type')
            self.activation = conf.get('general_HGNN', 'activation')
            self.activation = act_dict[self.activation]
            self.subgraph_extraction = conf.get('general_HGNN', 'subgraph_extraction')
            self.feat = conf.getint('general_HGNN', 'feat')
            self.has_bn = conf.getboolean('general_HGNN', 'has_bn')
            self.has_l2norm = conf.getboolean('general_HGNN', 'has_l2norm')
            self.macro_func = conf.get('general_HGNN', 'macro_func')

        elif self.model_name == 'homo_GNN':
            self.lr = conf.getfloat("homo_GNN", "lr")
            self.weight_decay = conf.getfloat("homo_GNN", "weight_decay")
            self.dropout = conf.getfloat("homo_GNN", "dropout")

            self.hidden_dim = conf.getint('homo_GNN', 'hidden_dim')
            self.num_heads = conf.getint('homo_GNN', 'num_heads')
            self.patience = conf.getint('homo_GNN', 'patience')
            self.max_epoch = conf.getint('homo_GNN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("homo_GNN", "mini_batch_flag")
            self.layers_gnn = conf.getint("homo_GNN", "layers_gnn")
            self.layers_pre_mp = conf.getint("homo_GNN", "layers_pre_mp")
            self.layers_post_mp = conf.getint("homo_GNN", "layers_post_mp")
            self.stage_type = conf.get('homo_GNN', 'stage_type')
            self.gnn_type = conf.get('homo_GNN', 'gnn_type')
            self.activation = conf.get('homo_GNN', 'activation')
            self.activation = act_dict[self.activation]
            self.subgraph = conf.get('homo_GNN', 'subgraph')
            self.feat = conf.getint('homo_GNN', 'feat')
            self.has_bn = conf.getboolean('homo_GNN', 'has_bn')
            self.has_l2norm = conf.getboolean('homo_GNN', 'has_l2norm')
        elif self.model_name == 'HeGAN':
            self.lr_gen = conf.getfloat('HeGAN', 'lr_gen')
            self.lr_dis = conf.getfloat('HeGAN', 'lr_dis')
            self.sigma = conf.getfloat('HeGAN', 'sigma')
            self.n_sample = conf.getint('HeGAN', 'n_sample')
            self.max_epoch = conf.getint('HeGAN', 'max_epoch')
            self.epoch_dis = conf.getint('HeGAN', 'epoch_dis')
            self.epoch_gen = conf.getint('HeGAN', 'epoch_gen')
            self.wd_dis = conf.getfloat("HeGAN", 'wd_dis')
            self.wd_gen = conf.getfloat('HeGAN', 'wd_gen')
            self.mini_batch_flag = conf.getboolean('HeGAN', 'mini_batch_flag')
            self.validation = conf.getboolean('HeGAN', 'validation')
            self.emb_size = conf.getint("HeGAN", 'emb_size')
            self.patience = conf.getint("HeGAN", 'patience')
            self.label_smooth = conf.getfloat("HeGAN", 'label_smooth')
        elif self.model_name == 'HDE':
            self.emb_dim = conf.getint('HDE', 'emb_dim')
            self.num_neighbor = conf.getint('HDE', 'num_neighbor')
            self.use_bias = conf.getboolean('HDE', 'use_bias')
            self.k_hop = conf.getint('HDE', 'k_hop')
            self.max_epoch = conf.getint('HDE', 'max_epoch')
            self.batch_size = conf.getint('HDE', 'batch_size')
            self.max_dist = conf.getint('HDE', 'max_dist')
            self.lr = conf.getfloat('HDE', 'lr')
        elif self.model_name == 'SimpleHGN':
            self.weight_decay = conf.getfloat("SimpleHGN", "weight_decay")
            self.lr = conf.getfloat("SimpleHGN", "lr")
            self.max_epoch = conf.getint("SimpleHGN", "max_epoch")
            self.seed = conf.getint("SimpleHGN", "seed")
            self.patience = conf.getint("SimpleHGN", "patience")
            self.edge_dim = conf.getint("SimpleHGN", "edge_dim")
            self.slope = conf.getfloat("SimpleHGN", "slope")
            self.feats_drop_rate = conf.getfloat("SimpleHGN", "feats_drop_rate")
            self.num_heads = conf.getint("SimpleHGN", "num_heads")
            self.h_dim = conf.getint("SimpleHGN", "h_dim")
            self.num_layers = conf.getint("SimpleHGN", "num_layers")
            self.beta = conf.getfloat("SimpleHGN", "beta")
            self.residual = conf.getboolean("SimpleHGN", "residual")
            self.mini_batch_flag = False
            self.hidden_dim = self.h_dim * self.num_heads

        elif self.model_name == 'GATNE-T':
            self.learning_rate = conf.getfloat("GATNE-T", "learning_rate")
            self.patience = conf.getint("GATNE-T", "patience")
            self.max_epoch = conf.getint("GATNE-T", "max_epoch")
            self.batch_size = conf.getint("GATNE-T", "batch_size")
            self.num_workers = conf.getint("GATNE-T", "num_workers")
            self.dim = conf.getint("GATNE-T", "dim")
            self.edge_dim = conf.getint("GATNE-T", "edge_dim")
            self.att_dim = conf.getint("GATNE-T", "att_dim")
            self.rw_length = conf.getint("GATNE-T", "rw_length")
            self.rw_walks = conf.getint("GATNE-T", "rw_walks")
            self.window_size = conf.getint("GATNE-T", "window_size")
            self.neg_size = conf.getint("GATNE-T", "neg_size")
            self.neighbor_samples = conf.getint("GATNE-T", "neighbor_samples")
            self.score_fn = conf.get("GATNE-T", "score_fn")

        elif self.model_name == 'HetSANN':
            self.lr = conf.getfloat("HetSANN", "lr")
            self.weight_decay = conf.getfloat("HetSANN", "weight_decay")
            self.dropout = conf.getfloat("HetSANN", "dropout")
            self.seed = conf.getint("HetSANN", "seed")
            self.h_dim = conf.getint("HetSANN", "h_dim")
            self.num_layers = conf.getint("HetSANN", "num_layers")
            self.num_heads = conf.getint("HetSANN", "num_heads")
            self.max_epoch = conf.getint("HetSANN", "max_epoch")
            self.patience = conf.getint("HetSANN", "patience")
            self.slope = conf.getfloat("HetSANN", "slope")
            self.residual = conf.getboolean("HetSANN", "residual")
            self.mini_batch_flag = False
            self.hidden_dim = self.h_dim * self.num_heads
            self.mini_batch_flag = False
            self.hidden_dim = self.h_dim * self.num_heads
        elif self.model_name == 'ieHGCN':
            self.weight_decay = conf.getfloat("ieHGCN", "weight_decay")
            self.lr = conf.getfloat("ieHGCN", "lr")
            self.max_epoch = conf.getint("ieHGCN", "max_epoch")
            self.seed = conf.getint("ieHGCN", "seed")
            self.attn_dim = conf.getint("ieHGCN", "attn_dim")
            self.num_layers = conf.getint("ieHGCN","num_layers")
            self.mini_batch_flag = False
            self.hidden_dim = conf.getint("ieHGCN", "hidden_dim")
            self.in_dim = conf.getint("ieHGCN", "in_dim")
            self.out_dim = conf.getint("ieHGCN", "out_dim")
            self.patience = conf.getint("ieHGCN", "patience")
        elif self.model_name == 'HGAT':
            self.weight_decay = conf.getfloat("HGAT", "weight_decay")
            self.lr = conf.getfloat("HGAT", "lr")
            self.max_epoch = conf.getint("HGAT", "max_epoch")
            self.seed = conf.getint("HGAT", "seed")
            self.attn_dim = conf.getint("HGAT", "attn_dim")
            self.num_layers = conf.getint("HGAT","num_layers")
            self.mini_batch_flag = False
            self.hidden_dim = conf.getint("HGAT", "hidden_dim")
            self.in_dim = conf.getint("HGAT", "in_dim")
            self.num_classes = conf.getint("HGAT", "num_classes")
            self.patience = conf.getint("HGAT", "patience")
            self.negative_slope = conf.getfloat("HGAT", "negative_slope")

        elif self.model_name == 'TransE':
            self.seed = conf.getint("TransE", "seed")
            self.patience = conf.getint("TransE", "patience")
            self.batch_size = conf.getint("TransE", "batch_size")
            self.neg_size = conf.getint("TransE", "neg_size")
            self.dis_norm = conf.getint("TransE", "dis_norm")
            self.margin = conf.getfloat("TransE", "margin")
            self.hidden_dim = conf.getint("TransE", "hidden_dim")
            self.optimizer = conf.get("TransE", "optimizer")
            self.lr = conf.getfloat("TransE", "lr")
            self.weight_decay = conf.getfloat("TransE", "weight_decay")
            self.max_epoch = conf.getint("TransE", "max_epoch")
            self.score_fn = conf.get("TransE", "score_fn")
            self.filtered = conf.get("TransE", "filtered")
            self.valid_percent = conf.getfloat("TransE", "valid_percent")
            self.test_percent = conf.getfloat("TransE", "test_percent")
            self.mini_batch_flag = True

        elif self.model_name == 'TransH':
            self.seed = conf.getint("TransH", "seed")
            self.patience = conf.getint("TransH", "patience")
            self.batch_size = conf.getint("TransH", "batch_size")
            self.neg_size = conf.getint("TransH", "neg_size")
            self.dis_norm = conf.getint("TransH", "dis_norm")
            self.margin = conf.getfloat("TransH", "margin")
            self.hidden_dim = conf.getint("TransH", "hidden_dim")
            self.optimizer = conf.get("TransH", "optimizer")
            self.lr = conf.getfloat("TransH", "lr")
            self.weight_decay = conf.getfloat("TransH", "weight_decay")
            self.max_epoch = conf.getint("TransH", "max_epoch")
            self.score_fn = conf.get("TransH", "score_fn")
            self.filtered = conf.get("TransH", "filtered")
            self.valid_percent = conf.getfloat("TransH", "valid_percent")
            self.test_percent = conf.getfloat("TransH", "test_percent")
            self.mini_batch_flag = True
        
        elif self.model_name == 'TransR':
            self.seed = conf.getint("TransR", "seed")
            self.patience = conf.getint("TransR", "patience")
            self.batch_size = conf.getint("TransR", "batch_size")
            self.neg_size = conf.getint("TransR", "neg_size")
            self.dis_norm = conf.getint("TransR", "dis_norm")
            self.margin = conf.getfloat("TransR", "margin")
            self.ent_dim = conf.getint("TransR", "ent_dim")
            self.rel_dim = conf.getint("TransR", "rel_dim")
            self.optimizer = conf.get("TransR", "optimizer")
            self.lr = conf.getfloat("TransR", "lr")
            self.weight_decay = conf.getfloat("TransR", "weight_decay")
            self.max_epoch = conf.getint("TransR", "max_epoch")
            self.score_fn = conf.get("TransR", "score_fn")
            self.filtered = conf.get("TransR", "filtered")
            self.valid_percent = conf.getfloat("TransR", "valid_percent")
            self.test_percent = conf.getfloat("TransR", "test_percent")
            self.mini_batch_flag = True
        
        elif self.model_name == 'TransD':
            self.seed = conf.getint("TransD", "seed")
            self.patience = conf.getint("TransD", "patience")
            self.batch_size = conf.getint("TransD", "batch_size")
            self.neg_size = conf.getint("TransD", "neg_size")
            self.dis_norm = conf.getint("TransD", "dis_norm")
            self.margin = conf.getfloat("TransD", "margin")
            self.ent_dim = conf.getint("TransD", "ent_dim")
            self.rel_dim = conf.getint("TransD", "rel_dim")
            self.optimizer = conf.get("TransD", "optimizer")
            self.lr = conf.getfloat("TransD", "lr")
            self.weight_decay = conf.getfloat("TransD", "weight_decay")
            self.max_epoch = conf.getint("TransD", "max_epoch")
            self.score_fn = conf.get("TransD", "score_fn")
            self.filtered = conf.get("TransD", "filtered")
            self.valid_percent = conf.getfloat("TransD", "valid_percent")
            self.test_percent = conf.getfloat("TransD", "test_percent")
            self.mini_batch_flag = True

    def __repr__(self):
        return '[Config Info]\tModel: {},\tTask: {},\tDataset: {}'.format(self.model_name, self.task, self.dataset)
