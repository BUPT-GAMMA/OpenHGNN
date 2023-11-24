import configparser
import re

import numpy as np
import torch as th
from .utils.activation import act_dict
import warnings


class Config(object):
    def __init__(self, file_path, model, dataset, task, gpu):
        conf = configparser.ConfigParser()
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
            self.dataset_name = self.dataset.name
        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = type(self.model).__name__
        self.optimizer = "Adam"
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
            self.dim_size["emd"] = conf.getint("NSHE", "emd_dim")
            self.dim_size["context"] = conf.getint("NSHE", "context_dim")
            self.dim_size["project"] = conf.getint("NSHE", "project_dim")

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

        elif self.model_name == "RGCN":
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
            self.use_uva = conf.getboolean("RGCN", "use_uva")

        elif self.model_name == "CompGCN":
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
            self.fanout = conf.getint("CompGCN", "fanout")
            self.batch_size = conf.getint("CompGCN", "batch_size")
            pass
        elif self.model_name == "HetGNN":
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
        elif self.model_name == "Metapath2vec":
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

        elif self.model_name == "HERec":
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

        elif self.model_name == "HAN":
            self.lr = conf.getfloat("HAN", "learning_rate")
            self.weight_decay = conf.getfloat("HAN", "weight_decay")
            self.seed = conf.getint("HAN", "seed")
            self.dropout = conf.getfloat("HAN", "dropout")

            self.hidden_dim = conf.getint("HAN", "hidden_dim")
            self.out_dim = conf.getint("HAN", "out_dim")
            num_heads = conf.get("HAN", "num_heads").split("-")
            self.num_heads = [int(i) for i in num_heads]
            self.patience = conf.getint("HAN", "patience")
            self.max_epoch = conf.getint("HAN", "max_epoch")
            self.mini_batch_flag = conf.getboolean("HAN", "mini_batch_flag")

        elif self.model_name == "RoHe":
            self.lr = conf.getfloat("RoHe", "learning_rate")
            self.weight_decay = conf.getfloat("RoHe", "weight_decay")
            self.seed = conf.getint("RoHe", "seed")
            self.dropout = conf.getfloat("RoHe", "dropout")

            self.hidden_dim = conf.getint("RoHe", "hidden_dim")
            self.out_dim = conf.getint("RoHe", "out_dim")
            num_heads = conf.get("RoHe", "num_heads").split("-")
            self.num_heads = [int(i) for i in num_heads]
            self.patience = conf.getint("RoHe", "patience")
            self.max_epoch = conf.getint("RoHe", "max_epoch")
            self.mini_batch_flag = conf.getboolean("RoHe", "mini_batch_flag")

        elif self.model_name == "NARS":
            self.lr = conf.getfloat("NARS", "learning_rate")
            self.weight_decay = conf.getfloat("NARS", "weight_decay")
            self.seed = conf.getint("NARS", "seed")
            self.dropout = conf.getfloat("NARS", "dropout")
            self.patience = conf.getint("HAN", "patience")
            self.hidden_dim = conf.getint("NARS", "hidden_dim")
            self.out_dim = conf.getint("NARS", "out_dim")
            num_heads = conf.get("NARS", "num_heads").split("-")
            self.num_heads = [int(i) for i in num_heads]
            self.num_hops = conf.getint("NARS", "num_hops")

            self.max_epoch = conf.getint("NARS", "max_epoch")
            self.mini_batch_flag = conf.getboolean("NARS", "mini_batch_flag")
            self.R = conf.getint("NARS", "R")
            self.cpu_preprocess = conf.getboolean("NARS", "cpu_preprocess")
            self.input_dropout = conf.getboolean("NARS", "input_dropout")

            self.ff_layer = conf.getint("NARS", "ff_layer")

        elif self.model_name == "MAGNN":
            self.lr = conf.getfloat("MAGNN", "learning_rate")
            self.weight_decay = conf.getfloat("MAGNN", "weight_decay")
            self.seed = conf.getint("MAGNN", "seed")
            self.dropout = conf.getfloat("MAGNN", "dropout")

            self.inter_attn_feats = conf.getint("MAGNN", "inter_attn_feats")
            self.hidden_dim = conf.getint("MAGNN", "hidden_dim")
            self.out_dim = conf.getint("MAGNN", "out_dim")
            self.num_heads = conf.getint("MAGNN", "num_heads")
            self.num_layers = conf.getint("MAGNN", "num_layers")

            self.patience = conf.getint("MAGNN", "patience")
            self.max_epoch = conf.getint("MAGNN", "max_epoch")
            self.encoder_type = conf.get("MAGNN", "encoder_type")
            self.mini_batch_flag = conf.getboolean("MAGNN", "mini_batch_flag")
            if self.mini_batch_flag:
                self.batch_size = conf.getint("MAGNN", "batch_size")
                self.num_samples = conf.getint("MAGNN", "num_samples")

        elif self.model_name == "RHGNN":
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
            self.use_uva = conf.getboolean("RHGNN", "use_uva")
            self.fanout = conf.getint("RHGNN", "fanout")

        elif self.model_name == "HGNN_AC":
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
                self.hidden_dim = conf.getint("MAGNN", "hidden_dim")
                self.out_dim = conf.getint("MAGNN", "out_dim")
                self.num_heads = conf.getint("MAGNN", "num_heads")
                self.num_layers = conf.getint("MAGNN", "num_layers")

                self.patience = conf.getint("MAGNN", "patience")
                self.max_epoch = conf.getint("MAGNN", "max_epoch")
                self.mini_batch_flag = conf.getboolean("MAGNN", "mini_batch_flag")
                self.encoder_type = conf.get("MAGNN", "encoder_type")
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
                self.num_heads = conf.getint("HGNN_AC", "num_heads")
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

        elif self.model_name == "HGT":
            self.lr = conf.getfloat("HGT", "learning_rate")
            self.weight_decay = conf.getfloat("HGT", "weight_decay")
            self.seed = conf.getint("HGT", "seed")
            self.dropout = conf.getfloat("HGT", "dropout")

            self.batch_size = conf.getint("HGT", "batch_size")
            self.hidden_dim = conf.getint("HGT", "hidden_dim")
            self.out_dim = conf.getint("HGT", "out_dim")
            self.num_heads = conf.getint("HGT", "num_heads")
            self.patience = conf.getint("HGT", "patience")
            self.max_epoch = conf.getint("HGT", "max_epoch")
            self.num_workers = conf.getint("HGT", "num_workers")
            self.mini_batch_flag = conf.getboolean("HGT", "mini_batch_flag")
            self.fanout = conf.getint("HGT", "fanout")
            self.norm = conf.getboolean("HGT", "norm")
            self.num_layers = conf.getint("HGT", "num_layers")
            self.num_heads = conf.getint("HGT", "num_heads")
            self.use_uva = conf.getboolean("HGT", "use_uva")
        elif self.model_name == "HeCo":
            self.lr = conf.getfloat("HeCo", "learning_rate")
            self.weight_decay = conf.getfloat("HeCo", "weight_decay")
            self.seed = conf.getint("HeCo", "seed")

            self.hidden_dim = conf.getint("HeCo", "hidden_dim")
            self.patience = conf.getint("HeCo", "patience")
            self.max_epoch = conf.getint("HeCo", "max_epoch")
            self.mini_batch_flag = conf.getboolean("HeCo", "mini_batch_flag")

            self.feat_drop = conf.getfloat("HeCo", "feat_drop")
            self.attn_drop = conf.getfloat("HeCo", "attn_drop")
            self.eva_lr = conf.getfloat("HeCo", "eva_lr")
            self.eva_wd = conf.getfloat("HeCo", "eva_wd")
            sample_rate = conf.get("HeCo", "sample_rate").split("_")
            # self.sample_rate = [int(i) for i in sample_rate]
            self.sample_rate = {}
            for i in sample_rate:
                one = i.split("-")
                self.sample_rate[one[0]] = int(one[1])
            self.tau = conf.getfloat("HeCo", "tau")
            self.lam = conf.getfloat("HeCo", "lam")

        elif self.model_name == "DMGI":
            self.lr = conf.getfloat("DMGI", "learning_rate")
            self.weight_decay = conf.getfloat("DMGI", "weight_decay")
            self.sc = conf.getint("DMGI", "sc")
            self.seed = conf.getint("DMGI", "seed")
            self.sup_coef = conf.getfloat("DMGI", "sup_coef")
            self.reg_coef = conf.getfloat("DMGI", "reg_coef")
            self.dropout = conf.getfloat("DMGI", "dropout")
            self.hidden_dim = conf.getint("DMGI", "hidden_dim")
            self.num_heads = conf.getint("DMGI", "num_heads")
            self.patience = conf.getint("DMGI", "patience")
            self.max_epoch = conf.getint("DMGI", "max_epoch")
            self.isSemi = conf.getboolean("DMGI", "isSemi")
            self.isBias = conf.getboolean("DMGI", "isBias")
            self.isAttn = conf.getboolean("DMGI", "isAttn")

        elif self.model_name == "SLiCE":
            self.data_name = conf.get("SLiCE", "data_name")
            self.num_walks_per_node = conf.getint("SLiCE", "num_walks_per_node")
            self.beam_width = conf.getint("SLiCE", "beam_width")
            self.max_length = conf.getint("SLiCE", "max_length")
            self.walk_type = conf.get("SLiCE", "walk_type")
            self.batch_size = conf.getint("SLiCE", "batch_size")
            self.outdir = conf.get("SLiCE", "outdir")
            self.n_pred = conf.getint("SLiCE", "n_pred")
            self.max_pred = conf.getint("SLiCE", "max_pred")
            self.lr = conf.getfloat("SLiCE", "lr")
            self.n_epochs = conf.getint("SLiCE", "n_epochs")
            self.get_bert_encoder_embeddings = conf.getboolean(
                "SLiCE", "get_bert_encoder_embeddings"
            )
            self.checkpoint = conf.getint("SLiCE", "checkpoint")
            self.path_option = conf.get("SLiCE", "path_option")
            self.ft_batch_size = conf.getint("SLiCE", "ft_batch_size")
            # self.embed_dir=conf.get('SLiCE','embed_dir')
            self.d_model = conf.getint("SLiCE", "d_model")
            self.ft_d_ff = conf.getint("SLiCE", "ft_d_ff")
            self.ft_layer = conf.get("SLiCE", "ft_layer")
            self.ft_drop_rate = conf.getfloat("SLiCE", "ft_drop_rate")
            self.ft_input_option = conf.get("SLiCE", "ft_input_option")
            self.num_layers = conf.getint("SLiCE", "num_layers")
            self.ft_lr = conf.getfloat("SLiCE", "ft_lr")
            self.ft_n_epochs = conf.getint("SLiCE", "ft_n_epochs")
            self.ft_checkpoint = conf.getint("SLiCE", "ft_checkpoint")
            self.pretrained_embeddings = conf.get("SLiCE", "pretrained_embeddings")
        elif self.model_name == "HPN":
            self.lr = conf.getfloat("HPN", "learning_rate")
            self.weight_decay = conf.getfloat("HPN", "weight_decay")
            self.seed = conf.getint("HPN", "seed")
            self.dropout = conf.getfloat("HPN", "dropout")
            self.hidden_dim = conf.getint("HPN", "hidden_dim")
            self.k_layer = conf.getint("HPN", "k_layer")
            self.alpha = conf.getfloat("HPN", "alpha")
            self.edge_drop = conf.getfloat("HPN", "edge_drop")
            self.patience = conf.getint("HPN", "patience")
            self.max_epoch = conf.getint("HPN", "max_epoch")
            self.mini_batch_flag = conf.getboolean("HPN", "mini_batch_flag")
        elif self.model_name == "KGCN":
            self.weight_decay = conf.getfloat("KGCN", "weight_decay")
            self.batch_size = conf.getint("KGCN", "batch_size")
            self.in_dim = conf.getint("KGCN", "in_dim")
            self.out_dim = conf.getint("KGCN", "out_dim")
            self.lr = conf.getfloat("KGCN", "lr")
            self.n_neighbor = conf.getint("KGCN", "n_neighbor")
            self.n_relation = conf.getint("KGCN", "n_relation")
            self.aggregate = conf.get("KGCN", "aggregate")
            self.n_item = conf.getint("KGCN", "n_relation")
            self.n_user = conf.getint("KGCN", "n_user")
            # self.epoch_iter = conf.getint("KGCN", "epoch_iter")
            self.max_epoch = conf.getint("KGCN", "max_epoch")

        elif self.model_name == "general_HGNN":
            self.lr = conf.getfloat("general_HGNN", "lr")
            self.weight_decay = conf.getfloat("general_HGNN", "weight_decay")
            self.dropout = conf.getfloat("general_HGNN", "dropout")

            self.hidden_dim = conf.getint("general_HGNN", "hidden_dim")
            self.num_heads = conf.getint("general_HGNN", "num_heads")
            self.patience = conf.getint("general_HGNN", "patience")
            self.max_epoch = conf.getint("general_HGNN", "max_epoch")
            self.mini_batch_flag = conf.getboolean("general_HGNN", "mini_batch_flag")
            self.layers_gnn = conf.getint("general_HGNN", "layers_gnn")
            self.layers_pre_mp = conf.getint("general_HGNN", "layers_pre_mp")
            self.layers_post_mp = conf.getint("general_HGNN", "layers_post_mp")
            self.stage_type = conf.get("general_HGNN", "stage_type")
            self.gnn_type = conf.get("general_HGNN", "gnn_type")
            self.activation = conf.get("general_HGNN", "activation")
            self.activation = act_dict[self.activation]
            self.subgraph_extraction = conf.get("general_HGNN", "subgraph_extraction")
            self.feat = conf.getint("general_HGNN", "feat")
            self.has_bn = conf.getboolean("general_HGNN", "has_bn")
            self.has_l2norm = conf.getboolean("general_HGNN", "has_l2norm")
            self.macro_func = conf.get("general_HGNN", "macro_func")

        elif self.model_name == "homo_GNN":
            self.lr = conf.getfloat("homo_GNN", "lr")
            self.weight_decay = conf.getfloat("homo_GNN", "weight_decay")
            self.dropout = conf.getfloat("homo_GNN", "dropout")

            self.hidden_dim = conf.getint("homo_GNN", "hidden_dim")
            self.num_heads = conf.getint("homo_GNN", "num_heads")
            self.patience = conf.getint("homo_GNN", "patience")
            self.max_epoch = conf.getint("homo_GNN", "max_epoch")
            self.mini_batch_flag = conf.getboolean("homo_GNN", "mini_batch_flag")
            self.layers_gnn = conf.getint("homo_GNN", "layers_gnn")
            self.layers_pre_mp = conf.getint("homo_GNN", "layers_pre_mp")
            self.layers_post_mp = conf.getint("homo_GNN", "layers_post_mp")
            self.stage_type = conf.get("homo_GNN", "stage_type")
            self.gnn_type = conf.get("homo_GNN", "gnn_type")
            self.activation = conf.get("homo_GNN", "activation")
            self.activation = act_dict[self.activation]
            self.subgraph = conf.get("homo_GNN", "subgraph")
            self.feat = conf.getint("homo_GNN", "feat")
            self.has_bn = conf.getboolean("homo_GNN", "has_bn")
            self.has_l2norm = conf.getboolean("homo_GNN", "has_l2norm")
        elif self.model_name == "HeGAN":
            self.lr_gen = conf.getfloat("HeGAN", "lr_gen")
            self.lr_dis = conf.getfloat("HeGAN", "lr_dis")
            self.sigma = conf.getfloat("HeGAN", "sigma")
            self.n_sample = conf.getint("HeGAN", "n_sample")
            self.max_epoch = conf.getint("HeGAN", "max_epoch")
            self.epoch_dis = conf.getint("HeGAN", "epoch_dis")
            self.epoch_gen = conf.getint("HeGAN", "epoch_gen")
            self.wd_dis = conf.getfloat("HeGAN", "wd_dis")
            self.wd_gen = conf.getfloat("HeGAN", "wd_gen")
            self.mini_batch_flag = conf.getboolean("HeGAN", "mini_batch_flag")
            self.validation = conf.getboolean("HeGAN", "validation")
            self.emb_size = conf.getint("HeGAN", "emb_size")
            self.patience = conf.getint("HeGAN", "patience")
            self.label_smooth = conf.getfloat("HeGAN", "label_smooth")
        elif self.model_name == "HDE":
            self.emb_dim = conf.getint("HDE", "emb_dim")
            self.num_neighbor = conf.getint("HDE", "num_neighbor")
            self.use_bias = conf.getboolean("HDE", "use_bias")
            self.k_hop = conf.getint("HDE", "k_hop")
            self.max_epoch = conf.getint("HDE", "max_epoch")
            self.batch_size = conf.getint("HDE", "batch_size")
            self.max_dist = conf.getint("HDE", "max_dist")
            self.lr = conf.getfloat("HDE", "lr")
        elif self.model_name == "SimpleHGN":
            self.weight_decay = conf.getfloat("SimpleHGN", "weight_decay")
            self.lr = conf.getfloat("SimpleHGN", "lr")
            self.max_epoch = conf.getint("SimpleHGN", "max_epoch")
            self.seed = conf.getint("SimpleHGN", "seed")
            self.patience = conf.getint("SimpleHGN", "patience")
            self.edge_dim = conf.getint("SimpleHGN", "edge_dim")
            self.slope = conf.getfloat("SimpleHGN", "slope")
            self.feats_drop_rate = conf.getfloat("SimpleHGN", "feats_drop_rate")
            self.num_heads = conf.getint("SimpleHGN", "num_heads")
            self.hidden_dim = conf.getint("SimpleHGN", "hidden_dim")
            self.num_layers = conf.getint("SimpleHGN", "num_layers")
            self.beta = conf.getfloat("SimpleHGN", "beta")
            self.residual = conf.getboolean("SimpleHGN", "residual")
            self.mini_batch_flag = conf.getboolean("SimpleHGN", "mini_batch_flag")
            self.fanout = conf.getint("SimpleHGN", "fanout")
            self.batch_size = conf.getint("SimpleHGN", "batch_size")
            self.use_uva = conf.getboolean("SimpleHGN", "use_uva")

        elif self.model_name == "GATNE-T":
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

        elif self.model_name == "HetSANN":
            self.lr = conf.getfloat("HetSANN", "lr")
            self.weight_decay = conf.getfloat("HetSANN", "weight_decay")
            self.dropout = conf.getfloat("HetSANN", "dropout")
            self.seed = conf.getint("HetSANN", "seed")
            self.hidden_dim = conf.getint("HetSANN", "hidden_dim")
            self.num_layers = conf.getint("HetSANN", "num_layers")
            self.num_heads = conf.getint("HetSANN", "num_heads")
            self.max_epoch = conf.getint("HetSANN", "max_epoch")
            self.patience = conf.getint("HetSANN", "patience")
            self.slope = conf.getfloat("HetSANN", "slope")
            self.residual = conf.getboolean("HetSANN", "residual")
            self.mini_batch_flag = conf.getboolean("HetSANN", "mini_batch_flag")
            self.batch_size = conf.getint("HetSANN", "batch_size")
            self.fanout = conf.getint("HetSANN", "fanout")
            self.use_uva = conf.getboolean("HetSANN", "use_uva")
        elif self.model_name == "ieHGCN":
            self.weight_decay = conf.getfloat("ieHGCN", "weight_decay")
            self.lr = conf.getfloat("ieHGCN", "lr")
            self.max_epoch = conf.getint("ieHGCN", "max_epoch")
            self.seed = conf.getint("ieHGCN", "seed")
            self.attn_dim = conf.getint("ieHGCN", "attn_dim")
            self.num_layers = conf.getint("ieHGCN", "num_layers")
            self.mini_batch_flag = conf.getboolean("ieHGCN", "mini_batch_flag")
            self.fanout = conf.getint("ieHGCN", "fanout")
            self.batch_size = conf.getint("ieHGCN", "batch_size")
            self.hidden_dim = conf.getint("ieHGCN", "hidden_dim")
            self.out_dim = conf.getint("ieHGCN", "out_dim")
            self.patience = conf.getint("ieHGCN", "patience")
            self.bias = conf.getboolean("ieHGCN", "bias")
            self.batchnorm = conf.getboolean("ieHGCN", "batchnorm")
            self.dropout = conf.getfloat("ieHGCN", "dropout")
        elif self.model_name == "HGAT":
            self.weight_decay = conf.getfloat("HGAT", "weight_decay")
            self.lr = conf.getfloat("HGAT", "lr")
            self.max_epoch = conf.getint("HGAT", "max_epoch")
            self.seed = conf.getint("HGAT", "seed")
            self.attn_dim = conf.getint("HGAT", "attn_dim")
            self.num_layers = conf.getint("HGAT", "num_layers")
            self.mini_batch_flag = False
            self.hidden_dim = conf.getint("HGAT", "hidden_dim")
            self.num_classes = conf.getint("HGAT", "num_classes")
            self.patience = conf.getint("HGAT", "patience")
            self.negative_slope = conf.getfloat("HGAT", "negative_slope")

        elif self.model_name == "HGSL":
            self.undirected_relations = conf.get("HGSL", "undirected_relations")
            self.gnn_dropout = conf.getfloat("HGSL", "gnn_dropout")
            self.fs_eps = conf.getfloat("HGSL", "fs_eps")
            self.fp_eps = conf.getfloat("HGSL", "fp_eps")
            self.mp_eps = conf.getfloat("HGSL", "mp_eps")
            self.hidden_dim = conf.getint("HGSL", "hidden_dim")
            self.num_heads = conf.getint("HGSL", "num_heads")
            self.gnn_emd_dim = conf.getint("HGSL", "gnn_emd_dim")
            self.lr = conf.getfloat("HGSL", "lr")
            self.weight_decay = conf.getfloat("HGSL", "weight_decay")
            self.mini_batch_flag = False
            self.max_epoch = conf.getint("HGSL", "max_epoch")

        elif self.model_name == "TransE":
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

        elif self.model_name == "TransH":
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

        elif self.model_name == "TransR":
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

        elif self.model_name == "TransD":
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

        elif self.model_name == "GIE":
            self.seed = conf.getint("GIE", "seed")
            self.patience = conf.getint("GIE", "patience")
            self.batch_size = conf.getint("GIE", "batch_size")
            self.neg_size = conf.getint("GIE", "neg_size")
            self.dis_norm = conf.getint("GIE", "dis_norm")
            self.margin = conf.getfloat("GIE", "margin")
            self.hidden_dim = conf.getint("GIE", "hidden_dim")
            self.optimizer = conf.get("GIE", "optimizer")
            self.lr = conf.getfloat("GIE", "lr")
            self.weight_decay = conf.getfloat("GIE", "weight_decay")
            self.max_epoch = conf.getint("GIE", "max_epoch")
            self.score_fn = conf.get("GIE", "score_fn")
            self.filtered = conf.get("GIE", "filtered")
            self.valid_percent = conf.getfloat("GIE", "valid_percent")
            self.test_percent = conf.getfloat("GIE", "test_percent")
            self.mini_batch_flag = True

        elif self.model_name == "GIN":
            self.hidden_dim = conf.getint("GIN", "hidden_dim")
            self.batch_size = conf.getint("GIN", "batch_size")
            self.lr = conf.getfloat("GIN", "lr")
            self.num_layers = conf.getint("GIN", "num_layers")
            self.out_dim = conf.getint("GIN", "out_dim")
            self.input_dim = conf.getint("GIN", "input_dim")
            self.weight_decay = conf.getfloat("GIN", "weight_decay")
            self.max_epoch = conf.getint("GIN", "max_epoch")
            self.patience = conf.getint("GIN", "patience")
            self.mini_batch_flag = conf.getboolean("GIN", "mini_batch_flag")
            self.learn_eps = conf.getboolean("GIN", "learn_eps")
            self.aggregate = conf.get("GIN", "aggregate")
            self.fanout = conf.getint("GIN", "fanout")

        elif self.model_name == "RGAT":
            self.weight_decay = conf.getfloat("RGAT", "weight_decay")
            self.lr = conf.getfloat("RGAT", "lr")
            self.max_epoch = conf.getint("RGAT", "max_epoch")
            self.seed = conf.getint("RGAT", "seed")
            self.num_layers = conf.getint("RGAT", "num_layers")
            self.mini_batch_flag = False
            self.hidden_dim = conf.getint("RGAT", "hidden_dim")
            self.in_dim = conf.getint("RGAT", "in_dim")
            self.patience = conf.getint("RGAT", "patience")
            self.num_heads = conf.getint("RGAT", "num_heads")
            self.dropout = conf.getfloat("RGAT", "dropout")
            self.out_dim = conf.getint("RGAT", "out_dim")

        elif self.model_name == "Rsage":
            self.weight_decay = conf.getfloat("Rsage", "weight_decay")
            self.lr = conf.getfloat("Rsage", "lr")
            self.max_epoch = conf.getint("Rsage", "max_epoch")
            self.seed = conf.getint("Rsage", "seed")
            self.num_layers = conf.getint("Rsage", "num_layers")
            self.mini_batch_flag = False
            self.hidden_dim = conf.getint("Rsage", "hidden_dim")
            self.in_dim = conf.getint("Rsage", "in_dim")
            self.patience = conf.getint("Rsage", "patience")
            self.aggregator_type = conf.get("Rsage", "aggregator_type")
            self.dropout = conf.getfloat("Rsage", "dropout")
            self.out_dim = conf.getint("Rsage", "out_dim")

        elif self.model_name == "Mg2vec":
            self.lr = conf.getfloat("MG2VEC", "learning_rate")
            self.max_epoch = conf.getint("MG2VEC", "max_epoch")
            self.emb_dimension = conf.getint("MG2VEC", "dim")
            self.batch_size = conf.getint("MG2VEC", "batch_size")
            self.num_workers = conf.getint("MG2VEC", "num_workers")
            self.sample_num = conf.getint("MG2VEC", "sample_num")
            self.alpha = conf.getfloat("MG2VEC", "alpha")
            self.seed = conf.getint("MG2VEC", "seed")

        elif self.model_name == "DHNE":
            self.lr = conf.getfloat("DHNE", "lr")
            emb_size = conf.getint("DHNE", "embedding_sizes")
            self.embedding_sizes = [emb_size, emb_size, emb_size]
            self.prefix_path = conf.get("DHNE", "prefix_path")
            self.hidden_size = conf.getint("DHNE", "hidden_size")
            self.epochs_to_train = conf.getint("DHNE", "epochs_to_train")
            self.max_epoch = conf.getint("DHNE", "max_epoch")
            self.batch_size = conf.getint("DHNE", "batch_size")
            self.alpha = conf.getfloat("DHNE", "alpha")
            self.num_neg_samples = conf.getint("DHNE", "num_neg_samples")
            self.seed = conf.getint("DHNE", "seed")
            self.dim_features = conf.get("DHNE", "dim_features")
            self.max_epoch = conf.getint("DHNE", "max_epoch")
            self.mini_batch_flag = True

        elif self.model_name == "DiffMG":
            self.lr = conf.getfloat("DiffMG", "lr")
            self.wd = conf.getfloat("DiffMG", "wd")
            self.dropout = conf.getfloat("DiffMG", "dropout")
            self.max_epoch = conf.getint("DiffMG", "max_epoch")
            self.hidden_dim = conf.getint("DiffMG", "hidden_dim")
            self.Amazon_train_seed = conf.getint("DiffMG", "Amazon_train_seed")
            self.Amazon_preprocess_seed = conf.getint(
                "DiffMG", "Amazon_preprocess_seed"
            )
            self.Amazon_gen_neg_seed = conf.getint("DiffMG", "Amazon_gen_neg_seed")
            self.embedding_sizes = conf.getint("DiffMG", "embedding_sizes")
            self.mini_batch_flag = conf.getboolean("DiffMG", "mini_batch_flag")
            self.attn_dim = conf.getint("DiffMG", "attn_dim")
            self.Amazon_search_seed = conf.getint("DiffMG", "Amazon_search_seed")
            self.search_lr = conf.getfloat("DiffMG", "search_lr")
            self.search_wd = conf.getfloat("DiffMG", "search_wd")
            self.search_alr = conf.getfloat("DiffMG", "search_alr")
            self.search_eps = conf.getfloat("DiffMG", "search_eps")
            self.search_decay = conf.getfloat("DiffMG", "search_decay")
            self.search_steps_s = conf.getint("DiffMG", "search_steps_s")
            self.search_steps_t = conf.getint("DiffMG", "search_steps_t")
            self.search_epochs = conf.getint("DiffMG", "search_epochs")
            # self.use_norm = conf.get("DiffMG", "use_norm")
            # self.out_nl = conf.get("DiffMG", "out_nl")

        elif self.model_name == "MeiREC":
            self.lr = conf.getfloat("MeiREC", "lr")
            self.weight_decay = conf.getfloat("MeiREC", "weight_decay")
            self.vocab = conf.getint("MeiREC", "vocab_size")
            self.max_epoch = conf.getint("MeiREC", "train_epochs")
            self.batch_num = conf.getint("MeiREC", "batch_num")

        elif self.model_name == "AEHCL":
            self.lr = conf.getfloat("AEHCL", "lr")
            self.hidden_dim = conf.getint("AEHCL", "hidden_dim")
            self.weight_intra_pair = conf.getfloat("AEHCL", "weight_intra_pair")
            self.weight_intra_multi = conf.getfloat("AEHCL", "weight_intra_multi")
            self.weight_inter = conf.getfloat("AEHCL", "weight_inter")
            self.num_of_attention_heads = conf.getint("AEHCL", "num_of_attention_heads")
            self.t = conf.getfloat("AEHCL", "t")
            self.batch_size = conf.getint("AEHCL", "batch_size")
            self.weight_decay = conf.getfloat("AEHCL", "weight_decay")
            self.eval_epoch = conf.getint("AEHCL", "eval_epoch")
            self.max_epoch = conf.getint("AEHCL", "max_epoch")
            self.neg_num = conf.getint("AEHCL", "neg_num")

        elif self.model_name == "KGAT":
            self.seed = conf.getint("KGAT", "seed")
            self.max_epoch = conf.getint("KGAT", "max_epoch")
            self.stopping_steps = conf.getint("KGAT", "stopping_steps")
            self.use_pretrain = conf.getint("KGAT", "use_pretrain")
            self.lr = conf.getfloat("KGAT", "lr")
            self.aggregation_type = conf.get("KGAT", "aggregation_type")
            self.entity_dim = conf.getint("KGAT", "entity_dim")
            self.relation_dim = conf.getint("KGAT", "relation_dim")
            self.conv_dim_list = conf.get("KGAT", "conv_dim_list")
            self.mess_dropout = conf.get("KGAT", "mess_dropout")
            self.cf_l2loss_lambda = conf.getfloat("KGAT", "cf_l2loss_lambda")
            self.kg_l2loss_lambda = conf.getfloat("KGAT", "kg_l2loss_lambda")
            self.cf_batch_size = conf.getint("KGAT", "cf_batch_size")
            self.kg_batch_size = conf.getint("KGAT", "kg_batch_size")
            self.test_batch_size = conf.getint("KGAT", "test_batch_size")
            self.multi_gpu = conf.getboolean("KGAT", "multi_gpu")
            self.K = conf.getint("KGAT", "K")

        elif self.model_name == "DSSL":
            self.epochs = conf.getint("DSSL", "epochs")
            self.lr = conf.getfloat("DSSL", "lr")
            self.weight_decay = conf.getfloat("DSSL", "weight_decay")
            self.hidden_channels = conf.getint("DSSL", "hidden_channels")
            self.num_layers = conf.getint("DSSL", "num_layers")
            self.dropout = conf.getfloat("DSSL", "dropout")
            self.normalize_features = conf.getboolean("DSSL", "normalize_features")
            self.seed = conf.getint("DSSL", "seed")
            self.display_step = conf.getint("DSSL", "display_step")
            self.train_prop = conf.getfloat("DSSL", "train_prop")
            self.valid_prop = conf.getfloat("DSSL", "valid_prop")
            self.batch_size = conf.getint("DSSL", "batch_size")
            self.rand_split = conf.getboolean("DSSL", "rand_split")
            self.embedding_dim = conf.getint("DSSL", "embedding_dim")
            self.neighbor_max = conf.getint("DSSL", "neighbor_max")
            self.cluster_num = conf.getint("DSSL", "cluster_num")
            self.no_bn = conf.getboolean("DSSL", "no_bn")
            self.alpha = conf.getfloat("DSSL", "alpha")
            self.gamma = conf.getfloat("DSSL", "gamma")
            self.entropy = conf.getfloat("DSSL", "entropy")
            self.tau = conf.getfloat("DSSL", "tau")
            self.encoder = conf.get("DSSL", "encoder")
            self.mlp_bool = conf.getint("DSSL", "mlp_bool")
            self.tao = conf.getfloat("DSSL", "tao")
            self.beta = conf.getfloat("DSSL", "beta")
            self.mlp_inference_bool = conf.getint("DSSL", "mlp_inference_bool")
            self.neg_alpha = conf.getint("DSSL", "neg_alpha")
            self.load_json = conf.getint("DSSL", "load_json")

        elif model == "SHGP":
            self.dataset = conf.get("SHGP", "dataset")
            self.target_type = conf.get("SHGP", "target_type")
            self.train_percent = conf.getfloat("SHGP", "train_percent")
            self.hidden_dim = re.findall(r"\[(.*?)\]", conf.get("SHGP", "hidden_dim"))[
                0
            ]
            self.hidden_dim = [int(s) for s in self.hidden_dim.split(",")]
            self.epochs = conf.getint("SHGP", "epochs")
            self.lr = conf.getfloat("SHGP", "lr")
            self.l2_coef = conf.getfloat("SHGP", "l2_coef")
            self.type_fusion = conf.get("SHGP", "type_fusion")
            self.type_att_size = conf.getint("SHGP", "type_att_size")
            self.warm_epochs = conf.getint("SHGP", "warm_epochs")
            self.compress_ratio = conf.getfloat("SHGP", "compress_ratio")
            self.cuda = conf.getint("SHGP", "cuda")

        elif model == "HGCL":
            self.lr = conf.getfloat("HGCL", "lr")
            self.batch = conf.getint("HGCL", "batch")
            self.wu1 = conf.getfloat("HGCL", "wu1")
            self.wu2 = conf.getfloat("HGCL", "wu2")
            self.wi1 = conf.getfloat("HGCL", "wi1")
            self.wi2 = conf.getfloat("HGCL", "wi2")
            self.epochs = conf.getint("HGCL", "epochs")
            self.topk = conf.getint("HGCL", "topk")
            self.hide_dim = conf.getint("HGCL", "hide_dim")
            self.reg = conf.getfloat("HGCL", "reg")
            self.metareg = conf.getfloat("HGCL", "metareg")
            self.ssl_temp = conf.getfloat("HGCL", "ssl_temp")
            self.ssl_ureg = conf.getfloat("HGCL", "ssl_ureg")
            self.ssl_ireg = conf.getfloat("HGCL", "ssl_ireg")
            self.ssl_reg = conf.getfloat("HGCL", "ssl_reg")
            self.ssl_beta = conf.getfloat("HGCL", "ssl_beta")
            self.rank = conf.getint("HGCL", "rank")
            self.Layers = conf.getint("HGCL", "Layers")

        elif self.model_name == "lightGCN":
            self.lr = conf.getfloat("lightGCN", "lr")
            self.weight_decay = conf.getfloat("lightGCN", "weight_decay")
            self.max_epoch = conf.getint("lightGCN", "max_epoch")
            self.batch_size = conf.getint("lightGCN", "batch_size")
            self.embedding_size = conf.getint("lightGCN", "embedding_size")
            self.num_layers = conf.getint("lightGCN", "num_layers")
            self.test_u_batch_size = conf.getint("lightGCN", "test_u_batch_size")
            self.topks = conf.getint("lightGCN", "topks")
            # self.alpha = conf.getfloat("lightGCN", "alpha")

        elif self.model_name == "HMPNN":
            self.lr = conf.getfloat("HMPNN", "lr")
            self.num_layers = conf.getint("HMPNN", "num_layers")
            self.hid_dim = conf.getint("HMPNN", "hid_dim")
            self.max_epoch = conf.getint("HMPNN", "max_epoch")

        if hasattr(self, "device"):
            self.device = th.device(self.device)
        elif gpu == -1:
            self.device = th.device("cpu")
        elif gpu >= 0:
            if not th.cuda.is_available():
                self.device = th.device("cpu")
                warnings.warn(
                    "cuda is unavailable, the program will use cpu instead. please set 'gpu' to -1."
                )
            else:
                self.device = th.device("cuda", int(gpu))

        if getattr(self, "use_uva", None):  # use_uva is set True
            self.use_uva = False
            warnings.warn(
                "'use_uva' is only available when using cuda. please set 'use_uva' to False."
            )

    def __repr__(self):
        return "[Config Info]\tModel: {},\tTask: {},\tDataset: {}".format(
            self.model_name, self.task, self.dataset
        )
