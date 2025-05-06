from importlib import import_module
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import OneOf, File, ListOfFloats, Folder, SubsetOf, BoolAsInt
import argparse
import os


#   我在json文件中给出的参数值都是 字符串""，所以要在Section内部规定这些参数名应该有的参数值的类型
Section(ns = 'general', desc ='General Configs').params(

    #   required = True表示必须命令行提供这个参数的参数值
    func = Param(OneOf(['pretrain', 'adapt', 'ete']), required=True),


    #   .cache用于存储  数据集
    cache_dir = Param(Folder(True), default='storage/.cache'),


    #   预训练阶段的save_dir用于保存预训练好的模型，默认值是  storage/fagcn/reconstruct
    #   下游节点的save_dir
    save_dir = Param(Folder(True), default=f'storage/tmp',required=True),


    seed = Param(int, default=777, desc='seed for general randomness'),

    #   预训练阶段不使用few_shot，下游阶段 few_shot == 1
    few_shot = Param(int, default=1, desc='few-shot node classification'),

    #   预训练阶段 reconstruct == 0.2，用来衡量重构LOSS，也就是  原始节点特征的保留程度
    #   下游阶段  reconstruct == 0.0
    reconstruct = Param(float, default=0.0),
)

Section(    'model.backbone',       'Backbone General Configs').params(
    model_type = Param(OneOf(['fagcn','gcn','bwgnn','gat']), default='fagcn', desc='backbone model to use'),
    hid_dim = Param(int, default=128),
)

Section(    'model.backbone.fagcn',         'FAGCN Model Configs').enable_if(
    lambda cfg: cfg['model.backbone.model_type'] == 'fagcn'
).params(
    num_conv_layers = Param(int, default=2),
    dropout = Param(float, default=0.2),
    epsilon = Param(float, default=0.1),
)



#   这个只在预训练阶段使用，表示预训练用到的  参数
Section('pretrain', 'Pretraining Configs').enable_if(
    lambda cfg: cfg['general.func'] == 'pretrain'
).params(
    #   预训练方法是 GraphCL
    method = Param(OneOf(['graphcl', 'simgrace']), default='graphcl'),
    learning_rate = Param(float, default=1e-2),
    weight_decay = Param(float, default=1e-5),
    #   预训练epoch == 100
    epoch = Param(int, default=100),

    #   预训练batch_size == 10，一个batch包含10个增强后的诱导子图
    batch_size = Param(int, default=10),

    noise_switch = Param(BoolAsInt(), default=False), 
    cross_link = Param(int, default=0), # cross_link=0 means no graph coordinator == isolated pretrain. cross_link>0 means includes graph coordinators == GCOPE
    
    #   没有用到
    cross_link_ablation = Param(BoolAsInt(), default=False), # false means the graph coordinator has inter-dataset edges. if true, has no inter-dataset edges.
    #   没有用到
    dynamic_edge = Param(OneOf(['internal', 'external', 'internal_external', 'similarity', 'none']), default='none'), # control inter-dataset edges
    #   没有用到
    dynamic_prune = Param(float, default=0.1),
    
    cl_init_method = Param(OneOf(['mean', 'sum', 'learnable', 'simple', 'none']), default='learnable'),

    #   用随机游走的方法构造诱导子图
    split_method = Param(OneOf(['metis', 'RandomWalk']), default='RandomWalk'),
)







#################           以下4个Section在预训练和下游阶段都不用

#   如果enable_if条件不满足，那么这个Section中的参数都是  None
Section('model.backbone.gcn', 'GCN Model Configs').enable_if(
    lambda cfg: cfg['model.backbone.model_type'] == 'gcn'
).params(
    num_conv_layers = Param(int, default=2),
    dropout = Param(float, default=0.2),
)

Section('model.backbone.gat', 'GAT Model Configs').enable_if(
    lambda cfg: cfg['model.backbone.model_type'] == 'gat'
).params(
    num_conv_layers = Param(int, default=2),
    dropout = Param(float, default=0.2),
    head = Param(int, default=8),
)

Section('model.saliency',       'Saliency Model Configs').params(
    model_type = Param(OneOf(['mlp', 'none']), default='none', desc='saliency model to use'),
)

Section('model.saliency.mlp').enable_if(
    lambda cfg: cfg['model.saliency.model_type'] == 'mlp'
).params(
    hid_dim = Param(int, default=4096),
    num_layers = Param(int, default=2),
)


######################

#   以下这两个Section只在下游阶段使用，是定义了 一个 2层MLP
Section('model.answering', 'Answering General Configs').enable_if(
    lambda cfg: cfg['general.func'] in ['adapt', 'ete']
).params(
    model_type = Param(OneOf(['mlp']), default='mlp'),
)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

Section('model.answering.mlp').enable_if(
    lambda cfg: cfg['model.answering.model_type'] == 'mlp'
).params(
    num_layers = Param(int, default=2),
)


#   预训练阶段的data.name == ["cora","citeseer","cornell"]
#   下游阶段的  data.name == "photo"
Section('data', 'Data Configs').params(
    name = Param(SubsetOf([
        'wisconsin', 'texas', 'cornell', 'chameleon', 'squirrel',
        'cora', 'citeseer', 'pubmed', 'computers', 'photo',
        ]), required=True),
    seed = Param(int, default=777, desc='seed for train/val/test split, fix this to get the same dataset'),
    node_feature_dim = Param(int, default=0, desc='0: use only structural information, >0: use node features, SVD if lower than the actual number of features else Padding'),
)


#   只在下游阶段使用（只有下游阶段是有标签（有监督）学习），表示  训练、验证、测试 的划分。
Section('data.supervised', 'Supervised Data Configs').enable_if(
    lambda cfg: cfg['general.func'] in ['adapt', 'ete']
).params(
    ratios = Param(ListOfFloats(),default='0.1,0.1,0.8', required=True, desc='train/val/test split ratios'),
)


#   这个Section是  下游阶段  用到的参数
Section('adapt', 'Adaptation Configs').enable_if(
    lambda cfg: cfg['general.func'] == 'adapt'
).params(
    repeat_times = Param(int, default=5),

    #   方法一般是微调
    method = Param(OneOf(['finetune', 'prog']), default='finetune'),

    #   预训练好的模型  存放的位置
    pretrained_file = Param(File(), required=True,default='storage/tmp/pretrained_model.pt'),

    # 下游epoch == 100
    epoch = Param(int, default=100),

    #   下游batch_size == 100
    batch_size = Param(int, default=10),
)




#   这个Section只有在 下游使用  微调  的情况才会用
Section('adapt.finetune', 'Finetune Configs').enable_if(
    lambda cfg: cfg['adapt.method'] == 'finetune'
).params(
    #   一般是  1
    backbone_tuning = Param(BoolAsInt(), default=False),

    #   用默认值false
    saliency_tuning = Param(BoolAsInt(), default=False),
    learning_rate = Param(float, default=1e-4),

    weight_decay = Param(float, default=1e-5),
)





#   这个Section只有在 下游使用提示的情况才会用
Section('adapt.prog', 'Prompt Configs').enable_if(
    lambda cfg: cfg['adapt.method'] == 'prog'
).params(
    prompt_lr = Param(float, default=1e-4),
    prompt_weight_decay = Param(float, default=1e-5),
    prompt_epoch = Param(int, default = 1),
    ans_lr = Param(float, default=1e-2),    
    ans_weight_decay = Param(float, default=1e-5),
    ans_epoch = Param(int, default = 1),
    backbone_tuning = Param(BoolAsInt(), default=False),
    saliency_tuning = Param(BoolAsInt(), default=False),    
    cross_prune = Param(float, default = 0.3),
    inner_prune = Param(float, default = 0.1),
    edge_attr_dim = Param(int, default = 0),
    prompting_target_batch_size = Param(int, default=128),
    prompting_source_batch_size = Param(int, default=2048),
    cross_link = Param(BoolAsInt(), default=True),
)

#   这个仅限于端到端模型
Section('ete', 'End-to-End Training Configs').enable_if(
    lambda cfg: cfg['general.func'] == 'ete'
).params(
    epoch = Param(int, default=100),
    batch_size = Param(int, default=10),
    learning_rate = Param(float, default=1e-4),
    weight_decay = Param(float, default=1e-5),
    repeat_times = Param(int, default=5),
)


@param('general.func')
@param('general.seed')
@param('data.name')
def run(func, seed,name):
    import sys
    sys.path.append('src')
    # Fix all randomness
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import_module(f'functional.{func}').run()


if __name__ == '__main__':

    config = get_current_config()
    #   这是一个命令行参数的解析器
    parser = argparse.ArgumentParser("All in One: Union of Homophily and Heterophily Graphs")
    config.augment_argparse(parser)     #   parser.add_argument("--config-file") ,定义了一个叫config-file的列表，初始为空，里面可以装一些json或yaml的配置文件，目的是之后从这些配置文件中读取参数。
    #   正式读取，同时读取命令行参数，以及命令行参数中 config_file指定的  配置文件中的  参数
    config.collect_argparse_args(parser)
    config.validate()

    #   把所有配置参数都导出，导出成一个config.json文件（只包含用到的参数，哪些enable_if条件不满足的参数是没有的）
    config.get_all_config(dump_path=os.path.join(config['general.save_dir'], 'config.json'))
    run()