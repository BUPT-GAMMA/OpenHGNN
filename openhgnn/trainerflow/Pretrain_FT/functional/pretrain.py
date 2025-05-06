from fastargs.decorators import param
import numpy as np
import torch
from copy import deepcopy
import os

@param('general.save_dir')
@param('data.name', 'dataset')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('pretrain.method')
@param('pretrain.noise_switch')
def run(
    save_dir,
    dataset,    #   ["cora","citeseer","cornell"]
    backbone_model,
    saliency_model,
    method,
    noise_switch,
    ):
     
    if(saliency_model == 'mlp'):    
        pass
    else:
        # load data
        # from data import get_clustered_data

        with torch.no_grad():   #   dataset是["cora","citeseer","corness"]
           
        #   data是  622个  诱导子图 （从大图中采样得来的）构成的列表
        #   gco_model.learnable_param  是  所有协调器节点的向量
        #   raw_data是  大图（包含协调器节点，和与协调器有关的边）。 raw_data.x[-3,-2,-1]就是最后3个节点向量，也就是3个协调器向量
            data, gco_model, raw_data = get_clustered_data(dataset)  #   raw_data.x包含6221个节点向量，最后3个向量  和 gco_model.learnable_param是相同的。都是代表3个协调器向量
#   这个get_clustered_data的作用：将多个graph构成一个大图，并且加入协调器节点和与协调器相连的边。之后在这个大图（包含协调器）上，通过随机游走，生成了 622个诱导子图（包含协调器）




        # init model
        
        model = get_model(
            backbone_kwargs = {
                'name': backbone_model, #   默认值：'fagcn'
                'num_features': data[0].x.size(-1), #   统一后的特征长度：默认值 100
            },
            saliency_kwargs = {
                'name': saliency_model,
                'feature_dim': data[0].x.size(-1),
            } if saliency_model != 'none' else None,    #   默认值是  None
        )                
        #   返回的  model.backbone就是一个2层的FAGCN层
        #   model.forward(input)  就是让 input  经过 backbone跑一遍

    # train
    if method == 'graphcl':            
        #   data： 622个诱导子图（包含协调器）构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]
        #   model.backbone 就是一个 2层的 FAGCN
        #   geo_model.learnable_param就是一个列表[],包含3个协调器节点向量
        #   raw_data是大图（包含协调器节点，和与协调器相连的边）
        model = graph_cl_pretrain(data=data, model=model, gco_model=gco_model, raw_data=raw_data)
    else:
        raise NotImplementedError(f'Unknown method: {method}')

    # save
    import os

    #   预训练阶段  的save_dir（"storage/fagcn/reconstruct"）  是用来保存预训练好的模型的
    torch.save(model.state_dict(), os.path.join(save_dir, 'pretrained_model.pt'))



#################################################
    
@param('pretrain.learning_rate')
@param('pretrain.weight_decay')
@param('pretrain.epoch')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('general.reconstruct')
@param('pretrain.split_method')
@param('pretrain.dynamic_edge')
def graph_cl_pretrain(
    #   显式传递实参的  4个参数，必须放在前面
    data,   #   data： 622个诱导子图构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]
    model,  #   model.backbone 就是一个 2层的 FAGCN
    gco_model,#     geo_model.learnable_param就是一个列表[],包含3个协调器节点向量
    raw_data,


    #   通过配置config隐式传递的参数必须放在后面，这几个参数的顺序可以打乱，但是参数名字一定要和config配置中的参数名对应
    learning_rate,
    weight_decay,
    epoch,  #   默认100
    cross_link, #   默认 1。每个原始图  有一个协调器，  协调器和原始图节点、协调器之间  都有边
    cl_init_method, #   默认 "learnable"，意思是用nn.Parameter初始化一个随机的长为100的向量作为协调器节点

    dynamic_edge,
    split_method,   #   默认Random_walk。用随机游走的方式构造  诱导子图
    reconstruct,    #   默认 0.2


    ):

    
    @param('pretrain.batch_size')
    def get_loaders(data, batch_size):
        #       data： 622个诱导子图（包含协调器）构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]
        import random
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        
                                #       删除节点， 边扰动  ，属性屏蔽。三者选2个，作为增强策略组合
        augs, aug_ratio = random.choices(['dropN', 'permE', 'maskN'], k=2), random.randint(1, 3) * 1.0 / 10
        #   例如，augs == [删除节点，边扰动]  ，aug_ratio == 0.2 
        view_list_1 = []
        view_list_2 = []
        for g in data:
            #   g  是  1个  诱导子图。

            #   用augs[0]（即第一种增强方式 ：删除节点） 来构造一个增强后视图
            #   例如 g 原来是 23个节点特征，72条边  。增强之后得到的view_g包含19个节点（删除20%的节点），58条边（删除与这些节点相连的边）
            view_g = graph_views(data=g, aug=augs[0], aug_ratio=aug_ratio)
            #   Pyg.Data代表一个同质图。给出节点特征(节点数 * 特征长度)  +  边 （2*边数） ，就能代表一个同质图
            view_list_1.append(Data(x=view_g.x, edge_index=view_g.edge_index))

            #   用augs[1]（即第二种增强方式 ，边扰动） 来构造一个增强后视图
            #   这里的 g 其实是 已经删除部分节点之后 的图（增强过的图）
            view_g = graph_views(data=g, aug=augs[1], aug_ratio=aug_ratio)
            view_list_2.append(Data(x=view_g.x, edge_index=view_g.edge_index))
        #   包含622个  删节点增强后  的视图，每次取10个，组成一个batch
        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=4)  
        #   包含622个  删除边增强后  的视图，每次取10个。和上面的loader1  一一对应。
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=4)  

#   测试代码
        # for graph_ in loader1:
        #     #   经过测试得出，loader_1用for循环迭代一次，取出的成员是  10个图组成的一个DataBatch
        #     pass

        # for graph_ in loader2:

        #     pass

        return loader1, loader2

    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, hidden_dim, temperature=0.5):
            super(ContrastiveLoss, self).__init__()
            #   对比任务头，包含2层 MLP。在GraphCL中，这个head似乎没有用到
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.temperature = temperature

        def forward(self, zi, zj):  #   有10个原始图，zi就是删除节点后的10个增强图的图向量，zj就是10个删除边后的增强图的图向量
            batch_size = zi.size(0) #   10
            x1_abs = zi.norm(dim=1) #   计算  每个图向量的  Frobenius 范数(即L2范数，就是根号下  每个向量元素的平方的和)
            x2_abs = zj.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
            sim_matrix = torch.exp(sim_matrix / self.temperature)
            #   得到的sim_matrix计算了  zi中每个向量  与  zj中每个向量的  余弦相似度
            

            #   这是每对正例的相似度，即zi[0]和zj[0]的相似度，zi[1]和zj[1]的相似度，zi[2]和zj[2]的相似度，.。。。。zi[9]和zj[9]的相似度，
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]

            #   以zi的10个向量为中心，每个向量 都有一个对应的LOSS，一共有10个LOSS值
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  #   分子上是1对正例相似度，分母上是9对负例的相似度
            loss = - torch.log(loss).mean()

            



            return loss

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, hidden_dim, feature_num):
            super(ReconstructionLoss, self).__init__()
            #   2层MLP
            self.decoder = torch.nn.Sequential(#   hidden_dim == 128 。 feature_dim == 100
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, feature_num),
            )

            self.loss_fn = torch.nn.MSELoss()

        def forward(self, input_features, hidden_features): #   input_features是原始节点特征（长度100），hidden_features是GNN聚合更新后的节点特征（长度128）
            reconstruction_features = self.decoder(hidden_features) #   GNN聚合后的节点特征（长度128）  ，再经过2层MLP（decoder），长度变换到100
            return self.loss_fn(input_features, reconstruction_features)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    #   初始化包括了 2层 MLP
    #   model.backbone.hidden_dim == 128  。
    loss_fn = ContrastiveLoss(model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = 100000.
    best_model = None
    if(gco_model==None):
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )                
    else:
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:   # here.  rec_loss_fn  包含 2层 MLP
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) +    #   3个协调器向量
                        list(model.parameters()) +  #   model.backbone(2层FAGCN)的参数
                        list(loss_fn.parameters()) +    #  对比LOSS的 2层MLP，GraphCL中不会用到这个head，所以这个参数不会改变
                        list(rec_loss_fn.parameters())),    # 2层MLP，这个主要是把GNN更新后的特征（长度128）  长度还原到原始长度100，这个参数会用到。
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm   #   用于打印进度条
    # from data.contrastive import update_graph_list_param    
    loss_metric = MeanMetric()

    #   默认epoch = 100，检查代码能否运行时，可以设置为2
    epoch = 2
    for e in range(epoch):
        
        #   在每个epoch中，last_updated_data （622个诱导子图，包含协调器节点）  和   gco_model.last_param（3个协调器  在本回合开始的初始值）  都是保持不变的
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='RandomWalk'):
                last_updated_data = deepcopy(data)  #   data： 622个诱导子图构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]

            loaders = get_loaders(data) #   data： 622个诱导子图构成的列表


        elif(e==0):
            loaders = get_loaders(data)

        pbar = tqdm(zip(*loaders), total=len(loaders[0]), ncols=100, desc=f'Epoch {e}, Loss: inf')
                
        for batch1, batch2 in pbar:
            #   batch1包含10个  删节点增强后的图  ，batch2包含10个  删除边增强后的图
            #   上一个batch1，batch2 会在最后更新 协调器，下面这个操作就是把  更新后的协调器向量  赋值  给 这个batch中的协调器。（相当于让这一个batch的协调器也得到更新）
            if(gco_model!=None):    #   以下这个操作，是把batch中的协调器向量  从不可改变的固定向量  变成  可学习的向量（协调器最新值）。
                batch1 = gco_model(batch1)  #   一个batch1包含10个图，batch1.batch取值为0~9，标明了属于10张图中的哪一个
                batch2 = gco_model(batch2)    

            optimizer.zero_grad()

            if(reconstruct==0.0):
                zi, zj = model(batch1.to(device)), model(batch2.to(device))
                loss = loss_fn(zi, zj)
            else:               
                zi, hi = model(batch1.to(device))   # zi是10个ReadOut操作之后的图向量，  hi是 batch1中的10个图的  所有节点  经过GNN聚合后的节点特征，
                zj, hj = model(batch2.to(device))
                                                    
                                                    
                                                    #   batch1.x是 没经过GNN的原始特征， hi是经过GNN之后的节点特征
                loss = loss_fn(zi, zj) + reconstruct*(rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch2.x, hj))
                #   loss_fn是对比LOSS，目的是希望 正例之间相似度高，负例之间相似度低，这样对比LOSS就会更小
                #   rec_loss_fn是重构LOSS，目的是希望  GNN更新后的节点特征  和  原始节点特征  的差距更小（保留更多的原始信息），这样LOSS会更小。
                
            loss.backward()     #   每个batch都有一个LOSS
            optimizer.step()    #   更新GNN参数（model），更新协调器向量（gco_model)，更新rec_loss的decoder
            
            loss_metric.update(loss.item(), batch1.size(0)) #   loss_metric会把 每个batch的loss都集合起来，最后loss_metric.compute()会返回一个平均值
            pbar.set_description(f'Epoch {e}, Loss {loss_metric.compute():.4f}', refresh=True)





        if(gco_model!=None):
            #   这个阶段 last_updated_data （622个诱导子图）中的协调器   和  gco_model.last_param是一样的
            #   以下这个函数的作用是，更新 622个诱导子图中  协调器的值，  返回的data（包含622个诱导子图）是  协调器更新后的  诱导子图
            data  = update_graph_list_param(last_updated_data, gco_model)   

            #   把last_pram这三个协调器  也更新成最新值
            gco_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()   #   这是本epoch内，所有batch的loss的平均值。作为本轮（本epoch）的LOSS
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model






################    以下4个函数，作用是生成增强后的视图，用于预训练对比学习



#   data是一个 pyg格式的图，对它做删除部分节点的操作
def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
    except:
        data = data

    return data

    #   data  是 一个图。这个是边扰动的增强策略，根据代码来看，就是随机删除  20%  的边
def permute_edges(data, aug_ratio):
    """
    only change edge_index, all the other keys unchanged and consistent
    """
    _, edge_num = data.edge_index.size()    #   图data  中  边的总数（例如58）
    permute_num = int(edge_num * aug_ratio) #   需要删除的边的数量（例如11）

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    data.edge_index = data.edge_index[:, idx_delete]

    return data     #   删除一部分边之后，还剩47条边

def mask_nodes(data, aug_ratio):
    node_num = data.x.size(0)
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token.clone().detach()

    return data    

def graph_views(data, aug, aug_ratio=0.1):

    if aug == 'dropN':
        data = drop_nodes(data, aug_ratio)
    elif aug == 'permE':
        data = permute_edges(data, aug_ratio)
    elif aug == 'maskN':
        data = mask_nodes(data, aug_ratio)
    else:
        raise NotImplementedError(f'{aug} not implemented')
    
    return data


#################################
#   以下两个函数，第一个函数是用于构造大图，生成622个诱导子图
#   第二个函数：用于更新诱导子图中的协调器


import time

from torch_geometric.data import Batch

@param('general.cache_dir')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('pretrain.cross_link_ablation')
@param('pretrain.dynamic_edge')
@param('pretrain.dynamic_prune')
@param('pretrain.split_method')
def get_clustered_data(dataset, cache_dir, cross_link, cl_init_method='learnable', cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0,split_method='RandomWalk'):
    #   这个get_clustered_data的作用：将多个graph构成一个大图，并且加入协调器节点和与协调器相连的边。之后在这个大图（包含协调器）上，通过随机游走，生成了 622个诱导子图（包含协调器）

    #################              实验代码，
    #   测试结论：iterate_datasets（dataset)函数，会返回一个可迭代的generator，这个generator有2种访问方式:
    #   第一种：用list()把它转换成列表来访问。第二种：在for循环中用in generator来依次访问每个成员
    #   实际情况：这个generator包含了多个data，每个 data  都是一个  PyG.Data。
    #   PyG中的cora数据集：2708个节点（特征长度1433），10556条边，train集 140个节点，val集500个节点，test集1000个节点。如果train_mask/val_mask/test_mask的划分是和节点数量对齐，那么这就是一个节点分类问题，y就是节点标签
    # ite_result = list(iterate_datasets(dataset))
    # data_list_test = []
    # for data in ite_result:

    #     #   preprocess(data)效果是：把data（一个PyG图数据集）中节点特征长度  变到规定的统一长度。
    #     #   然后把train_mask等三个mask删除掉，因为预训练阶段不需要label
    #     processed_data = preprocess(data)
    #     data_list_test.append(processed_data)

########################

    #   dataset是["cora","citeseer"]。  iterate_datasets生成一个generator(一种迭代器，粗略理解为一个列表[]就行)，包含多个数据集
    # from .utils import preprocess, iterate_datasets
    #   iterate_datasets(dataset)会返回一个iterator迭代器（可以粗略认为是一个列表[]），其中包含多个数据集（每个数据集都是一个Pyg.Data )
    #   preprocess(data)效果是：  把一个Pyg.Data中的 train_mask等删除（预训练不需要label，自然也不用划分训练集），并且把节点特征长度统一到  100



    #   根据上面的实验代码得出结论：以下这个返回的data_list，包含 n 个数据集（data）
    #   每个data的节点特征长度统一到固定长度，并且删除train_mask等，保留了label，但是预训练用不到label
    data_list = [preprocess(data) for data in iterate_datasets(dataset)]
    #   最终返回的data_list，就是一个[]，包含多个Pyg.Data
    #   data_list[0]  是  Cora 数据集 , 节点 ： [2708,100]，边 ：[2,10556]， 还有 y（label）:节点的类别标签
    #   data_list[1]  是  Computers 数据集 , 节点 ： [13752,100]，边 ：[2,491722]
    #   data_list[1]  是  Citeseer数据集，节点特征：[3327,100] ,  边：[2,9104]
    #   data_list[2]  是  Cornell 数据集 , 节点 ： [183,100]，边 ：[2,298]
########################################################################################################
    from torch_geometric.data import Batch  #   
    data = Batch.from_data_list(data_list)  #   把 多个graph  合并  成一个大图（各个子图之间没有连接）。（具体：把多个pyg.Data  构造成  一个pyg.Batch对象)
    #   构造后的data  是一个  pyg.batch(大图)，节点特征 x :[16460,100] , edge_index（边） 是：[2 , 502278] 
    #   此时的节点数等于3个小图节点个数之和，边数等于3个小图的边数量之和。（可以看出，此时这个大图还不包含协调器，也不存在和协调器连接的边）
    #   data.batch是一个 [0,0,......,1,1.....，2,2....]  :  6218个元素，batch中是0表示cora的节点，1表示citeseer节点，2表示cornell节点
    #   这个Batch.from_data_list()操作，是把多个图构造成一个大图（不含协调器）：data


    from copy import deepcopy
    data_for_similarity_computation = deepcopy(data)
    print(f'Isolated graphs have total {data.num_nodes} nodes, each dataset added {cross_link} graph coordinators')


    gco_model = None

    if(cross_link > 0):
        num_graphs = data.num_graphs
        graph_node_indices = []

        for graph_index in range(num_graphs):
            node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_node_indices.append(node_indices)
        #   graph_node_indices[0] 是 tensor :[0到2707]。  即第1个小图Cora的   大图节点ID
        #   graph_node_indices[1] 是 tensor :[2708到16459]  。即第2个小图Computers的   大图节点ID
            

        new_index_list = [i for i in range(num_graphs)]*cross_link  #   只有2个图的时候，new_index_list == [0,1]
        
        if(cl_init_method == 'mean'):
            pass
        elif(cl_init_method == 'learnable'):
            # from model.graph_coordinator import GraphCoordinator     

            #   定义协调器节点的初始化向量，geo_model.learnable_param就是一个列表[]， 包含了 k个长度为100的 一维向量(nn.Parameter)
            #   k个原始小图，每个图有一个长度为100的协调器节点
            #   因为设置了seed，所以每次跑代码，生成的协调器向量初始值都是一样的
            gco_model = GraphCoordinator(data.num_node_features,len(new_index_list))

            #   data是一个 Pyg.Batch（大图）， data.x包含了3个小图的全部节点特征。
            #   把3个协调器节点特征 加入到 原有的data.x当中。
            
            data.x = gco_model.add_learnable_features_with_no_grad(data.x)
            #   处理后的  data.x  是  6221 * 100的 节点特征。（所有 原始小图 节点特征 + k个协调器节点特征）
            #   Cora,Citeseer,Corenell三个原始图  节点总数加起来  6218. （不包含协调器），加上3个协调器共有6221个节点
            #   这个操作就是，把3个协调器加入到大图节点特征中


        #   原来的data.batch是一个 [0,0,......,1,1.....,2,2]。6218个元素，
        #   前2708个是0（表示cora的节点），后3327个是1（表示computers的节点），最后183个是2
        #   这里的data.batch 指明了节点属于哪个原始图。0表示cora。  1表示citeseer,    2表示cornell
        #   可以得出结论，pyg中的data.batch用来指示 每一个节点属于哪一个图。这段代码用来表示节点 属于（cora,citeseer,cornell)哪一个图
        #   之后pretrain代码中，用data.batch来表示  每一个节点属于  一个batch中10个图  中的那一个
        
        data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)
        #   处理之后的data.batch在最后加了个[0,1,2]，代表： cora图的协调器节点， citeseer图的协调器节点,cornell图的协调器节点


        #   把与协调器有关的边 ，加入到大图data中
        if(dynamic_edge=='none'):

        #   给大图data加入新的边：每个协调器节点  和  对应的小图原始节点都有  来、回2条边
            if(cross_link==1):   
                #   new_index_list = [0,1,2]
                for node_graph_index in new_index_list: #   node_graph_index 为0，表示第一个图Cora
                    #   data.batch == 0 ，表示属于Cora的节点。 ==1 表示Citeseer的节点。 ==2表示Corenell的节点

                    #   node_indices_corresponding_graph 是  Cora全部节点 + 对应的协调器节点  的大图ID。最后一项是协调器节点ID
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)  

                    #     Cora对应的协调器节点的  大图ID
                    new_node_index = node_indices_corresponding_graph[-1]   

                        #   new_node_index  是当前协调器节点ID， node_indices_corresponding_graph[:-1] 是当前原始图的原始节点ID
            #  让Cora的协调器节点 和 Cora每个原始节点，都连一条边 ，        [16460]         ,       [0,1,2....,2707]
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])

                    #   把上面的新边加入大图中
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                
                    #   把上面的边  的  反向边  ，也加入大图中。  也就是说，对于每个小图，图中原始节点和 协调器节点之间，有 来、回两条边
                    new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            else:
                for node_graph_index in new_index_list[:num_graphs]:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index_list = node_indices_corresponding_graph[-1*cross_link:]
                    for new_node_index in new_node_index_list:
                        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1*cross_link])
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                        
                        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1*cross_link], torch.tensor([new_node_index]))
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)                                    
            
        #   给大图加边：任意2个协调器之间都有来、回2条边。
            if(cross_link_ablation==False):

                #   all_added_node_index  就是  所有协调器节点的大图ID：[16460,16461]
                all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]

#               让每个当前协调器  都和其他协调器发出一条边。这样for循环结束后，每2个协调器直接都有来回2条边。(注意，最后一个协调器，没有向其他协调器发出边)
                for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
                    #   new_node_index是当前的 协调器节点
                    
                    #   other_added_node_index_list是  所有 其他协调器
                    other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index]


                    #   new_edges == [16460,16461]，是2个协调器之间的边。（具体是让当前协调器  发出一条指向其他协调器的边）
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)



    print(f'Unified graph has {data.num_nodes} nodes, each graph includes {cross_link} graph coordinators')

    #   这个data和raw_data就是  大图（包含3个原始图，以及协调器节点，和与协调器相连的边）
    raw_data = deepcopy(data)


#   此时的 data是 1个大图：包含cora,citeseer,cornell三个小图。  一共6221个节点（包含3个协调器），32398条边（包括原始边、协调器和原始图节点的边、协调器之间的边）
###############         上面有个问题，那就是为什么最后一个协调器不向其他协调器发射边
    


#########################
    if(split_method=='RandomWalk'):
        from torch_cluster import random_walk
        split_ratio = 0.1   #   大图节点总数 中  选择 10 %  的节点
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)   #   得到大图所有节点ID，乱序的
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)   # 大图节点总数 中  选择 10 %  
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk] #   这就是从大图所有节点中（包括协调器）  选出的  10%  的随机节点
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], 
                                start=random_node_list,     # 这些节点作为  起点
                                walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   # walk是一条 长度为 31  的节点路径（游走路径）
            subgraph_nodes = torch.unique(walk) #  subgraph_nodes  是这条路径上的  所有节点ID(  例如有23个节点ID    )
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            #   data是 1个 大图（包含协调器  和 与协调器有关的边）
            subgraph_data = data.subgraph(subgraph_nodes)   # 用这些节点ID ，从大图中构造一个【诱导子图】，诱导子图中的节点是：上述节点，边是：两端点都是上述节点的边

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

        #   graph_list是  622个  诱导子图（包含协调器节点）
        #   gco_model.learnable_param就是  所有协调器节点的向量
        #   raw_data是  大图（包含协调器节点，和与协调器有关的边）
        return graph_list, gco_model, raw_data  #   raw_data.x包含6221个节点向量，最后3个向量  和 gco_model.learnable_param是相同的。都是代表3个协调器向量


def update_graph_list_param(graph_list, gco_model):
    #   graph_list是622个诱导子图，  gco_model是3个协调器，这个函数的作用是  更新所有诱导子图中的协调器的值。（gco_model.last_param这是这一轮epoch中固定不变的协调器的值）
    #   而graph_list
    count = 0
    for graph_index, graph in enumerate(graph_list):    #   graph_list是 622个诱导子图（可能包含协调器节点）

        for index, param_value in enumerate(gco_model.last_updated_param):

            match_info = torch.where(graph.x==param_value)

            if(match_info[0].shape[0]!=0):
                target_node_indice = match_info[0].unique()[-1].item()  #   target_node_indice == 22,就表示 ，这个子图中的 22号节点 就是协调器
                graph.x[target_node_indice] = gco_model.learnable_param[index].data #   更新这个协调器的值
                count+=1
    updated_graph_list = graph_list
    return updated_graph_list    



######################  以下4个函数，主要是用于生成初始的图数据集，并且把特征长度统一


from copy import deepcopy

from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork
from torch_geometric.data import Data
from torch_geometric.utils import degree, add_self_loops
from fastargs.decorators import param
import math

def x_padding(data, out_dim):
    
    assert data.x.size(-1) <= out_dim
    
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros((data.x.size(0), incremental_dimension), dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, zero_features], dim=-1)

    return data

def x_svd(data, out_dim):
    
    assert data.x.size(-1) >= out_dim

    reduction = SVDFeatureReduction(out_dim)
    return reduction(data)


#   这个函数会返回一个 迭代器，包含多个 Pyg.Data（比如cora,citeseer等）
@param('general.cache_dir')
def iterate_datasets(data_names, cache_dir):
    #   预训练阶段 dataset是 ["cora","citeseer","cornell"]
    #   下游阶段  dataset 是 "phote"
    if isinstance(data_names, str):
        data_names = [data_names]
    #   dataset是["cora","citeseer","cornell"]
    for data_name in data_names:
        if data_name in ['cora', 'citeseer', 'pubmed']:
            data = Planetoid(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['wisconsin', 'texas', 'cornell']:
            data = WebKB(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['computers', 'photo']:
            data = Amazon(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['chameleon', 'squirrel']:
            preProcDs = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=False)
            data = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=True)._data
            data.edge_index = preProcDs[0].edge_index
        else:
            raise ValueError(f'Unknown dataset: {data_name}')
        
        assert isinstance(data, (Data, dict)), f'Unknown data type: {type(data)}'
        #   yield 作用：例如generator_01 = iterate_datasets(data_names)。    会返回一个generator（一种特殊的迭代器iterator），其中包含多个成员，每个data就是一个成员。  
        #   每次调用yield就会产生一个data。  每次generator.__next__()方法就会  调用yield（产生）  并  返回一个成员。
        #   或者 for _ in generator：  这样每次循环访问的一个  _ ，效果就是调用yield  ，产生下一个成员，然后把这个成员返回。
        #   也可以用 list(generator)  把迭代器变成列表，这样可以用list[0]这样按下标访问。
        yield data if isinstance(data, Data) else Data(**data)


# including projection operation, SVD。  这个函数是  把给定的Pyg.Data（一个graph），删除mask，并且节点特征维度统一到100
@param('data.node_feature_dim')
def preprocess(data, node_feature_dim):
    #   比如data是  photo数据集 ，节点特征[7650,745]，  edge_index:[2,238162]
    #   删除train_mask等，因为预训练不需要label，自然也就不需要用train_mask来划分训练节点等，预训练是自监督
    if hasattr(data, 'train_mask'):
        del data.train_mask
    if hasattr(data, 'val_mask'):
        del data.val_mask
    if hasattr(data, 'test_mask'):
        del data.test_mask

    if node_feature_dim <= 0:
        edge_index_with_loops = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.x = degree(edge_index_with_loops[1]).reshape((-1,1))
    
    else:   #   把节点特征长度  统一到 node_feature_dim （默认值100）
        # import pdb
        # pdb.set_trace()        
        if data.x.size(-1) > node_feature_dim:
            data = x_svd(data, node_feature_dim)
        elif data.x.size(-1) < node_feature_dim:
            data = x_padding(data, node_feature_dim)
        else:
            pass
    
    return data



###################     以下两个函数，用于返回初始化好的model


from importlib import import_module
import torch

#   Model.backbone就是主要模型（一个2层的FAGCN）
class Model(torch.nn.Module):
    def __init__(
                self,
                backbone,
                answering = torch.nn.Identity(),
                saliency = torch.nn.Identity(),
            ):
        super().__init__()
        self.backbone = backbone    #   fagcn 2层模型
        self.answering = answering  #   空操作层
        self.saliency = saliency    #   空操作层

    def forward(self, data):    #   让 data  经过  backbone（2层FAGCN)跑一次

        data.x = self.saliency((data.x))    #   空操作


        #   在预训练阶段，model.forward会让节点特征会只经过2层GNN的变换(backbone)
        #   在下游阶段，model.forward会让节点特征，先经过2层GNN变换，再经过answering（任务头）的变换
        return self.answering(self.backbone(data))  #   经过多层fagcn，得到输出结果
        #   注意：backbone(GNN).forward 会先通过GNN聚合更新节点向量，然后再ReadOut得到图向量                                    

def get_model(
        backbone_kwargs,    #  {"name" : "fagcn", "num_features" : 100  } 
        answering_kwargs = None,    #   仅在  下游微调阶段 使用
        saliency_kwargs = None, #   预训练和微调阶段都用不到
    ):
    
    backbone_kwargs.pop('name')
                #               model.backbone.fagcn文件中的  FAGCN类： 默认包含 2层  FAGCN层
    backbone = get_model_FAGCN(**backbone_kwargs)

    #   torch.nn.Identity()是一个空操作层，它的forward(input) 不会对input做出任何改变

    #   仅在微调阶段使用answering（2层MLP），本质上是下游任务头。  
    #   在下游阶段，初始节点特征（100）会首先经过backbone（2层GNN）的变换，变到长度128
    #   然后会经过 answering的变换， 变成长度为8 的  预测类别概率
    
    if answering_kwargs is None:
        answering = torch.nn.Identity()
    else:
        answering_kwargs.pop('name')
        answering = get_model_MLP(**answering_kwargs)
    
    
    #   saliency在预训练和  微调阶段  都不会用到
    saliency = torch.nn.Identity() 

    return Model(backbone, answering, saliency)




import torch
from torch_geometric.nn import global_add_pool, FAConv
from fastargs.decorators import param

class FAGCN(torch.nn.Module):

    def __init__(self, num_features, hidden, num_conv_layers, dropout, epsilon):
        super(FAGCN, self).__init__()
        self.global_pool = global_add_pool
        self.eps = epsilon              #   默认epsilon  为 0.1
        self.layer_num = num_conv_layers    #   默认  2 层
        self.dropout = dropout          #   默认dropout 为 0.2
        self.hidden_dim = hidden        #   默认128
        #   self.layers包含  多层的 FAConv
        self.layers = torch.nn.ModuleList() 
        for _ in range(self.layer_num):
            self.layers.append(FAConv(hidden, epsilon, dropout))

        self.t1 = torch.nn.Linear(num_features, hidden)
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    @param('general.reconstruct')
    def forward(self, data, reconstruct):   #   这里data是 一个batch（包含10个增强后的图）

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch #   edge_index是 10个图的 所有边，batch用来指明每个节点属于10个图中的哪个图

        h = torch.dropout(x, p=self.dropout, train=self.training)
        h = torch.relu(self.t1(h))  #   对初始节点特征（长度100）做一个线性变换，变成长度128
        h = torch.dropout(h, p=self.dropout, train=self.training)
        raw = h #   raw是初始特征经过一层Linear层变换后的  长度为128的特征（相当于 没经过GNN的初始特征）


        for i in range(self.layer_num): #   这是经过 多层GNN聚合，得到更新后的  节点特征 h 
            h = self.layers[i](h, raw, edge_index)
        h = self.t2(h)

        #   这就是ReadOut操作，这里有10个图，得到每个图的  图向量。
        graph_emb = self.global_pool(h, batch)

        if(reconstruct==0.0):   #   reconstruct == 0，这是下游微调阶段，这一阶段只需要把子图向量返回
            return graph_emb
        else:   # 返回多层GNN聚合后的节点特征 h  ， 以及ReadOut操作之后的 图向量
            return graph_emb, h

from fastargs.decorators import param

@param('model.backbone.hid_dim')
@param('model.backbone.fagcn.num_conv_layers')
@param('model.backbone.fagcn.dropout')
@param('model.backbone.fagcn.epsilon')
def get_model_FAGCN(num_features, hid_dim, num_conv_layers, dropout, epsilon):
    return FAGCN(num_features, hid_dim, num_conv_layers, dropout, epsilon)



class MLPAnswering(torch.nn.Module):
    def __init__(self, hid_dim, num_class, answering_layer_num):
        super().__init__()
        self.answering_layer_num = answering_layer_num  #   2，表示2层MLP
        self.num_class = num_class  #   8，表示8分类的节点分类问题（下游阶段）
        
        self.answering = torch.nn.ModuleList()
        self.bns_answer = torch.nn.ModuleList()

        for i in range(answering_layer_num-1):
            self.bns_answer.append(torch.nn.BatchNorm1d(hid_dim))
            self.answering.append(torch.nn.Linear(hid_dim,hid_dim)) #   从hid_dim  到  hid_dim的  线性变换层
        
        self.bn_hid_answer = torch.nn.BatchNorm1d(hid_dim)
        self.final_answer = torch.nn.Linear(hid_dim, num_class)     #   这是从 hid_dim  到  num_class的最后输出

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)        
    
    def forward(self, x):
        
        for i, lin in enumerate(self.answering):
            x = self.bns_answer[i](x)
            x = torch.relu(lin(x))  #   做一个 hid -> hid的变换
            
        x = self.bn_hid_answer(x)
        x = self.final_answer(x)    #   做一次 hid -> num_class的变换（8分类问题）
        prediction = torch.log_softmax(x, dim=-1)
        return prediction
    

@param('model.backbone.hid_dim')
@param('model.answering.mlp.num_layers')
def get_model_MLP(hid_dim, num_class, num_layers):
    return MLPAnswering(hid_dim, num_class, num_layers)



##############  以下这个是协调器的相关功能


import torch
import torch.nn as nn
from copy import deepcopy

class GraphCoordinator(nn.Module):
    def __init__(self, num_node_features, num_graph_coordinators):
        super(GraphCoordinator, self).__init__()
        #   如果有k个图，那么self.learnable_param就是一个  列表[  ]，其中包含  k个  长度为num_node_features 的向量（就是协调器节点），节点是正态分布初始化的随机值
        self.learnable_param = nn.ParameterList(    
            [nn.Parameter(torch.randn(num_node_features)) for _ in range(num_graph_coordinators)]   
            )
        self.last_updated_param = [deepcopy(param.data) for param in self.learnable_param]
    

    #   把所有小图的全部节点特征，和协调器节点特征，整合到一起
    def add_learnable_features_with_no_grad(self, original_node_features):
        #   3个协调器节点向量
        graph_coordinator_features = torch.cat([p.data.reshape((1,-1)) for p in self.learnable_param], dim=0)

        #   把协调器节点向量 和  所有原始小图节点向量  都拼到一起。 返回的updated_feats 是大图 节点向量（包含协调器）
        updated_features = torch.cat([original_node_features, graph_coordinator_features], dim=0)
        return updated_features



    #   这个函数传入一个batch（例如包含10个图），效果就是把图中的节点特征中的 协调器向量（一开始是不可学习的固定向量）变成  可学习的向量
    def forward(self, batch_with_no_grad_node_features):
                        #   这里的batch不带梯度，不带梯度意思就是向量固定（不可改变），有梯度就代表着可以改变（可学习）
        count = 0
        graph_index_list = [x for x in set(batch_with_no_grad_node_features.batch.tolist())]    #   graph_index_list == 0~9，表示这一个batch中的 9个图
        for graph_index in graph_index_list:
            #   例如graph_index == 0，表示这一个batch中的第 0个图。 node_indices_corre_graph == [0~18]，表示这个图上的 节点ID
            node_indices_corresponding_graph = (batch_with_no_grad_node_features.batch == graph_index).nonzero(as_tuple=False).view(-1)
            for node_indice in reversed(node_indices_corresponding_graph):
                #   例如node_indice == 18 时，表示的是这第0个图的19个节点中的  18号节点
                #   self.last_updated_param  是  3个  长为100 的协调器向量
                for index, param_value in enumerate(self.last_updated_param):
                    if(torch.equal(batch_with_no_grad_node_features.x[node_indice], param_value)):  #   比较 这个节点特征  和  协调器向量  是否相等
                        #     如果相等的话，就说明当前节点（例如18号节点） 恰好就是  协调器（第1个协调器，即cora的协调器）。
                        #   如果相等，那么就把 这个协调器向量   赋值给  当前节点向量  。
                        #   虽然这两个值原来就相等，但是之前比较的时候，last_updated_param是 不可学习的固定向量。现在的learnable_param是可学习的向量。
                        batch_with_no_grad_node_features.x[node_indice] = self.learnable_param[index]
                        count+=1
        batch_with_learnable_param = batch_with_no_grad_node_features
        return batch_with_learnable_param

    def update_last_params(self):
    #   把更新后的协调器   赋值给  last_param这3个协调器（这3个协调器在每个epoch过程中保持不变，只会在本轮epoch的最后得到更新）
        self.last_updated_param = [deepcopy(param.data) for param in self.learnable_param]

