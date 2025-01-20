import copy
from fastargs.decorators import param
import torch


@param('data.name', 'dataset')
@param('adapt.batch_size')
@param('data.supervised.ratios')
@param('adapt.method')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('model.answering.model_type', 'answering_model')
@param('adapt.pretrained_file')
@param('general.save_dir')
@param('adapt.repeat_times')
def run(
    dataset,    #   "photo"，下游任务阶段，只用1个graph
    batch_size, # 100
    ratios, #       0.1， 0.1 ， 0.8 这是训练集、验证集、测试集的划分
    method, #   一般是finetune
    backbone_model, #   fagcn
    saliency_model, #   none
    answering_model,    #   mlp，是一个2层MLP
    pretrained_file,    #   预训练好的 GNN模型的 存放位置
    save_dir,
    repeat_times,   #   5
    ):
    
    # load data
    
    
    from torch_geometric.loader import DataLoader

    #   get_supervised_data("photo",ratios = [0.1,0.1,0.8])   的作用
    #   首先下载到photo数据集（一共7650个节点），给每个节点都以它为中心构造一个诱导子图，一共7650个诱导子图
    #   然后，训练集中：每个类别，有一个对应的诱导子图（子图的中心节点就属于该类别，这是一个节点分类问题）
    #   剩下的7642个子图，按照1比9的比例，分配给val_set（765个子图）和test_set（6877个子图）
    datasets, num_classes = get_supervised_data(dataset[0], ratios=ratios)


    #           batch_size == 100。 datasets包括 train：8个图，val:765个图，test:6877个图
    loaders = { k: DataLoader(v, batch_size=batch_size, shuffle=True, num_workers=4) for k, v in datasets.items() }
    #   k   :  v
    #   "train" : 8个图，
    #   "val" : 765个图，每个batch包含100个图，
    #   "test" : 6877个图，每个batch包含100个图


    # init model
    from .pretrain import get_model
    model = get_model(
        backbone_kwargs = {
            'name': backbone_model,
            'num_features': datasets['train'][0].x.size(-1),
        },
        answering_kwargs = {
            'name': answering_model,    #   MLP
            'num_class': num_classes,   #   8，表示8分类问题，也是MLP最后输出的大小
        },
        saliency_kwargs = {
            'name': saliency_model,
            'feature_dim': datasets['train'][0].x.size(-1),
        } if saliency_model != 'none' else None,
    )


    #   取出预训练好的模型
    model.load_state_dict(torch.load(pretrained_file,map_location=lambda storage, loc: storage.cuda(0)), strict=False)

    # train
    all_results = []
    
    #   默认repeat_times = 5
    repeat_times = 1

    for _ in range(repeat_times):   #   一共做5次微调，这5次微调彼此之间完全独立，没有任何关联。
        if method == 'finetune':
            results = finetune(loaders, model)      #   返回的results包含 在测试集上的  ACC。代表这一次微调的结果
        elif method == 'prog':
            pass
        else:
            raise NotImplementedError(f'Unknown method: {method}')
        
        results.pop('model')
        all_results.append(results)        #    1次finetune其实就是完整的 预训练—微调了，返回的results可以理解为实验结果（ACC）。
    ######                                  #   只不过由于做实验，实验结果有偶然性，所以要多做几次取一个平均值
    # 打印 acc, auroc, f1   的平均值和标准差
    import numpy as np      #   all_results包含了 5次独立实验的  全部结果（ACC）
    for k in all_results[0].keys(): #   把5次独立的实验结果，求一下平均值和标准差
        print(f'{k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}')
        
    import os

    if(method!='prog'): #   微调阶段的  save_dir : 'storage/fagcn/balanced_few_shot_fine_tune_backbone_with_rec'  ，用于保存下游测试结果
        with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys(): #   把5次实验的平均值和标准差 输出到文件中
                f.write(method+f'FT on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')
    else:
        pass


@param('adapt.finetune.backbone_tuning')
@param('adapt.finetune.saliency_tuning')
@param('adapt.finetune.learning_rate')
@param('adapt.finetune.weight_decay')
@param('adapt.epoch')
def finetune(
        loaders,
        model,
        backbone_tuning,
        saliency_tuning,
        learning_rate,
        weight_decay,
        epoch,
        ):

    model.backbone.requires_grad_(backbone_tuning)  #   表示backbone（2层GNN）是会在下游阶段不断更新的
    model.saliency.requires_grad_(saliency_tuning)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  #   下游阶段要更新的是  GNN  和  2层MLP（answering)
        lr = learning_rate,
        weight_decay = weight_decay,
        )

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    best_acc = 0.
    best_model = None


    #   默认epoch = 100
    epoch = 2

    for e in range(epoch):  
        #   每一个epoch（每一轮），都包括  训练  和  验证。训练就是更新模型，验证就是计算一下模型的预测效果
        #   训练过程  会 更新模型
        #   如果在验证集上的ACC优于best_acc，那么就用这一轮更新后的model来作为最优模型



        model.train()

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        #       这一轮的训练

        #   首先是 train_set，一共8个图（加起来220个节点），因为一个batch包含100个图，所以训练集只有一个batch
        pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} Training, Loss: inf')

        for batch in pbar:
            optimizer.zero_grad()   #   这个循环只会执行一次，因为训练集一共就一个batch（包含8个图，加起来220个节点）
            batch = batch.to(device)
            pred = model(batch) #   这里的pred是每个图 的预测值（长度为8的向量，表示8个类别的预测值，可以是负数）
            loss = torch.nn.functional.cross_entropy(pred, batch.y) #   每个图的预测值向量  先经过softmax得到预测概率（8个类别的概率的和为1），然后取正确类别的概率的 -ln值，这样正确类别的预测概率越大，其他类别的预测概率越小，LOSS值就越小
            loss.backward() #   LOSS对 参数求梯度
            optimizer.step()    #   更新参数，更新GNN参数（backbone)  和  下游任务头（2层MLP）

            loss_metric.update(loss.detach(), batch.size(0))
            pbar.set_description(f'Epoch {e} Training Loss: {loss_metric.compute():.4f}', refresh=True)
        pbar.close()


        ###     这一轮的验证


        model.eval()
        
        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():   #   有梯度 就代表要更新模型参数。没有梯度就代表不会更新模型参数
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch).argmax(dim=-1)  #   预测类别

                acc_metric.update(pred, batch.y)    #   预测类别pred  和  真实类别标签y  ，计算出ACC（预测准确率）
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(batch), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_model = deepcopy(model)
    







    #   上述有100轮，每一轮都会有一个ACC，我们选取ACC最高的那一轮（表示那一轮的模型预测效果最好，就是在验证集上效果最好），用那一轮的模型来作为最优模型
    model = best_model if best_model is not None else model


#####       所有100轮跑完之后，选取验证集上效果最好的一轮，用这一轮的模型来作为最优模型


###         
    # test
    model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)#   预测类别

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(batch), batch.y)  #   以下输出的是  每一个batch的acc。
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()
    
    return {
        'acc': acc_metric.compute().item(), #   返回模型在测试集上的  ACC
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }









##############      以下两个函数  是构造下游任务数据集的过程



import torch

#   下游节点，传入的data是  一个Pyg.Data对象，表示的是photo数据集
#   对于data中的每一个节点（比如photo有7650个节点），都生成一个以该节点为中心的诱导子图，返回值就是这些诱导子图
def induced_graphs(data, smallest_size=10, largest_size=30):

    from torch_geometric.utils import subgraph, k_hop_subgraph
    from torch_geometric.data import Data
    import numpy as np

    induced_graph_list = []
    total_node_num = data.x.size(0) #   data的节点总数

    for index in range(data.x.size(0)): 
        # index表示每一个  节点ID

        current_label = data.y[index].item()    #   当前index节点 的 label

        current_hop = 2 #   跳数

        #   k_hop_subgraph是 Pyg提供的寻找k跳子图的函数。返回的subset是 子图中包含的节点ID（整图ID）
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index)
            
        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label)) 
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]    #   从subset中随机选择 29个节点ID（整图节点ID）
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))    # 把这29个节点ID和  当前节点ID（index）拼到一起
        
    #   经过上面的操作后，subset就是  以index节点为中心的2跳子图  中的 所有节点ID（整图节点ID）

    #   这个subgraph是生成诱导子图的操作，给出subset（一些节点ID），返回一个诱导子图
    #   诱导子图中，节点是：subset这些节点，边是：这些节点之间的边
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)   #   返回的sub_edge_index就是诱导子图中的边（边上两端点的ID是子图ID）
    #   x是诱导子图中的节点的特征
        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)   #   这个induced_graph是就是生成的诱导子图，包含节点特征和边
        induced_graph_list.append(induced_graph)
        if(index%1000==0):
            print('生成的第{}/{}张子图数据'.format(index,total_node_num))
    
    print('生成了{}/{}张子图数据'.format(index,total_node_num))
    return induced_graph_list


@param('data.seed')
@param('general.cache_dir')
@param('general.few_shot')
def get_supervised_data(dataset, ratios, seed, cache_dir,few_shot):
    #   dataset  是 "photo"，表示下游阶段的数据集。
    #   ratios是[0.1,0.1,0.8]是训练、验证、测试的划分。
    #   cache_dir是存数据集的位置
    #   ratios只有在few_shot == 0的时候才用到。我们设定few_shot == 1，所以用不到ratios
    import os
    cache_dir = os.path.join(cache_dir, dataset)    #   返回值是 storage/.cache/photo
    os.makedirs(cache_dir, exist_ok=True)

    if(few_shot == 0):
        pass

    else:   #   storage/cache/photo 目录下， 创建一个  few_shot_seed.pt文件。 （.pt文件是用来保存 torch模型参数  的文件）
        #   在本代码中，cache_path保存的是下游任务阶段的数据集，包括train_set（8个子图，每个类别1个子图）,val_set（765个子图）,test_set（6877个子图），以及num_classes==8（8分类的节点分类问题）
        cache_path = os.path.join(cache_dir + f'/{few_shot}_shot' + f'_s{seed}' + '.pt')

        if os.path.exists(cache_path):
            return torch.load(cache_path)

        from .pretrain import preprocess, iterate_datasets

        #   iterate_datasets()返回一个迭代器，包含多个Pyg.Data（比如cora,citeseer等）。
        #   下游阶段只返回一个data，即Photo数据集。  Preprocess的作用是，对于给定的data，删除它的mask，并且把节点特征长度统一到100
        data = preprocess(next(iterate_datasets(dataset)))


        #   以photo数据集为例，data.y是一个长度为 7650的，表示每个节点的类别。 
        #   torch.unique(data.y) 返回的是[0,1,2,3,4,5,6,7] ，表示这个photo数据集是一个 8分类的节点分类问题    
        num_classes = torch.unique(data.y).size(0)



        #   train_dict_list是一个 字典，key是 0到7，每个关键字对应的值目前是一个空列表[]
        train_dict_list = {key.item():[] for key in torch.unique(data.y)}


        val_test_list = []
        target_graph_list = induced_graphs(data)    #   data(photo数据集)包含7650个节点，生成7650个诱导子图。（每个节点都生成一个以该节点为中心的诱导子图）



        from torch.utils.data import random_split, Subset

        for index, graph in enumerate(target_graph_list):
            #   index是诱导子图的ID，其实本质上是 photo数据集中每一个节点的ID。而graph就是以这个节点为中心的诱导子图
            i_class = graph.y   #   y就是这个子图中心节点的  类别标签

            if( len(train_dict_list[i_class]) >= few_shot):
                val_test_list.append(graph)
            else:
                train_dict_list[i_class].append(index)
        #   此时，train_dict_list代表的是：每个类别，都有1个该类别的节点的诱导子图（存着这个诱导子图的中心节点ID）。因为few_shot == 1，所以每个类别都只有 1个子图作为训练集
        #   而val_test_list保存了除了训练集之外其余的所有诱导子图        
                


        all_indices = []
        for i_class, indice_list in train_dict_list.items():
            all_indices+=indice_list
        #   此时all_indices 包含8个 节点ID，分别是训练集中，0号类别到7号类别的，诱导子图中心节点ID
            



        train_set = Subset(target_graph_list, all_indices)  #   train_set包含8个子图，就是8个类别，每个类别一个子图


        #   整个photo数据集，一共7650个节点，构造了7650个子图
        #   其中train_set训练集包含了8个子图，每个类别1个子图
        #   剩下的7642个子图，按照 1 比 9的比例分配给val_set和test_set
        #   分配结果是，val_set有765个子图， test_set有6877个子图
        val_set, test_set = random_split(val_test_list, [0.1,0.9], torch.Generator().manual_seed(seed))
        

        
    # import pdb
    # pdb.set_trace()

    results = [     #   这个就是下游阶段，所有的数据数据集
    {
        'train': train_set,
        'val': val_set,
        'test': test_set,
    }, 
        num_classes
    ]

    # save to cache
    torch.save(results, cache_path)

    return results


