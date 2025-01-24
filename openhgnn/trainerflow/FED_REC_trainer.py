import dgl
from time import time
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter
from scipy.sparse import lil_matrix
import torch.utils.data as dataloader
import copy
import random 
import math
from tqdm import tqdm
import torch
from ..dataset import build_dataset
from ..models import build_model
from . import  register_flow
from random import sample
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from abc import ABC


@register_flow("FED_REC_trainer")
class FED_Recommendation(ABC):
    def __init__(self,args):
        super(FED_Recommendation, self).__init__()
        self.dataset = build_dataset(args.dataset,'node_classification',args=args,logger=args.logger)
        self.args = args
        self.logger = self.args.logger
        self.model_name = args.model_name
        self.model = args.model
        self.device = args.device
        self.optimizer = None
        self.dataname = 'acm'
        self.meta_paths_dict =  {'acm':{'user': [['pa','ap'],['pc','cp']],'item':[['ap','pa']]}, \
                   'dblp':{'user': [['pa','ap'], ['pc','cp']],'item':[['ap','pa']]}, \
                   'yelp':{'user': [['pa','ap'], ['pa','aca','caa', 'ap']],'item':[['ap', 'pa']]}, \
                   'DoubanBook':{'item':[['bu','ub'],['bg', 'gb']], 'user':[['ub','bu'],['ua','au']]}}
        p_vs_f = self.dataset.data['PvsL']#12499,73
        p_vs_a = self.dataset.data['PvsA']#(12499, 17431)
        p_vs_t = self.dataset.data['PvsT']#文本特征(12499,1903)
        p_vs_c = self.dataset.data['PvsC']#(12499, 14)
        adj = (p_vs_f, p_vs_a, p_vs_t, p_vs_c)
        label_count, labels, label_author, author_label, shared_knowledge_rep = gen_shared_knowledge(adj, self.args.shared_num)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]
        p_vs_c_filter = p_vs_c[:, conf_ids]#(12499, 5)
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]#只保留至少参与过一次会议的论文,获取非0元素的索引
        p_vs_f = p_vs_f[p_selected]#(4025,73)
        p_vs_a = p_vs_a[p_selected]#(4025,17431)
        p_vs_t = p_vs_t[p_selected]#(4025,1903)
        p_vs_c = p_vs_c[p_selected]#CSC (4025, 14)
        num_nodes_dict = {'paper': p_vs_a.shape[0], 'author': p_vs_a.shape[1], 'field': p_vs_f.shape[1], 'conf': p_vs_c.shape[1]}

        #test_id测试矩阵，train_id训练矩阵，train_fed_id记录作者id，test_negative_id记录负样本，p_vs_a是train矩阵，p_vs_a_random是随机扰动矩阵
        p_vs_a, p_vs_a_random, train_fed_id, train_id, test_id, test_negative_id=train_test_split(p_vs_a)
        #train_id的形式如：[[0,4610],[0,5298]...]
        self.logger.info(self.meta_paths_dict)
        features_user = np.random.normal(loc=0., scale=1., size=[p_vs_a.shape[0], self.args.in_dim])
        features_item = np.random.normal(loc=0., scale=1., size=[p_vs_a.shape[1], self.args.in_dim])
        features = (features_user, features_item)
        self.args.in_size = features_user.shape[1]
        self.args.meta_paths = self.meta_paths_dict[self.dataname]['user']
        model_user = build_model(self.model).build_model_from_args(self.args)#需要修改
        self.args.meta_paths = self.meta_paths_dict[self.dataname]['item']
        self.args.in_size = features_item.shape[1]
        model_item = build_model(self.model).build_model_from_args(self.args)#需要修改
        self.model = (model_user, model_item)
        # 测试集
        test_dataset = RecDataSet(test_id, test_negative_id, is_training=False)
        self.test_dataloader = dataloader.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        
        # 以每篇论文为对象进行扰动,二阶扰动过程,perturb_adj是扰动后的pa关系矩阵
        client_list = []
        p_vs_a_ = []
        all_edges=0
        remain_edges=0
        all_edges_after=0
        for i, items in enumerate(train_fed_id):#each paper
            pre_edges = list(p_vs_a[i].nonzero()[1])
            all_edges+=len(pre_edges)

            client = Client(i, items, self.args)
            perturb_adj = client.perturb_adj(p_vs_a.todense()[i], label_author, author_label, label_count,
                                            shared_knowledge_rep, self.args.p1, self.args.p2)

            cur_edges = list(lil_matrix(perturb_adj).nonzero()[1])
            all_edges_after+=len(cur_edges)
            cur_remain_edges = len(set(pre_edges)&set(cur_edges))#扰动前后保留的边数
            remain_edges+=cur_remain_edges
            p_vs_a_.append(perturb_adj)
            client_list.append(client.to(torch.device(self.args.device)))
        
        self.client_list = client_list
        p_vs_a_ = np.squeeze(np.array(p_vs_a_))
        p_vs_a_ = lil_matrix(p_vs_a_)
        #建图
        self.hg = dgl.heterograph({
            ('paper', 'pa', 'author'): p_vs_a_.nonzero(),
            ('author', 'ap', 'paper'): p_vs_a_.transpose().nonzero(),
            ('paper', 'pf', 'field'): p_vs_f.nonzero(),
            ('field', 'fp', 'paper'): p_vs_f.transpose().nonzero(),
            ('paper', 'pc', 'conf'): p_vs_c.nonzero(),
            ('conf', 'cp', 'paper'): p_vs_c.transpose().nonzero(),
        }, num_nodes_dict = num_nodes_dict).to(torch.device(self.args.device))
        self.server = Server(client_list, self.model, self.hg, features, self.args).to(torch.device(self.args.device))
        
    def train(self):
        loss = 0
        best_sum_score = 0
        best_epoch = 0
        best_score = ()
        # for ep_index in range(self.args.epochs):
        #     for va_index in range(self.args.valid_step):
        for ep_index in tqdm(range(self.args.epochs), desc="Epochs", ncols=80):
            for va_index in tqdm(range(self.args.valid_step), desc="Validation Steps", ncols=80, leave=False):
                t1 = time()
        #随机采样客户端用于训练
                sample_client = random.sample(self.client_list, self.args.batch_size)#64#采样
                self.server.distribute(sample_client)#model
                '''train'''
                param_list = []
                loss_list = []
                t = time()
                if self.args.mini_batch_flag:
                    param_list,loss_list = self.mini_train_step(sample_client)
                else:
                    param_list,loss_list = self._full_train_step(sample_client)
                print(time() - t)
                #聚合参数
                self.server.aggregate(param_list) #!聚合参数
                loss_ = np.mean(np.array(loss_list)).item()
                loss+=loss_
                self.logger.info('training average loss: %.5f, time:%.1f s' % (
                    loss / (ep_index * self.args.valid_step + va_index + 1), time() - t1))
            #test
            hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10 = self._test_step(split='test', ep_index=ep_index)
            # self.server.eval()
            # hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10 = self.server.predict(self.test_dataloader, ep_index)
            cur_score = hit_at_5 + hit_at_10 + ndcg_at_5 + ndcg_at_10
            if(cur_score>best_sum_score):
                best_sum_score = cur_score
                best_epoch = ep_index
                best_score = (hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10)
            self.logger.info('Best Epoch: %d, hit_at_5 = %.4f, hit_at_10 = %.4f, ndcg_at_5 = %.4f, ndcg_at_10 = %.4f'
                % (best_epoch, best_score[0], best_score[1], best_score[2], best_score[3]))
        return 1

    def _full_train_step(self,sample_client):
        param_list = []
        loss_list = []
        for idx, client in enumerate(sample_client):
            client.train()#训练模式
            #要更新的user及其embedding,要更新的item及其embedding，model参数梯度以及loss被返回
            param, loss_c = client.train_(self.hg, self.server.user_emb, self.server.item_emb)
            param_list.append(param)  # !
            loss_list.append(loss_c.cpu())
        return param_list,loss_list 
    def _test_step(self, split=None, ep_index=None):   
        self.server.eval()
        hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10 = self.server.predict(self.test_dataloader, ep_index)
        return hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10

def train_test_split(p_vs_a):
    train_id = []
    train_fed_id = []
    test_id = []
    test_negative_id = []
    p_vs_a_ = copy.deepcopy(p_vs_a)#
    p_vs_a_random = copy.deepcopy(p_vs_a)
    p_vs_a_random = p_vs_a_random.tolil()
    p_num = p_vs_a_.shape[0]
    a_num = p_vs_a_.shape[1]
    for i in range(p_num):#each paper
        cur_a = p_vs_a_[i].nonzero()[1]
        '''p_vs_a random'''
        p_vs_a_random[i,:]=0
        sample_len = len(cur_a)
        sample_a = random.sample(list(range(p_vs_a_.shape[1])), sample_len)
        #print(sample_a)
        p_vs_a_random[i, sample_a] = 1
        # print(p_vs_a_random[i].nonzero()[1])
        '''end'''
#如果只有一个作者，放在训练集，如果有多个，random一个为测试，其余放在训练，test_id测试矩阵，train_id训练矩阵，train_fed_id记录作者id，test_negative_id记录负样本
        if(len(cur_a)==1):
            train_id.append([i, cur_a[0]])
            train_fed_id.append(list(cur_a))#
        elif(len(cur_a)!=0):
            sample_train = random.sample(list(cur_a), len(cur_a)-1)
            train_fed_id.append(sample_train)#
            for j in sample_train:
                train_id.append([i, j])
            cur_test_id =list(set(cur_a)-set(sample_train))[0]
            test_id.append([i, cur_test_id])
            p_vs_a_[i, cur_test_id] = 0

            '''p_vs_a random'''
            p_vs_a_random[i, cur_test_id] = 0#random
            '''end'''

            test_negative_pool = list(set(range(a_num))-set(cur_a))#0-10... -
            test_negative_id.append(random.sample(test_negative_pool, 99))
        else:
            train_fed_id.append([])
    #print(len(train_fed_id))
    #print(test_negative_id[2])
    return p_vs_a_, p_vs_a_random, train_fed_id, train_id, test_id, test_negative_id
'''cluster'''
def cluster(feature_list, n_clusters):
    s = KMeans(n_clusters=n_clusters).fit(feature_list)
    #print(len(s.cluster_centers_))
    #每个样本所属的簇
    #print(len(s.labels_))
    label_count = {}
    for i in s.labels_:
        if(i not in label_count.keys()):
            label_count[i] = 1
        else:
            label_count[i]+=1
# 20个簇中每个样本的数量
    print(label_count)
    #print(s.labels_)
# 簇中有哪些作者，以及作者对应哪些簇
    label_author = {}
    author_label = {}
    labels = []
    for i, k in enumerate(s.labels_):
        author = i
        label = k
        labels.append(label)

        author_label[author] = label

        if(label not in label_author.keys()):
            label_author[label] = [author]
        else:
            label_author[label].append(author)

    # with open("./data_event/author_label", "w") as f:
    #     for l in author_label:
    #         f.write(l[0] + '\t' + l[1] + '\n')
# 返回簇的数量，所有作者的簇标签，簇到作者的映射，作者到簇的映射
    return label_count, labels, label_author, author_label
#cluster()
def gen_shared_knowledge(adj, group_num):
    p_vs_f = adj[0]#(4025,73)
    p_vs_a = adj[1]#(4025,17431)
    p_vs_t = adj[2]#(4025,1903)
    p_vs_c = adj[3]#CSC (4025, 14)
    a_vs_t = p_vs_a.T * p_vs_t#(17431,1903)作者和文本特征
    a_vs_f = p_vs_a.T * p_vs_f#(17431,73)作者和特征词
    a_vs_c = p_vs_a.T * p_vs_c#(17431,14)作者和会议
    a_vs_p = p_vs_a.T#(17431,12499)作者和文章
    #稀疏转稠密
    a_vs_t_dense = a_vs_t.todense()
    a_vs_f_dense = a_vs_f.todense()
    a_vs_c_dense = a_vs_c.todense()
    a_vs_p_dense = a_vs_p.todense()
    #print(np.sum(a_vs_c_dense.sum(-1)==0))#大部分(10264)=0
    #print(a_vs_t_dense[1])
    # 沿着列合并，一个数组的话不改变
    a_feature = np.concatenate([a_vs_c_dense], -1)
    a_feature = np.asarray(a_feature)
    label_count, labels, label_author, author_label = cluster(a_feature, group_num) #20
    # x,y = tsne(a_feature)
    # plot_embedding_2d(x, y, labels)
    shared_knowledge_rep = get_shared_knowledge_rep(a_feature, label_author)
    return label_count, labels, label_author, author_label, shared_knowledge_rep
# 为每个共享HIN计算一个代表的特征表示
'''get shared knowledge rep(每个shared HIN的表示为它所包含的item的表示的平均）'''
def get_shared_knowledge_rep(item_feature_list, label_author):
    shared_knowledge_rep = {}
    for label, author_list in label_author.items():
        features = item_feature_list[author_list]
        rep = np.mean(features, 0)#rep的shape是(1,14)
        # sum = np.array([0.0]*len(item_feature_list[0]))
        # l = len(author_list)
        # for author in author_list:
        #     sum+= item_feature_list[author]
        # rep = sum/l
        shared_knowledge_rep[label] = rep
    return shared_knowledge_rep
class RecDataSet(torch.utils.data.Dataset):
    def __init__(self, test_id, test_negative_id, is_training=True):
        super(RecDataSet, self).__init__()
        self.is_training = is_training
        if(self.is_training==False): #test
            self.data = (np.array(test_id), np.array(test_negative_id))

    def __getitem__(self,index):
        if(self.is_training==False):
            user = self.data[0][index][0]
            item = self.data[0][index][1]
            negtive_item = self.data[1][index]
            return user, item, negtive_item


    def __len__(self):
        #return self.x.size()
        if(self.is_training==False):
            return len(self.data[0])
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def eps2p(epsilon, n=2):
    return np.e ** epsilon / (np.e ** epsilon + n - 1)
#进行扰动
'''矩阵形式'''
def perturbation_test(value, perturbed_value, epsilon):
    value = np.array(value)
    perturbed_value = np.array(perturbed_value)
    per_eps = epsilon#epsilon/(value.shape[0]*value.shape[1])
    rnd = np.random.random(value.shape)
    p = eps2p(per_eps)
    return np.where(rnd<p, value, np.ones((value.shape))*perturbed_value)
class Client(nn.Module):
    def __init__(self, user_id, item_id, args):
        super().__init__()
        self.device = args.device
        self.user_id = user_id
        self.item_id = item_id #list
        #self.semantic_neighbors = semantic_neighbors


    def negative_sample(self, total_item_num):
        '''生成item负样本集合'''
        #从item列表里随机选取item作为user的负样本
        item_neg_ind = []
        #item_neg_ind和item_id数量一样
        for _ in self.item_id:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in self.item_id:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        '''生成item负样本集合end'''
        return item_neg_ind
#生成负样本
    def negative_sample_with_augment(self, total_item_num, sampled_items):
        item_set = self.item_id+sampled_items
        '''生成item负样本集合'''
        #从item列表里随机选取item作为user的负样本
        item_neg_ind = []
        #item_neg_ind和item_id数量一样
        for _ in item_set:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in item_set:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        '''生成item负样本集合end'''
        return item_neg_ind

    def sample_item_augment(self, item_num):
        ls = [i for i in range(item_num) if i not in self.item_id]
        sampled_items = sample(ls, 5)

        return sampled_items

# 二阶扰动
    def perturb_adj(self, value, label_author, author_label, label_count, shared_knowledge_rep, eps1, eps2):
        #print(value.shape) #1,17431
        #此用户的item共可分成多少个groups
        groups = {}
        for item in self.item_id:
            group = author_label[item]
            if(group not in groups.keys()):
                groups[group] = [item]
            else:
                groups[group].append(item)

        '''step1:EM'''
        num_groups = len(groups)
        quality = np.array([0.0]*len(label_author))
        G_s_u =  groups.keys()
        if(len(G_s_u)==0):#此用户没有交互的item，则各个位置quality平均
            for group in label_author.keys():
                quality[group] = 1
            num_groups = 1
        else:
            for group in label_author.keys():
                qua = max([(cosine_similarity(shared_knowledge_rep[g].reshape(1, -1), shared_knowledge_rep[group].reshape(1, -1))+1)/2.0 for g in G_s_u])
                quality[group] = qua

        EM_eps = eps1/num_groups
        EM_p = EM_eps*quality/2 #隐私预算1 eps
        EM_p = softmax(EM_p)

        #按照概率选择group
        select_group_keys = np.random.choice(range(len(label_author)), size = len(groups), replace = False, p = EM_p)
        select_group_keys_temp = list(select_group_keys)
        degree_list = [len(v) for _, v in groups.items()]
        new_groups = {}

        for key in select_group_keys:#先把存在于当前用户的shared knowledge拿出来
            key_temp = key
            if(key_temp in groups.keys()):
                new_groups[key_temp] = groups[key_temp]
                degree_list.remove(len(groups[key_temp]))
                select_group_keys_temp.remove(key_temp)

        for key in select_group_keys_temp:#不存在的随机采样交互的item，并保持度一致
            key_temp = key
            cur_degree = degree_list[0]
            if(len(label_author[key_temp]) >= cur_degree):
                new_groups[key_temp] = random.sample(label_author[key_temp], cur_degree)
            else:#需要的度比当前group的size大，则将度设为当前group的size
                new_groups[key_temp] = label_author[key_temp]
            degree_list.remove(cur_degree)

        groups = new_groups
        value = np.zeros_like(value)#一定要更新value
        for group_id, items in groups.items():
            value[:,items] = 1
        '''pure em'''
        #value_rr = value


# all_items包含所有items的index，select_items是其shared hin簇里面所有对应的index，mask_rr是两个的差
        '''step2:rr'''
        all_items = set(range(len(author_label)))
        select_items = []
        for group_id, items in groups.items():
            select_items.extend(label_author[group_id])
        mask_rr = list(all_items - set(select_items))

        '''rr'''
        value_rr = perturbation_test(value, 1-value, eps2)
        #print(np.sum(value_rr)) 4648
        value_rr[:, mask_rr] = 0#由mask_rr指定的列为0
        # #print(np.sum(value_rr)) 469
        #
        '''dprr'''
        for group_id, items in groups.items():
            degree = len(items)
            n = len(label_author[group_id])
            p = eps2p(eps2)
            q = degree/(degree*(2*p-1) + (n)*(1-p))
            rnd = np.random.random(value_rr.shape)
            #原来是0的一定还是0，原来是1的以概率q保持1，以达到degree减少
            dprr_results = np.where(rnd<q, value_rr, np.zeros((value_rr.shape)))
            value_rr[:, label_author[group_id]] = dprr_results[:, label_author[group_id]]
        # print('....')
        # print(self.item_id)
        # print(value_rr.nonzero()[1])
        return value_rr




#赋值
    def update(self, model_user, model_item):
        self.model_user = copy.deepcopy(model_user)
        self.model_item = copy.deepcopy(model_item)
        # self.item_emb.weight.data = Parameter(aggr_param['item'].weight.data.clone())


    def train_(self, hg, user_emb, item_emb):
        total_item_num = item_emb.weight.shape[0]#共有17431个item，即用户
        user_emb = torch.clone(user_emb.weight).detach()
        item_emb = torch.clone(item_emb.weight).detach()
        user_emb.requires_grad = True
        item_emb.requires_grad = True
        user_emb.grad = torch.zeros_like(user_emb)#初始化梯度为0
        item_emb.grad = torch.zeros_like(item_emb)


        self.model_user.train()
        self.model_item.train()

        #sample_item_augment
        sampled_item = self.sample_item_augment(total_item_num)#随机采样
        item_neg_id = self.negative_sample_with_augment(total_item_num, sampled_item)#得到负样本,neg_id的数量是self.item_id+sample_item_id
        #item_neg_id = self.negative_sample(total_item_num)
#这里还没看，model的过程
        logits_user = self.model_user(hg, user_emb)#+user_emb
        logits_item = self.model_item(hg, item_emb)#+item_emb

        cur_user = logits_user[self.user_id]
        #cur_item_pos = logits_item[self.item_id]
        cur_item_pos = logits_item[self.item_id+sampled_item]
        cur_item_neg = logits_item[item_neg_id]

        pos_scores = torch.sum(cur_user * cur_item_pos, dim=-1)
        neg_scores = torch.sum(cur_user * cur_item_neg, dim=-1)
        loss = -(pos_scores - neg_scores).sigmoid().log().sum()


        self.model_user.zero_grad()
        self.model_item.zero_grad()

        loss.backward()
        #self.optimizer.step()

        #grad获取每个参数梯度
        model_grad_user = []
        model_grad_item = []
        for param in list(self.model_user.parameters()):
            grad = param.grad
            model_grad_user.append(grad)
        for param in list(self.model_item.parameters()):
            grad = param.grad
            model_grad_item.append(grad)
#筛选需要更新的embedding->updated_items
        mask_item = item_emb.grad.sum(-1)!=0#直接通过grad！=0
        updated_items = np.array(range(item_emb.shape[0]))[mask_item.cpu()]#list(set(self.item_id + item_neg_id))
        #print(updated_items)
        item_grad = item_emb.grad[updated_items, :]#提取要更新的item_emb,updated_item是item的索引，item_grad是其对应的embedding


        mask_user = user_emb.grad.sum(-1)!=0
        updated_users = np.array(range(user_emb.shape[0]))[mask_user.cpu()]#list(set([self.user_id] + self.semantic_neighbors))
        #print(len(updated_users))
        user_grad = user_emb.grad[updated_users, :]#
        #print(user_grad)
        # torch.cuda.empty_cache()


        return {'user': (user_grad, updated_users), 'item' : (item_grad, updated_items), 'model': (model_grad_user, model_grad_item)}, \
               loss.detach()

class Server(nn.Module):
    def __init__(self, client_list, model, hg, features, args):
        super().__init__()
        self.device = args.device
        self.hg = hg
        self.client_list = client_list
        self.features = features
        self.model_user = model[0]#(0:model_user, 1: model_item)
        self.model_item = model[1]
        self.user_emb = nn.Embedding(features[0].shape[0], features[0].shape[1]).to(self.device)
        self.item_emb = nn.Embedding(features[1].shape[0], features[1].shape[1]).to(self.device)
        self.user_emb.weight.data = Parameter(torch.Tensor(features[0])).to(self.device)
        self.item_emb.weight.data = Parameter(torch.Tensor(features[1])).to(self.device)
        #nn.init.normal_(self.item_emb.weight, std=0.01)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.logger = args.logger



    def aggregate(self, param_list):
        flag = False
        number = 0#更新了多少个项目
        gradient_item = torch.zeros_like(self.item_emb.weight)#(17431,64)
        gradient_user = torch.zeros_like(self.user_emb.weight)#(4025,64)
        item_count = torch.zeros(self.item_emb.weight.shape[0]).to(self.device)
        user_count = torch.zeros(self.user_emb.weight.shape[0]).to(self.device)

        for parameter in param_list:
            model_grad_user, model_grad_item = parameter['model']
            item_grad, returned_items = parameter['item']
            user_grad, returned_users = parameter['user']
            num = len(returned_items)
            item_count[returned_items] += 1
            user_count[returned_users] += num

            number += num
            if not flag:
                flag = True
                gradient_model_user = []
                gradient_model_item = []
                gradient_item[returned_items, :] += item_grad * num#为什么要乘num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user.append(model_grad_user[i]* num)
                for i in range(len(model_grad_item)):
                    gradient_model_item.append(model_grad_item[i]* num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user[i] += model_grad_user[i] * num
                for i in range(len(model_grad_item)):
                    gradient_model_item[i] += model_grad_item[i] * num

        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)
        for i in range(len(gradient_model_user)):
            gradient_model_user[i] = gradient_model_user[i] / number
        for i in range(len(gradient_model_item)):
            gradient_model_item[i] = gradient_model_item[i] / number


        #更新model参数,model的参数更新
        ls_model_param_user = list(self.model_user.parameters())
        ls_model_param_item = list(self.model_item.parameters())
        for i in range(len(ls_model_param_user)):
            ls_model_param_user[i].data = ls_model_param_user[i].data - self.lr * gradient_model_user[i] - self.weight_decay * ls_model_param_user[i].data
        for i in range(len(ls_model_param_item)):
            ls_model_param_item[i].data = ls_model_param_item[i].data - self.lr * gradient_model_item[i] - self.weight_decay * ls_model_param_item[i].data

        # for i in range(len(list(self.model_user.parameters()))):
        #     print(ls_model_param_user[i].data)
        #     break
        #更新item/user参数
        item_index = gradient_item.sum(dim = -1) != 0
        user_index = gradient_user.sum(dim = -1) != 0
        with torch.no_grad():#不加会报错
            self.item_emb.weight[item_index] = self.item_emb.weight[item_index] -  self.lr * gradient_item[item_index] - self.weight_decay * self.item_emb.weight[item_index]
            self.user_emb.weight[user_index] = self.user_emb.weight[user_index] -  self.lr * gradient_user[user_index] - self.weight_decay * self.user_emb.weight[user_index]



    def distribute(self, client_list):
        for client in client_list:
            client.update(self.model_user, self.model_item)


    def predict(self, test_dataloader, epoch):
        hit_at_5 = []
        hit_at_10 = []
        ndcg_at_5 = []
        ndcg_at_10 = []

        self.model_item.eval()
        self.model_user.eval()
        logits_user = self.model_user(self.hg, self.user_emb.weight)
        logits_item = self.model_item(self.hg, self.item_emb.weight)
        for u, i, neg_i in test_dataloader: #test_i算上了test_negative, 真实的放在最后一位[99],#获取user,user的真实item，负样本集neg_i
            cur_user = logits_user[u]#(64,64)
            cur_item = logits_item[i]#(64,64)
            rating = torch.sum(cur_user * cur_item, dim=-1)#当前client user和所有item点乘(include test item),(64,)

            for eva_idx, eva in enumerate(rating):
                cur_neg = logits_item[neg_i[eva_idx]]#(99,64)
                cur_rating_neg = torch.sum(cur_user[eva_idx] * cur_neg, dim=-1)#(99,)
                #print(np.shape(cur_rating_neg))
                cur_eva = torch.cat([cur_rating_neg, torch.unsqueeze(rating[eva_idx], 0)], dim=0)#(100,)
                #print(np.shape(rating[eva_idx]))
                # print(cur_eva)
                hit_at_5_ = evaluate_recall(cur_eva, [99], 5)#[99]是测试集(ground truth)
                hit_at_10_ = evaluate_recall(cur_eva, [99], 10)
                ndcg_at_5_ = evaluate_ndcg(cur_eva, [99], 5)
                ndcg_at_10_ = evaluate_ndcg(cur_eva, [99], 10)
                #print(hit_at_10_)
                hit_at_5.append(hit_at_5_)
                hit_at_10.append(hit_at_10_)
                ndcg_at_5.append(ndcg_at_5_)
                ndcg_at_10.append(ndcg_at_10_)
        hit_at_5 = np.mean(np.array(hit_at_5)).item()
        hit_at_10 = np.mean(np.array(hit_at_10)).item()
        ndcg_at_5 = np.mean(np.array(ndcg_at_5)).item()
        ndcg_at_10 = np.mean(np.array(ndcg_at_10)).item()
        self.logger.info('Epoch: %d, hit_at_5 = %.4f, hit_at_10 = %.4f, ndcg_at_5 = %.4f, ndcg_at_10 = %.4f'
              % (epoch, hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10))
        # logging.info('Epoch: %d, hit_at_5 = %.4f, hit_at_10 = %.4f, ndcg_at_5 = %.4f, ndcg_at_10 = %.4f'
        #       % (epoch, hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10))
        return hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10

def getP(ranklist, gtItems):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)

def getR(ranklist, gtItems):
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return  dcg

def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg

def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg


'''下面是两个大指标(recall或ndcg)，4个小指标的计算代码'''
#指标1 top_k=5 或 10   得到recall@5 或recall@10 指标
def evaluate_recall(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()

    hit = 0 #
    for i, v in enumerate(rating_k):
        if v in ground_truth:
            hit += 1

    recall = hit / len(ground_truth)
    return recall

#指标2 top_k = 5 或 10 得到ndcg@5 或ndcg@10 指标
def evaluate_ndcg(rating, ground_truth, top_k):#参照NDCG的定义
    _, rating_k = torch.topk(rating, top_k)#values, indices
    rating_k = rating_k.cpu().tolist() #indices
    dcg, idcg = 0., 0.

    for i, v in enumerate(rating_k):
        if i < len(ground_truth):#前len（）个是真实交互的
            idcg += (1 / np.log2(2 + i))#这里相关性为0或1（真实交互为1，未交互为0）
        if v in ground_truth:
            dcg += (1 / np.log2(2 + i))

    ndcg = dcg / idcg
    return ndcg