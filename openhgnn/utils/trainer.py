import time
import torch.optim as optim
import torch as th
from openhgnn.utils.sampler import get_epoch_samples
from openhgnn.utils.utils import print_dict
import torch.nn.functional as F
import dgl
from openhgnn.utils.evaluater import evaluate

def cal_node_pairwise_loss(node_emd, edge, neg_edge):
    # cross entropy loss from LINE
    # pos loss
    inner_product = cal_inner_product(node_emd, edge)
    pos_loss = - th.mean(F.logsigmoid(inner_product))
    # neg loss
    inner_product = cal_inner_product(node_emd, neg_edge)
    neg_loss = - th.mean(F.logsigmoid(-1 * inner_product))
    loss = pos_loss + neg_loss
    return loss

def cal_inner_product(node_emd, edge):
     emb_u_i = node_emd[edge[0]]
     emb_u_j = node_emd[edge[1]]
     inner_product = th.sum(emb_u_i * emb_u_j, dim=1)
     return inner_product

def cal_cla_loss(predict, ns_label):
    BCE_loss = th.nn.BCELoss()
    return BCE_loss(predict, ns_label)

def run(model, g, config):
    g_homo = dgl.to_homogeneous(g)
    pos_edges = g_homo.edges()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    evaluate(config.seed, config.dataset, g.ndata['h'], g)
    model.train()
    for epoch in range(config.max_epoch):
        epoch_start_time = time.time()
        neg_edges, ns_samples = get_epoch_samples(g, epoch, config.dataset, config.num_ns_neg, config.device)

        optimizer.zero_grad()
        node_emb, ns_prediction, eva_h = model(g, ns_samples)
        # compute loss
        pairwise_loss = cal_node_pairwise_loss(node_emb, pos_edges, neg_edges)
        ns_label = th.cat([ns['label'] for ns in ns_samples]).type(th.float32).to(config.device)
        cla_loss = cal_cla_loss(ns_prediction, ns_label)
        loss = pairwise_loss + cla_loss * config.beta
        loss.backward()
        epoch_dict = {'Epoch': epoch, 'train_loss': loss.item(),
                      'pairwise_loss': pairwise_loss.item(),
                      'cla_loss': cla_loss.item(),
                      'time': time.time() - epoch_start_time}
        print_dict(epoch_dict, '\n')
        optimizer.step()
    model.eval()
    node_emb, ns_prediction, eva_h = model(g, ns_samples)
    evaluate(config.seed, config.dataset, eva_h, g)
    return eva_h




def run_GTN(model, g, config):
    CEloss = th.nn.CrossEntropyLoss()
    if config.adaptive_lr == 'False':
        optimizer = th.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    else:
        optimizer = th.optim.Adam([{'params': model.weight},
                                      {'params': model.linear1.parameters()},
                                      {'params': model.linear2.parameters()},
                                      {"params": model.layers.parameters(), "lr": 0.5}
                                      ], lr=0.005, weight_decay=0.001)

    for i in range(config.max_epoch):
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 0.005:
                param_group['lr'] = param_group['lr'] * 0.9
        print('Epoch:  ', i + 1)
        model.zero_grad()
        model.train()
        g_homo = dgl.to_homogeneous(g, ndata=['h'])
        y_train = model(g_homo)
        tar_y = y_train[5912:8937]
        tar_data = g.nodes['paper'].data
        loss, train_f1 = cal_loss(tar_y, tar_data)
        print('Train - Loss: {}, Macro_F1: {}'.format(loss, train_f1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Valid

from sklearn.metrics import f1_score
def cal_loss(y, tar_data):
    train_mask = tar_data['train_mask']
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    tar_train = tar_data['label'][train_idx].to('cpu')
    y_train = y[train_idx]
    loss_func = th.nn.CrossEntropyLoss()
    loss = loss_func(y_train, tar_train)
    train_f1 = f1_score(tar_train, th.argmax(y_train, dim=1), average='micro')
    return loss, train_f1


