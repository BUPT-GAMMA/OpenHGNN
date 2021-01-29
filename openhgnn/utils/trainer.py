import time
import torch.optim as optim
import torch as th
from openhgnn.utils.sampler import get_epoch_samples
from openhgnn.utils.utils import print_dict
import torch.nn.functional as F
import dgl

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

def train(model, g, config):
    g_homo = dgl.to_homogeneous(g)
    pos_edges = g_homo.edges()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    for epoch in range(config.max_epoch):
        epoch_start_time = time.time()
        neg_edges, ns_samples = get_epoch_samples(g, epoch, config.dataset, config.num_ns_neg)

        optimizer.zero_grad()
        node_emb, ns_prediction = model(g, ns_samples)
        # compute loss
        pairwise_loss = cal_node_pairwise_loss(node_emb, pos_edges, neg_edges)
        ns_label = th.cat([ns['label'] for ns in ns_samples]).type(th.float32)
        cla_loss = cal_cla_loss(ns_prediction, ns_label)
        loss = pairwise_loss + cla_loss * config.beta
        loss.backward()
        epoch_dict = {'Epoch': epoch, 'train_loss': loss.item(),
                      'pairwise_loss': pairwise_loss.item(),
                      'cla_loss': cla_loss.item(),
                      'time': time.time() - epoch_start_time}
        print_dict(epoch_dict, '\n')
        optimizer.step()


