import dgl
import time
import tqdm
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from openhgnn.utils.sampler import get_epoch_samples
from openhgnn.utils.dgl_graph import give_one_hot_feats, normalize_edges, edata_in_out_mask, load_link_pred
from openhgnn.utils.utils import print_dict, h2dict
from openhgnn.utils.evaluator import evaluate, cal_loss_f1


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
    if config.adaptive_lr == 'False':
        optimizer = th.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = th.optim.Adam([{'params': model.weight},
                                      {'params': model.linear1.parameters()},
                                      {'params': model.linear2.parameters()},
                                      {"params": model.layers.parameters(), "lr": 0.5}
                                      ], lr=0.005, weight_decay=0.001)

    best_val_loss = 10000
    best_test_loss = 10000
    best_train_loss = 10000
    best_train_f1 = 0
    best_val_f1 = 0
    best_test_f1 = 0
    loss_func = th.nn.CrossEntropyLoss()
    g_homo = dgl.to_homogeneous(g, ndata=['h'])
    for i in range(config.max_epoch):
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 0.005:
                param_group['lr'] = param_group['lr'] * 0.9
        print('Epoch:  ', i + 1)
        model.zero_grad()
        model.train()

        y_train = model(g_homo)
        emd = h2dict(y_train, g.ndata['h'])
        tar_y = emd['paper']
        tar_data = g.nodes['paper'].data

        #evaluate(config.seed, config.dataset, g.ndata['h'], g)
        # evaluate(config.seed, config.dataset, h2dict(X_, g.ndata['h']), g)
        # evaluate(config.seed, config.dataset, h2dict(a, g.ndata['h']), g)

        loss, macro_f1, micro_f1 = cal_loss_f1(tar_y, tar_data, loss_func, 'train_mask')
        print('Train - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(loss.item(), macro_f1, micro_f1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_f1 = macro_f1
        # Valid
        model.eval()
        with th.no_grad():
            #y_test = models.forward(g_homo)
            emd = h2dict(y_train, g.ndata['h'])
            tar_y = emd['paper']
            loss, macro_f1, micro_f1 = cal_loss_f1(tar_y, tar_data, loss_func, 'valid_mask')
            print('Valid - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(loss.item(), macro_f1, micro_f1))
            val_f1 = macro_f1

            loss, macro_f1, micro_f1 = cal_loss_f1(tar_y, tar_data, loss_func, 'test_mask')
            print('Test - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(loss.item(), macro_f1, micro_f1))
            test_f1 = macro_f1
            if val_f1 > best_val_f1:
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
        #th.empty_cache()
    print('---------------Best Results--------------------')
    print('Train -  Macro_F1: {}'.format(best_train_f1))
    print('Valid -   Macro_F1: {}'.format(best_val_f1))
    print('Test -  Macro_F1: {}'.format(best_test_f1))


def get_split(hg, g, category):
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('labels')
    # get target category id
    node_ids = th.arange(g.number_of_nodes())
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    return target_idx, train_idx, test_idx, labels


def run_RSHN(model, hg, cl_graph, config):

    optimizer = th.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    g = dgl.to_homogeneous(hg)
    target_idx, train_idx, test_idx, labels = get_split(hg, g, config.category)
    if config.validation == True:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    best_test_acc = 0
    for epoch in range(config.max_epoch):
        print('Epoch:  ', epoch + 1)
        model.train()
        optimizer.zero_grad()
        logits = model(g, cl_graph)

        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        loss.backward()
        optimizer.step()

        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        if config.validation == True:
            val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
            val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
            print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                  format(train_acc, loss.item(), val_acc, val_loss.item()))
        else:
            print("Train Accuracy: {:.4f} | Train Loss: {:.4f} ".
              format(train_acc, loss.item()))

        model.eval()
        logits = model.forward(g, cl_graph)
        logits = logits[target_idx]
        test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
        print()
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    print('Test - ACC: {}'.format(best_test_acc))


    return


def run_CompGCN(model, hg, config):

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    edata_in_out_mask(hg)
    g = dgl.to_homogeneous(hg, edata=['in_edges_mask', 'out_edges_mask'])

    target_idx, train_idx, test_idx, labels = get_split(hg, g, config.category)
    if config.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx
    model.train()
    best_test_acc = 0
    for epoch in range(config.max_epoch):
        optimizer.zero_grad()
        t0 = time.time()
        model.train()
        logits = model(g)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
        print()
        model.eval()
        logits = model.forward(g)
        logits = logits[target_idx]
        test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
        print()
        print()
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        # print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
    print('Test - ACC: {}'.format(best_test_acc))


def run_RGCN(model, hg, config):

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    # calculate norm for each edge type and store in edge
    hg = normalize_edges(hg, 'norm')
    g = dgl.to_homogeneous(hg, edata=['norm'])

    # since the nodes are featureless, the input feature is then the node id.
    feats = th.arange(g.number_of_nodes()).to(config.device)

    edge_type = g.edata[dgl.ETYPE].long()
    edge_norm = g.edata['norm']

    target_idx, train_idx, test_idx, labels = get_split(hg, g, config.category)
    if config.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx
    model.train()
    for epoch in range(config.max_epoch):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, feats)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
        print()
        model.eval()
        logits = model.forward(g, feats, edge_type, edge_norm)
        logits = logits[target_idx]
        test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
        print()

        print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))



class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for _, etype, _ in g.canonical_etypes
        }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict




def run_HetGNN(model, hg, het_graph, config):
    # het_graph is used to sample neighbour
    hg = hg.to('cpu')
    category = config.category
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('label')
    label_mask = hg.nodes[category].data.pop('label_mask')
    emd =hg.nodes[category].data['dw_embedding']
    train_batch = load_link_pred('./openhgnn/dataset/a_a_list_train.txt')
    test_batch = load_link_pred('./openhgnn/dataset/a_a_list_test.txt')
    author_link_prediction(emd, train_batch, test_batch)
    micro_f1, macro_f1 = nc_with_split(emd, labels, train_idx, test_idx)
    print('<Classification> DW    Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro_f1, macro_f1))
    # HetGNN Sampler
    from torch.utils.data import IterableDataset, DataLoader
    from openhgnn.utils.sampler import SkipGramBatchSampler, HetGNNCollator, NeighborSampler
    batch_sampler = SkipGramBatchSampler(hg, config.batch_size, config.window_size)
    neighbor_sampler = NeighborSampler(het_graph, hg.ntypes, batch_sampler.num_nodes, config.device)
    collator = HetGNNCollator(neighbor_sampler, hg)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=config.num_workers)

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    # train_eid_dict = {etype: hg.edges(etype=etype, form='eid')
    #     for etype in hg.etypes}
    opt = th.optim.Adam(model.parameters())
    # dataloader = dgl.dataloading.EdgeDataLoader(
    #     # The following arguments are specific to NodeDataLoader.
    #     hg,  # The graph
    #     train_eid_dict,  # The edges to iterate over
    #     sampler,  # The neighbor sampler
    #     negative_sampler=NegativeSampler(hg, 1),  # The negative sampler
    #     device=config.device,  # Put the MFGs on CPU or GPU
    #     # The following arguments are inherited from PyTorch DataLoader.
    #     batch_size=1024,  # Batch size
    #     shuffle=True,  # Whether to shuffle the nodes for every epoch
    #     drop_last=False,  # Whether to drop the last incomplete batch
    #     num_workers=0  # Number of sampler processes
    # )
    pred = ScorePredictor()
    dataloader_it = iter(dataloader)
    het_graph = het_graph.to(config.device)
    input_features = extract_feature(het_graph, hg.ntypes)
    x = model(het_graph, input_features)
    author_link_prediction(x['author'].to('cpu').detach(), train_batch, test_batch)

    micro_f1, macro_f1 = nc_with_split(x[config.category].to('cpu').detach(), labels, train_idx, test_idx)
    print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro_f1, macro_f1))
    for i in range(config.max_epoch):
        model.train()
        for batch_id in tqdm.trange(config.batches_per_epoch):
            positive_graph, negative_graph, blocks = next(dataloader_it)
            blocks = [b.to(config.device) for b in blocks]
            positive_graph = positive_graph.to(config.device)
            negative_graph = negative_graph.to(config.device)
            # we need extract multi-feature
            input_features = extract_feature(blocks[0], hg.ntypes)

            x = model(blocks[0], input_features)
            loss = compute_loss(pred(positive_graph, x), pred(negative_graph, x))

            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch {:05d} |Train - Loss: {:.4f}'.format(i, loss.item()))
        input_features = extract_feature(het_graph, hg.ntypes)
        x = model(het_graph, input_features)
        author_link_prediction(x['author'].to('cpu').detach(), train_batch, test_batch)
        micro_f1, macro_f1 = nc_with_split(x[config.category].to('cpu').detach(), labels, train_idx, test_idx)
        print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro_f1, macro_f1))
    pass


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']

def compute_loss(pos_score, neg_score):
    # an example hinge loss
    loss = []
    for i in pos_score:
        loss.append(F.logsigmoid(pos_score[i]))
        loss.append(F.logsigmoid(-neg_score[i]))
    loss = th.cat(loss)
    return -loss.mean()


def extract_feature(g, ntypes):
    input_features = {}
    for n in ntypes:
        ndata = g.srcnodes[n].data
        data = {}
        data['dw_embedding'] = ndata['dw_embedding']
        data['abstract'] = ndata['abstract']
        if n == 'paper':
            data['title'] = ndata['title']
            data['venue'] = ndata['venue']
            data['author'] = ndata['author']
            data['reference'] = ndata['reference']
        input_features[n] = data

    return input_features
