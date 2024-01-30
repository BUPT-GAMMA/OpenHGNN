import os
import random
import logging
import argparse
import math
import dgl
import numpy as np
import time
from numpy.random.mtrand import set_state
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.contrib.data import load_data
from torch.nn.init import xavier_normal_
from dgl import function as fn
from torch.utils.data import DataLoader
from ..utils.wgcn_data import load_data

from ..utils.wgcn_evaluation_dgl import ranking_and_hits
from ..utils.wgcn_utils import EarlyStopping
from ..utils.wgcn_batch_prepare import TrainBatchPrepare, EvalBatchPrepare
from . import BaseFlow, register_flow
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task






@register_flow("SACN_trainer")
class SACNTrainer(BaseFlow):
    def __init__(self, args):
        self.args = args
        self.args.dataset_name=args.dataset
        self.model_name=args.model
        args.model_name=args.model
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.task = build_task(args)
        # self.model = build_model(self.model_name).build_model_from_args(
        #     self.args).model
        print(args)
        main(args)
    def train(self):
        print("111")

def process_triplets(triplets, all_dict, num_rels):
    """
    process triples, store the id of all entities corresponding to (head, rel)
    and (tail, rel_reverse) into dict
    """
    data_dict = {}
    for i in range(triplets.shape[0]):
        e1, rel, e2 = triplets[i]
        rel_reverse = rel + num_rels

        if (e1, rel) not in data_dict:
            data_dict[(e1, rel)] = set()
        if (e2, rel_reverse) not in data_dict:
            data_dict[(e2, rel_reverse)] = set()

        if (e1, rel) not in all_dict:
            all_dict[(e1, rel)] = set()
        if (e2, rel_reverse) not in all_dict:
            all_dict[(e2, rel_reverse)] = set()

        all_dict[(e1, rel)].add(e2)
        all_dict[(e2, rel_reverse)].add(e1)

        data_dict[(e1, rel)].add(e2)
        data_dict[(e2, rel_reverse)].add(e1)

    return data_dict


def preprocess_data(train_data, valid_data, test_data, num_rels):
    all_dict = {}

    train_dict = process_triplets(train_data, all_dict, num_rels)
    valid_dict = process_triplets(valid_data, all_dict, num_rels)
    test_dict = process_triplets(test_data, all_dict, num_rels)

    return train_dict, valid_dict, test_dict, all_dict


def main(args):
    os.makedirs('./logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(
                'logs', args.decoder+"_"+args.name)),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)
    # load graph data
    data = load_data(args.dataset_data)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    save_path = 'checkpoints/'
    os.makedirs(save_path, exist_ok=True)
    stopper = EarlyStopping(
        save_path=save_path, model_name=args.decoder+"_"+args.name, patience=args.patience)

    # check cuda
    if args.gpu >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    args.num_entities=num_nodes
    args.num_relations=num_rels * 2 + 1
    # create model
    model = build_model(args.model_name).build_model_from_args(
        args).model
    # model = WGCN(num_entities=num_nodes,
    #              num_relations=num_rels * 2 + 1, args=args)

    # build graph
    g = dgl.graph([])
    g.add_nodes(num_nodes)
    src, rel, dst = train_data.transpose()
    # add reverse edges, reverse relation id is between [num_rels, 2*num_rels)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    # get new train_data with reverse relation
    train_data_new = np.stack((src, rel, dst)).transpose()

    # unique train data by (h,r)
    train_data_new_pandas = pandas.DataFrame(train_data_new)
    train_data_new_pandas = train_data_new_pandas.drop_duplicates([0, 1])
    train_data_unique = np.asarray(train_data_new_pandas)

    if not args.wni:
        if args.rat:
            if args.ss > 0:
                high = args.ss
            else:
                high = num_nodes
            g.add_edges(src, np.random.randint(
                low=0, high=high, size=dst.shape))
        else:
            g.add_edges(src, dst)

    # add graph self loop
    if not args.wsi:
        g.add_edges(g.nodes(), g.nodes())
        # add self loop relation type, self loop relation's id is 2*num_rels.
        if args.wni:
            rel = np.ones([num_nodes]) * num_rels * 2
        else:
            rel = np.concatenate((rel, np.ones([num_nodes]) * num_rels * 2))
    print(g)
    entity_id = torch.LongTensor([i for i in range(num_nodes)])

    model = model.to(device)
    g = g.to(device)
    all_rel = torch.LongTensor(rel).to(device)
    entity_id = entity_id.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # process the triples and get all tails corresponding to (h,r)
    # here valid_dict and test_dict are not used.
    train_dict, valid_dict, test_dict, all_dict = preprocess_data(
        train_data, valid_data, test_data, num_rels)

    train_batch_prepare = TrainBatchPrepare(train_dict, num_nodes)

    # eval needs to use all the data in train_data, valid_data and test_data
    eval_batch_prepare = EvalBatchPrepare(all_dict, num_rels)

    train_dataloader = DataLoader(
        dataset=train_data_unique,
        batch_size=args.batch_size,
        collate_fn=train_batch_prepare.get_batch,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        collate_fn=eval_batch_prepare.get_batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        collate_fn=eval_batch_prepare.get_batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    # training loop
    print("start training...")
    for epoch in range(args.n_epochs):
        model.train()
        epoch_start_time = time.time()
        for step, batch_tuple in enumerate(train_dataloader):
            e1_batch, rel_batch, labels_one_hot = batch_tuple
            e1_batch = e1_batch.to(device)
            rel_batch = rel_batch.to(device)
            labels_one_hot = labels_one_hot.to(device)
            labels_one_hot = ((1.0 - 0.1) * labels_one_hot) + \
                (1.0 / labels_one_hot.size(1))

            pred = model.forward(g, all_rel, e1_batch, rel_batch, entity_id)
            optimizer.zero_grad()
            loss = model.loss(pred, labels_one_hot)
            loss.backward()
            optimizer.step()

        logger.info("epoch : {}".format(epoch))
        logger.info("epoch time: {:.4f}".format(
            time.time() - epoch_start_time))
        logger.info("loss: {}".format(loss.data))

        model.eval()
        if epoch % args.eval_every == 0:
            with torch.no_grad():
                val_mrr = ranking_and_hits(
                    g, all_rel, model, valid_dataloader, 'dev_evaluation', entity_id, device, logger)
            if stopper.step(val_mrr, model):
                break

    print("training done")
    model.load_state_dict(torch.load(os.path.join(
        save_path, args.decoder+"_"+args.name+'.pt')))
    ranking_and_hits(g, all_rel, model, test_dataloader,
                     'test_evaluation', entity_id, device, logger)


if __name__ == '__main__':
    print("2")
    # parser = argparse.ArgumentParser(description='WGCN')
    # parser.add_argument("--seed", type=int, default=12345, help="random seed")
    # parser.add_argument("--init_emb_size", type=int, default=100,
    #                     help="initial embedding size")
    # parser.add_argument("--gc1_emb_size", type=int, default=150,
    #                     help="embedding size after gc1")
    # parser.add_argument("--embedding_dim", type=int, default=200,
    #                     help="embedding dim")
    # parser.add_argument("--input_dropout", type=float, default=0,
    #                     help="input feature dropout")
    # parser.add_argument("--dropout_rate", type=float, default=0.2,
    #                     help="dropout rate")
    # parser.add_argument("--channels", type=int, default=200,
    #                     help="number of channels")
    # parser.add_argument("--kernel_size", type=int, default=5,
    #                     help="kernel size")
    # parser.add_argument("--gpu", type=int, default=0,
    #                     help="gpu")
    # parser.add_argument("--lr", type=float, default=0.002,
    #                     help="learning rate")
    # parser.add_argument("--n-epochs", type=int, default=5000,
    #                     help="number of training epochs")
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument("--eval-every", type=int, default=1)
    # parser.add_argument("-d", "--dataset", type=str, default="FB15k-237",
    #                     help="dataset to use")
    # parser.add_argument("--batch-size", type=int, default=128,
    #                     help="batch size")
    # parser.add_argument("--patience", type=int, default=100,
    #                     help="early stop patience")
    # parser.add_argument("--decoder", type=str, default='distmult',
    #                     help="decoders")
    # parser.add_argument("--gamma", type=float, default=9.0,
    #                     help="gamma for transe training")
    # parser.add_argument("--name", type=str, default="debug",
    #                     help="model name")
    # parser.add_argument("--n-layer", type=int, default=2,
    #                     help="number of gcn layers")
    # parser.add_argument("--rat", action="store_true",
    #                     default=False, help='random adjacency matrices')
    # parser.add_argument("--wsi", action="store_true",
    #                     default=False, help='without self loop')
    # parser.add_argument("--wni", action="store_true",
    #                     default=False, help='without neighbor information')
    # parser.add_argument("--ss", type=int, default=-1, help="sample size")
    #
    # parser.add_argument("--final_bn", '-fb',
    #                     action="store_true", default=False)
    # parser.add_argument("--final_act", '-fa',
    #                     action="store_true", default=False)
    # parser.add_argument("--final_drop", '-fd',
    #                     action="store_true", default=False)
    #
    # args = parser.parse_args()




