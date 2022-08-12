# TODO: This file should be deprecated, it's just for testing.
# TODO: TEST!!!
import sys
import os
sys.path.append('/home/ubuntu/openhgnn/')
os.chdir('/home/ubuntu/openhgnn/')

from openhgnn.models.MAGNN import MAGNN
from openhgnn.sampler.MAGNN_sampler import MAGNN_sampler, collate_fn
from openhgnn.sampler.test_config import CONFIG
import argparse
import warnings
import time
import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        th.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

def load_hg(args):
    hg_dir = 'openhgnn/dataset/'
    hg,_ = dgl.load_graphs(hg_dir+'{}/graph.bin'.format(args.dataset), [0])
    hg = hg[0]
    return hg

def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    # This method is implemented by author
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append([np.mean(macro_f1_list), np.std(macro_f1_list)])
        result_micro_f1_list.append([np.mean(micro_f1_list), np.std(micro_f1_list)])
    return result_macro_f1_list, result_micro_f1_list

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    args = argparse.Namespace(**CONFIG)
    hg = load_hg(args)
    train_mask = hg.nodes[args.category].data['train_mask']
    val_mask = hg.nodes[args.category].data['val_mask']
    test_mask = hg.nodes[args.category].data['test_mask']
    model = MAGNN.build_model_from_args(args, hg)
    print(args)

    sampler = MAGNN_sampler(g=hg, mask=train_mask.cpu().numpy(), num_layers=args.num_layers, category=args.category,
                            metapath_list=model.metapath_list, num_samples=args.num_samples, dataset_name=args.dataset)

    dataloader = DataLoader(dataset=sampler, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=collate_fn, drop_last=False)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stop = EarlyStopping(patience=args.patience, verbose=True,
                               save_path='/home/ubuntu/openhgnn/openhgnn/sampler/checkpoint/checkpoint_dblp.pt')
    model = model.to(args.device)

    for epoch in range(args.max_epoch):
        sampler.mask = train_mask
        t = time.perf_counter()
        model.train()
        print("...Start the mini batch training...")
        for num_iter, (sub_g, mini_mp_inst, seed_nodes) in enumerate(dataloader):
            print("Sampling {} seed_nodes with duration(s): {}".format(len(seed_nodes[args.category]), time.perf_counter() - t))
            model.mini_reset_params(mini_mp_inst)
            sub_g = sub_g.to(args.device)
            pred, _ = model(sub_g)
            pred = pred[args.category][seed_nodes[args.category]]
            lbl = sub_g.nodes[args.category].data['labels'][seed_nodes[args.category]]
            loss = F.cross_entropy(pred, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Iter:{}, the training loss of this batch is {}".format(
                num_iter, loss.item()
            ))
            t = time.perf_counter()

        print()
        model.eval()
        with th.no_grad():
            sampler.mask = val_mask
            val_loader = DataLoader(dataset=sampler, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)

            logp_val_all = []
            embed_val_all = []
            for num_iter, (sub_g, mini_mp_inst, seed_nodes) in enumerate(val_loader):
                sub_g = sub_g.to(args.device)
                model.mini_reset_params(mini_mp_inst)
                pred_val, embed_val = model(sub_g)
                pred_val = pred_val[args.category][seed_nodes[args.category]]
                embed_val = embed_val[args.category][seed_nodes[args.category]]
                logp_val = F.log_softmax(pred_val, 1)
                logp_val_all.append(logp_val)
                embed_val_all.append(embed_val.cpu().numpy())

            lbl_val = hg.nodes[args.category].data['labels'][val_mask]
            lbl_val = lbl_val.cuda()
            embed_val_all = np.concatenate(embed_val_all, 0)
            loss_val = F.nll_loss(th.cat(logp_val_all, 0), lbl_val)
            lbl_val = lbl_val.cpu().numpy()
            val_f1_macro_list, val_f1_micro_list = svm_test(embed_val_all, lbl_val)

            print('Epoch {}. val_loss is {}'.format(epoch, loss_val))
            print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
                macro_f1[0], macro_f1[1], train_size) for macro_f1, train_size in
                zip(val_f1_macro_list, [0.8, 0.6, 0.4, 0.2])]))
            print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
                micro_f1[0], micro_f1[1], train_size) for micro_f1, train_size in
                zip(val_f1_micro_list, [0.8, 0.6, 0.4, 0.2])]))
            early_stop(loss_val, model)
            if early_stop.early_stop:
                print("Early Stopping!")
                break

    print('----------TEST-----------')
    model.eval()  # Test on full graphprint()
    with th.no_grad():
        sampler.mask = test_mask
        test_loader = DataLoader(dataset=sampler, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)

        logp_test_all = []
        embed_test_all = []
        for num_iter, (sub_g, mini_mp_inst, seed_nodes) in enumerate(test_loader):
            sub_g = sub_g.to(args.device)
            model.mini_reset_params(mini_mp_inst)
            pred_test, embed_test = model(sub_g)
            pred_test = pred_test[args.category][seed_nodes[args.category]]
            embed_test = embed_test[args.category][seed_nodes[args.category]]
            logp_test = F.log_softmax(pred_test, 1)
            logp_test_all.append(logp_test)
            embed_test_all.append(embed_test.cpu().numpy())

        lbl_test = hg.nodes[args.category].data['labels'][test_mask]
        lbl_test = lbl_test.cuda()
        embed_test_all = np.concatenate(embed_test_all, 0)
        loss_test = F.nll_loss(th.cat(logp_test_all, 0), lbl_test)
        lbl_test = lbl_test.cpu().numpy()
        test_f1_macro_list, test_f1_micro_list = svm_test(embed_test_all, lbl_test)

        print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
            macro_f1[0], macro_f1[1], train_size) for macro_f1, train_size in
            zip(test_f1_macro_list, [0.8, 0.6, 0.4, 0.2])]))
        print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
            micro_f1[0], micro_f1[1], train_size) for micro_f1, train_size in
            zip(test_f1_micro_list, [0.8, 0.6, 0.4, 0.2])]))
