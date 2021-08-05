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
    sampler = MAGNN_sampler(g=hg, mask=train_mask.cpu().numpy(), n_layers=args.num_layers, category=args.category,
                            metapath_list=model.metapath_list, num_samples=args.num_samples, dataset_name=args.dataset)

    dataloader = DataLoader(dataset=sampler, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            collate_fn=collate_fn, drop_last=False)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(args.device)

    for epoch in range(args.max_epoch):
        t = time.perf_counter()
        model.train()
        print("...Start the mini batch training...")
        for num_iter, (sub_g, mini_mp_inst, seed_nodes) in enumerate(dataloader):
            print("Sampling {} seed_nodes with duration(s): {}".format(len(seed_nodes[args.category]), time.perf_counter() - t))
            model.mini_reset_params(mini_mp_inst)
            sub_g = sub_g.to(args.device)
            # pred = model(sub_g)[args.category][seed_nodes[args.category]]
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
            break
        print()
        model.eval() # Evaluation on full graph
        with th.no_grad():
            model.restore_params()
            hg = hg.to(args.device)
            pred_eval, embed_eval = model(hg)
            pred_eval = pred_eval[args.category][val_mask]
            embed_eval = embed_eval[args.category][val_mask]
            lbl_eval = hg.nodes[args.category].data['labels'][val_mask]
            eval_loss = F.cross_entropy(pred_eval, lbl_eval)
            lbl_eval, embed_eval, pred_eval = \
                lbl_eval.cpu().numpy(), embed_eval.cpu().numpy(), pred_eval.cpu().numpy()
            eval_f1_macro_list, eval_f1_micro_list = svm_test(embed_eval, lbl_eval)
        print('Epoch {}. eval_loss is {}'.format(epoch, eval_loss))
        print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
            macro_f1[0], macro_f1[1], train_size) for macro_f1, train_size in
            zip(eval_f1_macro_list, [0.8, 0.6, 0.4, 0.2])]))
        print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
            micro_f1[0], micro_f1[1], train_size) for micro_f1, train_size in
            zip(eval_f1_micro_list, [0.8, 0.6, 0.4, 0.2])]))

    print('----------TEST-----------')
    model.eval()  # Test on full graph
    with th.no_grad():
        model.restore_params()
        hg = hg.to(args.device)
        pred_test, embed_test = model(hg)
        pred_test = pred_test[args.category][test_mask]
        embed_test = embed_test[args.category][test_mask]
        lbl_test = hg.nodes[args.category].data['labels'][test_mask]
        test_loss = F.cross_entropy(pred_test, lbl_test)
        lbl_test, embed_test, pred_test = \
            lbl_test.cpu().numpy(), embed_test.cpu().numpy(), pred_test.cpu().numpy()
        test_f1_macro_list, test_f1_micro_list = svm_test(embed_test, lbl_test)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[0], macro_f1[1], train_size) for macro_f1, train_size in
        zip(eval_f1_macro_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[0], micro_f1[1], train_size) for micro_f1, train_size in
        zip(eval_f1_micro_list, [0.8, 0.6, 0.4, 0.2])]))






