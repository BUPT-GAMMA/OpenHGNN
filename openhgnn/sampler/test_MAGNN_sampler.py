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
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

def load_hg(args):
    hg_dir = 'openhgnn/dataset/'
    hg,_ = dgl.load_graphs(hg_dir+'{}/graph.bin'.format(args.dataset), [0])
    hg = hg[0]
    return hg

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
            pred = model(sub_g)[args.category][seed_nodes[args.category]]
            lbl = sub_g.nodes[args.category].data['labels'][seed_nodes[args.category]]
            loss = F.cross_entropy(pred, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Iter:{}, the batch_size is {}, the loss of this batch is {}\n".format(
                num_iter, args.batch_size, loss.item()
            ))
            t = time.perf_counter()
        print()
        model.eval() # Evaluation on full graph
        with th.no_grad():
            model.restore_params()
            hg = hg.to(args.device)
            pred_eval = model(hg)[args.category][val_mask]
            lbl_eval = hg.nodes[args.category].data['labels'][val_mask]
            loss_eval = F.cross_entropy(pred_eval, lbl_eval)

            lbl_eval, pred_eval = lbl_eval.cpu().numpy(), pred_eval.cpu().numpy().argmax(axis=1)
            eval_f1_macro = f1_score(lbl_eval, pred_eval, average='macro')
            eval_f1_micro = f1_score(lbl_eval, pred_eval, average='micro')
        print("Epoch: {}, eval_loss: {}, eval_f1_mac: {}, eval_f1_mic: {}"
              .format(epoch, loss_eval, eval_f1_macro, eval_f1_micro))





