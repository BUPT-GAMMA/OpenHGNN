# -*- coding: utf-8 -*-
"""
code from github.com/JD-AI-Research-Silicon-Valley/SACN
"""

import torch
import numpy as np
import time


def ranking_and_hits(g, v, model, dev_rank_batcher, name, entity_id, device, logger):
    print('')
    print(name)
    print('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])
    # with open('output_model2.txt', 'w') as file:
    for i, batch_tuple in enumerate(dev_rank_batcher):
        # print("evaluation batch {}".format(i))
        e1, e2, rel, rel_reverse, e2_multi1, e2_multi2 = batch_tuple
        e1 = e1.to(device)
        e2 = e2.to(device)
        rel = rel.to(device)
        rel_reverse = rel_reverse.to(device)

        pred1 = model.forward(g, v, e1, rel, entity_id)
        pred2 = model.forward(g, v, e2, rel_reverse, entity_id)
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        # e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data

        batch_score_start_time = time.time()
        for i in range(len(e2_multi1)):
            # these filters contain ALL labels
            filter1 = e2_multi1[i]
            filter2 = e2_multi2[i]

            # save the prediction that is relevant
            target_value1 = pred1[i, e2.cpu().numpy()[i, 0].item()].item()
            target_value2 = pred2[i, e1.cpu().numpy()[i, 0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2

        batch_sort_and_rank_start_time = time.time()
        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(len(e2_multi1)):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i] == e2.cpu().numpy()[i, 0])[0][0]
            rank2 = np.where(argsort2[i] == e1.cpu().numpy()[i, 0])[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)
    logger.info('MRR: {0}, MRR left: {1}, MRR right: {2}'.format(
        np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))
    logger.info('MR: {0}, MR left: {1}, MR right: {2}'.format(
        np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
    for i in [0, 2, 9]:
        logger.info('Hits @{0}: {1}, Hits left @{0}: {2}, Hits right @{0}: {3}'.format(
            i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))

    print('-' * 50)
    return np.mean(1. / np.array(ranks))
