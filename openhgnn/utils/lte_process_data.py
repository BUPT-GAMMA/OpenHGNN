from collections import defaultdict as ddict


def process(dataset, num_rel):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """
    sr2o = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
        sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            sr2o[(subj, rel)].add(obj)
            sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    triplets = ddict(list)

    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
            triplets[f"{split}_head"].append(
                {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    triplets = dict(triplets)
    return triplets
