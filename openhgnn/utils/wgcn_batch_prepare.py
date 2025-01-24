import torch
import numpy as np

class EvalBatchPrepare(object):
    def __init__(self, eval_dict, num_rels):
        # eval_dict uses all the data in train, valid, and test
        self.eval_dict = eval_dict
        self.num_rels = num_rels

    def get_batch(self, batch_trip):
        batch_trip = np.asarray(batch_trip)
        e1_batch = batch_trip[:, 0]
        rel_batch = batch_trip[:, 1]
        e2_batch = batch_trip[:, 2]
        # reversed relation id is `rel + num_rels`
        rel_reverse_batch = rel_batch + self.num_rels

        head_to_multi_tail_list = []
        tail_to_multi_head_list = []
        keys1 = list(zip(e1_batch, rel_batch))
        keys2 = list(zip(e2_batch, rel_reverse_batch))
        # get (h,r)'s tails
        for key in keys1:
            cur_tail_id_list = list(self.eval_dict.get(key))
            head_to_multi_tail_list.append(np.asarray(cur_tail_id_list))
        # get (t,r_reverse)'s heads
        for key in keys2:
            cur_tail_id_list = list(self.eval_dict.get(key))
            tail_to_multi_head_list.append(np.asarray(cur_tail_id_list))

        e1_batch = torch.from_numpy(e1_batch).reshape(-1, 1)
        e2_batch = torch.from_numpy(e2_batch).reshape(-1, 1)
        rel_batch = torch.from_numpy(rel_batch).reshape(-1, 1)
        rel_reverse_batch = torch.from_numpy(rel_reverse_batch).reshape(-1, 1)

        return e1_batch, e2_batch, rel_batch, rel_reverse_batch, head_to_multi_tail_list, tail_to_multi_head_list


class TrainBatchPrepare(object):
    def __init__(self, train_dict, num_nodes):
        self.entity_num = num_nodes
        self.train_dict = train_dict

    def get_batch(self, batch_trip):
        # batch_trip shape is (batch_size, 3)
        batch_trip = np.asarray(batch_trip)
        e1_batch = batch_trip[:, 0]
        rel_batch = batch_trip[:, 1]
        keys = list(zip(e1_batch, rel_batch))

        # get (h,r) corresponding tails, convert them to one-hot label
        labels_one_hot = np.zeros((batch_trip.shape[0], self.entity_num), dtype=np.float32)
        cur_row = 0
        for key in keys:
            indices = list(self.train_dict.get(key))
            labels_one_hot[cur_row][indices] = 1
            cur_row += 1

        e1_batch = torch.from_numpy(e1_batch).reshape(-1, 1)
        rel_batch = torch.from_numpy(rel_batch).reshape(-1, 1)
        labels_one_hot = torch.from_numpy(labels_one_hot)

        return e1_batch, rel_batch, labels_one_hot
