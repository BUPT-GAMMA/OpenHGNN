import dgl
import os
import random
import torch
from dgl.data.utils import load_graphs, download, extract_archive
import torch as th
from . import BaseDataset, register_dataset
from tqdm import tqdm



@register_dataset('abnorm_event_detection')
class AbnormEventDetectionDataset(BaseDataset):
    r"""
    The class *AbnormalEventDetectionDataset* is a class for datasets which can be used in task *abnormal event detection*.

    Attributes
    -------------
    g : dgl.DGLHeteroGraph
        The heterogeneous graph.
    in_dim : int
        Dimension of node features.
    event_features : tensor
        All event features.
    event_mask : tensor
        All event mask.
        It will be used to mask useless features.
    neg_event_features : tensor
        The neg event is event with fewer meta paths to target event.
    pos_event_features : tensor
        The pos event is event with more meta paths to target event.
    neg_context_features : tensor
        The neg context is that context of original event with partially replaced some nodes.
    neg_entity_features : tensor
        The neg entity is composed that node not with edge to target node.
    event_list : list
        The list composed event.
    pos_node_set_dict : dict
        The node's pos node set that node with edge to original node.
    neg_node_set_dict : dict
        The node's neg node set that node not with edge to original node.
    neg_event_set_list : list
        All event's neg event.
    pos_event_set_list : list
        All event's pos event.
    type_max_num : dict
        The max in the number of nodes of an event of a certain type.
    neg_num : int
        The size of neg entity.
    type_label_node_list : list
        All node of a type and a label
    context_type_num : dict
        A number used to represent a type.
    type_num : torch
        The type num will be used to compute embedding.
    max_type_features_len : int
        The max of dimension of all type node features.
    center_type : str
        The center node type from the event.
    context_type : list
        The context node type from the event.
    event_label : tensor
        The event label.
        0 indicates that it is not an abnormal event.
        1 indicates that it is an abnormal event.
    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(AbnormEventDetectionDataset, self).__init__(*args, **kwargs)
        self.in_dim = None
        self.event_features = None
        self.event_mask = None
        self.neg_event_features = None
        self.pos_event_features = None
        self.neg_context_features = None
        self.neg_entity_features = None
        self.event_list = None
        self.pos_node_set_dict = None
        self.neg_node_set_dict = None
        self.neg_event_set_list = []
        self.pos_event_set_list = []
        self.type_max_num = None
        self.neg_num = 10
        self.type_label_node_list = None
        self.context_type_num = None
        self.type_num = None
        self.max_type_features_len = 0
        self.g, self.center_type, self.context_type, self.event_label = self.get_graph(dataset_name)
        print("get graph is Ok")
        self.preprocess()
        print("preprocess is Ok")
        self.get_complete_events_features()
        print("get features is Ok")

    def set_neg_num(self, neg_num):
        self.neg_num = neg_num

    def get_batch(self, batch_size, shuffle: bool = True, device='cpu'):
        """
        Get all batches dataset from the tensor dataset.

        Parameters
        ----------
        batch_size
            Size of each batch.
        shuffle
            True: use random shuffle.
            False: use original shuffle.
        device
            Use cpu or gpu to composed.

        Returns
        -------
            A list is composed all batch.
        """
        event_len = self.event_features[self.center_type].shape[0]
        shuffle_list = [n for n in range(event_len)]
        if shuffle:
            random.shuffle(shuffle_list)
        event_batch = self.get_batch_sample(batch_size, self.event_features, event_len, shuffle_list, device=device)
        neg_event_batch = self.get_batch_sample(batch_size, self.neg_event_features, event_len, shuffle_list,
                                                device=device)
        pos_event_batch = self.get_batch_sample(batch_size, self.pos_event_features, event_len, shuffle_list,
                                                device=device)
        neg_context_batch = self.get_batch_sample(batch_size, self.neg_context_features, event_len, shuffle_list,
                                                  device=device)
        neg_entity_batch = self.get_batch_sample(batch_size, self.neg_entity_features, event_len, shuffle_list,
                                                 device=device)
        event_mask = self.get_batch_mask(batch_size, self.event_mask, event_len, shuffle_list, device=device)

        type_num = self.get_batch_type_num(batch_size, self.type_num, event_len, shuffle_list, device=device)

        return event_batch, neg_event_batch, pos_event_batch, neg_context_batch, neg_entity_batch, event_mask, type_num

    def get_batch_type_num(self, batch_size, type_num: dict, event_len, shuffle_list, device='cpu'):
        type_num_batch_list = []
        type_num_batch_num = int(event_len / batch_size)
        type_num_ = dict()
        for key in type_num.keys():
            type_num_[key] = type_num[key][shuffle_list]
        for i in range(type_num_batch_num):
            type_num_batch = dict()
            for key in type_num.keys():
                type_num_batch[key] = type_num_[key][i * batch_size:(i + 1) * batch_size, :].to(device)
            type_num_batch_list.append(type_num_batch)
        if batch_size * type_num_batch_num < event_len:
            type_num_batch = dict()
            for key in type_num.keys():
                type_num_batch[key] = type_num_[key][type_num_batch_num * batch_size:, :].to(device)
            type_num_batch_list.append(type_num_batch)

        return type_num_batch_list


    def get_batch_mask(self, batch_size, mask: dict, event_len, shuffle_list, device='cpu'):
        mask_batch_list = []
        mask_batch_num = int(event_len / batch_size)
        mask_ = dict()
        for key in mask.keys():
            mask_[key] = mask[key][shuffle_list]
        for i in range(mask_batch_num):
            mask_batch = dict()
            for key in mask.keys():
                mask_batch[key] = mask_[key][i * batch_size:(i + 1) * batch_size, :].to(device)
            mask_batch_list.append(mask_batch)

        if batch_size * mask_batch_num < event_len:
            mask_batch = dict()
            for key in mask.keys():
                mask_batch[key] = mask_[key][mask_batch_num * batch_size:, :].to(device)
            mask_batch_list.append(mask_batch)

        return mask_batch_list

    def get_batch_sample(self, batch_size, event: dict, event_len, shuffle_list, device='cpu'):
        event_batch_list = []
        event_batch_num = int(event_len / batch_size)
        event_ = dict()
        for key in event.keys():
            event_[key] = event[key][shuffle_list]
        for i in range(event_batch_num):
            event_batch = dict()
            for key in event.keys():
                event_batch[key] = event_[key][i * batch_size:(i + 1) * batch_size, :, :].to(device)
            event_batch_list.append(event_batch)

        if batch_size * event_batch_num < event_len:
            event_batch = dict()
            for key in event.keys():
                event_batch[key] = event_[key][event_batch_num * batch_size:, :, :].to(device)
            event_batch_list.append(event_batch)

        return event_batch_list

    def get_graph(self, name_dataset):
        if name_dataset == 'aminer4AEHCL':
            data_path = './openhgnn/dataset/aminer4aehcl.zip'
            if not os.path.exists(data_path):
                # download file
                download(url='https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/aminer4aehcl.zip', path='./openhgnn/dataset/')
            extract_archive(data_path, './openhgnn/dataset/aminer4aehcl')
            g, label = load_graphs('./openhgnn/dataset/aminer4aehcl/aminer4aehcl.bin')
            g = g[0]
            event_label = label["event_label"]
            center = 'paper'
            context = ['conf', 'author']
        else:
            raise ValueError()
        return g, center, context, event_label

    def get_node_features(self, node):
        if len(node) == 3:
            return self.g.nodes[node[2]].data["features"][node[1]]
        return self.g.nodes[node[0]].data["features"][node[1]]

    def get_node_label(self, node):
        return int(self.g.nodes[node[0]].data["label"][node[1]])

    def preprocess(self):
        """
        Preprocess dataset.
        Changing heterogeneous graph dataset to that lists or sets are composed node.

        Returns
        -------
            None
        """
        self.context_type_num = dict()
        type_num = 0
        for tp in self.context_type:
            type_num += 1
            self.context_type_num[tp] = type_num

        self.event_list = []
        self.type_max_num = dict()
        events_context_type_node = []
        center_node_number = self.g.num_nodes(self.center_type)

        with tqdm(range(center_node_number), desc='get event') as tbar:
            for i in tbar:
                event = [(self.center_type, i)]
                edge_types_ = self.g.canonical_etypes
                event_context_type_node = dict()
                for edge_type_ in edge_types_:
                    edge_type = edge_type_[1]
                    node_type = edge_type_[2]
                    node_list = self.g.out_edges(i, etype=edge_type)[1]
                    # print(node_list)
                    event_context_type_node[node_type] = node_list.tolist()
                    for j in node_list:
                        event.append((node_type, int(j)))
                events_context_type_node.append(event_context_type_node)
                self.event_list.append(event)

        with tqdm(range(center_node_number), desc='get type num') as tbar:
            for i in tbar:
                event = self.event_list[i]
                type_num = dict()
                for node in event:
                    if node[0] not in type_num.keys():
                        type_num[node[0]] = 1
                    else:
                        type_num[node[0]] += 1
                for key in type_num.keys():
                    if key not in self.type_max_num.keys():
                        self.type_max_num[key] = type_num[key]
                    else:
                        if type_num[key] > self.type_max_num[key]:
                            self.type_max_num[key] = type_num[key]

        # get event's positive and negative event
        have_the_node_dict = dict()
        with tqdm(range(center_node_number), desc='process pos and neg event') as tbar:
            for i in tbar:
                for key in events_context_type_node[i].keys():
                    if key not in have_the_node_dict.keys():
                        have_the_node_dict[key] = dict()
                    for node_num in events_context_type_node[i][key]:
                        if node_num not in have_the_node_dict[key].keys():
                            have_the_node_dict[key][node_num] = []
                        have_the_node_dict[key][node_num].append(i)

        with tqdm(range(center_node_number), desc='get pos and neg event') as tbar:
            for i in tbar:
                pos_event_num_set = set()
                neg_event_num_set = set()
                meta_times = dict()
                for j in range(1000):
                    can_use = False
                    while can_use is False:
                        num = random.randint(0, center_node_number - 1)
                        can_use = True
                        for key in events_context_type_node[i].keys():
                            if len(set(events_context_type_node[i][key]) & set(events_context_type_node[num][key])) > 0:
                                can_use = False
                        if num in neg_event_num_set:
                            can_use = False
                    neg_event_num_set.add(num)

                for key in events_context_type_node[i].keys():
                    for node_num in events_context_type_node[i][key]:
                        for num in have_the_node_dict[key][node_num]:
                            if num != i:
                                if num not in meta_times.keys():
                                    meta_times[num] = 1
                                else:
                                    meta_times[num] = meta_times[num] + 1
                maxn = 0
                for key, value in meta_times.items():
                    if value > maxn:
                        maxn = value
                if maxn == 0:
                    for j in range(1000):
                        can_use = False
                        while can_use is False:
                            num = random.randint(0, center_node_number - 1)
                            can_use = True
                            if num in pos_event_num_set:
                                can_use = False
                        pos_event_num_set.add(num)
                else:
                    for key, value in meta_times.items():
                        if value == maxn:
                            pos_event_num_set.add(key)

                self.neg_event_set_list.append(neg_event_num_set)
                self.pos_event_set_list.append(pos_event_num_set)

        all_type_set = dict()
        relation_type = dict()
        self.pos_node_set_dict = dict()
        with tqdm(range(center_node_number), desc='get pos node') as tbar:
            for i in tbar:
                event = self.event_list[i]
                for node in event:
                    if node[0] not in all_type_set.keys():
                        all_type_set[node[0]] = set()
                    if node[0] not in self.pos_node_set_dict.keys():
                        self.pos_node_set_dict[node[0]] = dict()
                    if node[1] not in self.pos_node_set_dict[node[0]].keys():
                        self.pos_node_set_dict[node[0]][node[1]] = set()
                    all_type_set[node[0]].add(node)
                    for node1 in event:
                        if node != node1:
                            if node[0] not in relation_type.keys():
                                relation_type[node[0]] = set()
                            relation_type[node[0]].add(node1[0])
                            self.pos_node_set_dict[node[0]][node[1]].add(node1)

        self.neg_node_set_dict = dict()
        for key in all_type_set.keys():
            self.neg_node_set_dict[key] = dict()
            with tqdm(all_type_set[key], desc='get neg node '+key) as tbar:
                for node in tbar:
                    self.neg_node_set_dict[key][node[1]] = set()
                    for tp in relation_type[key]:
                        for node1 in all_type_set[tp]:
                            if node != node1 and (node1 not in self.pos_node_set_dict[node[0]][node[1]]):
                                self.neg_node_set_dict[key][node[1]].add(node1)
                    if len(self.neg_node_set_dict[key][node[1]]) > 1000:
                        temp = random.sample(self.neg_node_set_dict[key][node[1]], 1000)
                        del self.neg_node_set_dict[key][node[1]]
                        self.neg_node_set_dict[key][node[1]] = set(temp)

        self.max_type_features_len = self.g.nodes[self.center_type].data['features'].shape[1]
        for tp in self.context_type:
            if self.g.nodes[tp].data['features'].shape[1] > self.max_type_features_len:
                self.max_type_features_len = self.g.nodes[tp].data['features'].shape[1]

        self.type_label_node_list = dict()
        for key in self.context_type:
            self.type_label_node_list[key] = dict()
            key_size = self.g.num_nodes(key)
            for i in range(key_size):
                label = self.get_node_label((key, i))
                if label not in self.type_label_node_list[key].keys():
                    self.type_label_node_list[key][label] = []
                self.type_label_node_list[key][label].append((key, i))


    def get_complete_events_features(self):
        """
        Get tensor datasets from the node's sets or lists.

        Returns
        -------
            None
        """
        self_events = []
        neg_events = []
        pos_events = []
        neg_contexts = []
        neg_entities = []

        len_ = len(self.event_list)
        for i in range(len(self.event_list)):
            self_event = self.event_list[i]

            neg_event_num = random.sample(self.neg_event_set_list[i], 1)[0]
            neg_event = self.event_list[neg_event_num]

            pos_event_num = random.sample(self.pos_event_set_list[i], 1)[0]
            pos_event = self.event_list[pos_event_num]

            neg_entity = []
            for node in self_event:
                nd_list = random.sample(self.neg_node_set_dict[node[0]][node[1]], self.neg_num)
                ndd_list = []
                for nd in nd_list:
                    ndd_list.append(nd)
                neg_entity.append((node[0], ndd_list))

            neg_context = []
            type_context = dict()
            for node in self_event:
                if node[0] == self.center_type:
                    neg_context.append(node)
                else:
                    if node[0] not in type_context.keys():
                        type_context[node[0]] = []
                    type_context[node[0]].append(node)
            for key in type_context.keys():
                nd = random.sample(type_context[key], 1)[0]
                nd_label = self.get_node_label(nd)
                for nd_other in type_context[key]:
                    if nd_other != nd:
                        neg_context.append(nd_other)
                neg_nd_label = random.sample(self.type_label_node_list[key].keys(), 1)[0]
                neg_nd = random.sample(self.type_label_node_list[key][neg_nd_label], 1)[0]
                while neg_nd in type_context[key] or neg_nd_label == nd_label:
                    # print(key,": ","get neg node")
                    neg_nd_label = random.sample(self.type_label_node_list[key].keys(), 1)[0]
                    neg_nd = random.sample(self.type_label_node_list[key][neg_nd_label], 1)[0]
                neg_context.append(neg_nd)

            self_events.append(self_event)
            neg_events.append(neg_event)
            pos_events.append(pos_event)
            neg_contexts.append(neg_context)
            neg_entities.append(neg_entity)

        event_features, self.event_mask, self.type_num = self.get_events_features(self_events)
        in_dim = event_features[self.center_type].shape[-1]
        self.in_dim = in_dim
        self.event_features = event_features
        self.neg_event_features, _, _ = self.get_events_features(neg_events)
        self.pos_event_features, _, _ = self.get_events_features(pos_events)
        self.neg_context_features, _, _ = self.get_events_features(neg_contexts)
        self.neg_entity_features = self.get_entities_features(neg_entities)

    def get_entities_features(self, entities):
        entities_features = dict()
        for entity in entities:
            entity_features = dict()
            entity_features[self.center_type] = []
            for tp in self.context_type:
                entity_features[tp] = []
            for tp, node_list in entity:
                the_features = []
                for nd in node_list:
                    features = self.get_node_features(nd).tolist()
                    features += [0.] * (self.max_type_features_len - len(features))
                    the_features.append(features)
                entity_features[tp].append(the_features)
            for key in entity_features.keys():
                entity_features[key] += [[[0.] * self.max_type_features_len] * self.neg_num] * (
                            self.type_max_num[key] - len(entity_features[key]))
                if key not in entities_features.keys():
                    entities_features[key] = []
                entities_features[key].append(entity_features[key])
        for key in entities_features.keys():
            entities_features[key] = torch.tensor(entities_features[key])
        return entities_features

    def get_events_features(self, events):
        events_features = dict()
        masks = dict()
        type_nums = dict()
        for event in events:
            event_features = dict()
            event_features[self.center_type] = []
            mask = dict()
            mask[self.center_type] = []
            type_num = dict()
            for tp in self.context_type:
                event_features[tp] = []
                type_num[tp] = []
                mask[tp] = []
            for node in event:
                features = self.get_node_features(node).tolist()
                features += [0.] * (self.max_type_features_len - len(features))
                event_features[node[0]].append(features)
            for key in event_features:
                mask[key] = [1.] * len(event_features[key]) + [0.] * (self.type_max_num[key] - len(event_features[key]))
                if key != self.center_type:
                    type_num[key] = [self.context_type_num[key]] * len(event_features[key]) + [0] * (self.type_max_num[key] - len(event_features[key]))
                event_features[key] += [[0.] * self.max_type_features_len] * (
                            self.type_max_num[key] - len(event_features[key]))
                if key not in events_features.keys():
                    events_features[key] = []
                    masks[key] = []
                    if key != self.center_type:
                        type_nums[key] = []
                if key != self.center_type:
                    type_nums[key].append(type_num[key])
                events_features[key].append(event_features[key])
                masks[key].append(mask[key])
        for key in events_features.keys():
            events_features[key] = torch.FloatTensor(events_features[key])
            masks[key] = torch.FloatTensor(masks[key])
            if key != self.center_type:
                type_nums[key] = torch.LongTensor(type_nums[key])
        return events_features, masks, type_nums
