import torch
import os
from abc import ABC
from ..dataset.THePUffDataset import THePUffDataset
from ..sampler.THePUffSampler import THePUffSampler
from ..models.THePUffModel import THePUffModel
from torch.autograd import Variable
from collections import Counter
import scipy.sparse as sp
from torch.utils.data import DataLoader
import numpy as np
import pickle
import random
from .base_flow import BaseFlow
from openhgnn.trainerflow import register_flow
import networkx as nx
from scipy.sparse.csgraph import connected_components
from torch.nn import CrossEntropyLoss
from operator import itemgetter

@register_flow("THePUffTrainflow")
class THePUffTrainflow(BaseFlow):
    def __init__(self, config):
        self.config = config
        self.train_flow = THePUfftrainflow(config)

    def train(self):
        pass

class THePUfftrainflow(ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.batch_size = 64
        args.gpu_num = 0
        args.W_down_generator_size = 128
        args.noise_dim = 64
        args.noise_type = "Uniform"
        args.hidden_units = 128
        args.num_G_layer = 5
        args.max_path_len = 5
        args.h = 4
        args.N = 1
        args.dropout = 0.2
        args.d_model = 128
        args.d_ff = 128
        args.lr_gen = 1e-4
        args.lr_dis1 = 1e-3
        args.lr_dis2 = 1e-4
        args.n_critic = 5
        args.n_epochs = 5
        args.n_epochs_pre = 10
        args.dataset_name = "ml-100k"
        args.load_d2 = True
        args.load_model = True
        args.force_reload = False
        args.if_raw_dir = False
        args.raw_dir = './data'
        args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        args.dataset_name = args.dataset
        dir = os.getcwd()
        self.if_raw_dir = args.if_raw_dir
        self.dataset = self.args.dataset_name
        self.device = args.device
        self.save_dir = os.path.join(dir, '/data/' + self.dataset + '/')
        os.makedirs(self.save_dir, exist_ok=True)
        self.path = self.save_dir + self.dataset + '_p.pickle'
        self.dp_path = self.save_dir + self.dataset + '_dp_p.pickle'
        self.raw_dir = args.raw_dir if self.if_raw_dir else None
        node_dict, node_dict_dp, node_index, node_type, node_embs, node_embs_dp, real_level_embs, dp_level_embs, node_value, N, node_embs_classified, node_type_classified, M = self._dataset()
        self._sampler(node_dict, node_dict_dp, node_index)
        self.dataloader, self.dp_dataloader = self._dataloader(node_type, node_embs, node_embs_dp, real_level_embs, dp_level_embs, N)
        self._model(N, node_embs_classified, node_type_classified, real_level_embs, node_embs, dp_level_embs, node_embs_dp, node_value, M)
        self.evaluate(self.dataset)

    def _dataset(self):
        data = THePUffDataset(dataset=self.args.dataset_name, force_reload=self.args.force_reload, raw_dir=self.raw_dir, if_dp=False)
        node_dict, node_index, original_node_index, node_value, node_level, node_type = data.get_output()
        dp_data = THePUffDataset(dataset=self.args.dataset_name, force_reload=self.args.force_reload, raw_dir=self.raw_dir, node_index=node_index, node_value=node_value, node_type=node_type, if_dp=True)
        node_dict_dp, node_index, node_value, node_level_dp, node_type = dp_data.get_output()
        N = len(node_index)
        self.args.N_ = N
        M = 0
        for node in node_dict_dp.keys():
            M += len(node_dict_dp[node])
        emb = THePUffDataset(dataset=self.args.dataset_name, force_reload=self.args.force_reload, raw_dir=self.raw_dir, if_dp=False, node_index=node_index, node_value=node_value, node_type=node_type, node_dict=node_dict, node_dict_dp=node_dict_dp, node_level=node_level, node_level_dp=node_level_dp, N=N)
        real_level_embs, dp_level_embs, node_type_classified, node_embs_classified, node_embs, node_embs_dp = emb.embedding()
        return node_dict, node_dict_dp, node_index, node_type, node_embs, node_embs_dp, real_level_embs, dp_level_embs, node_value, N, node_embs_classified, node_type_classified, M

    def _sampler(self, node_dict, node_dict_dp, node_index):
        samp = THePUffSampler(node_dict, node_dict_dp, node_index, self.args.dataset_name)
        samp.sampler()

    def _dataloader(self, node_type, node_embs, node_embs_dp, real_level_embs, dp_level_embs, N):
        walk_data = self.get_walk_data(self.path, N, node_type)
        dp_walk_data = self.get_walk_data(self.dp_path, N, node_type)
        dataloader = DataLoader(self.HTNDataset(walk_data, node_embs, real_level_embs), shuffle=True, batch_size=self.args.batch_size, num_workers=0, drop_last=True)
        dp_dataloader = DataLoader(self.HTNDataset(dp_walk_data, node_embs_dp, dp_level_embs), shuffle=True, batch_size=self.args.batch_size, num_workers=0, drop_last=True)
        return dataloader, dp_dataloader

    def _model(self, N, node_embs_classified, node_type_classified, real_level_embs, node_embs, dp_level_embs, node_embs_dp, node_value, M):
        model = THePUffModel(self.device, self.args, N, node_embs_classified, node_type_classified, self.dataloader, real_level_embs, node_embs, dp_level_embs, node_embs_dp, self.dp_dataloader)
        generator, discriminator1, discriminator2, optimizer_G, optimizer_D1, optimizer_D2 = model.init_model()
        if self.args.load_model == True:
            generator.load_state_dict(torch.load("./data/" + self.dataset + "/" + self.dataset + '_g.pt'))
            discriminator1.load_state_dict(torch.load("./data/" + self.dataset + "/" + self.dataset + '_d1.pt'))
            discriminator2.load_state_dict(torch.load("./data/" + self.dataset + "/" + self.dataset + '_d2.pt'))
            if self.args.load_d2 == False:
                model.train()
        else:
            if self.args.load_d2 == True:
                discriminator2.load_state_dict(torch.load("./data/" + self.dataset + "/" + self.dataset + '_d2.pt'))
            else:
                model.pretrain()
            model.train()
        syn_graph = self.assemble(N, self.args, self.device, generator, M, node_type_classified)
        os.makedirs("./data/" + self.dataset + "/generated/", exist_ok=True)
        self.save_graph("./data/" + self.dataset + "/generated/" + self.dataset + "_gen", syn_graph, node_value)

    def get_walk_data(self, path, N, node_type):
        walk_data = []
        with open(path, 'rb') as f:
            paths = pickle.load(f)
        for path in paths:
            node_classes = 3 if self.args.dataset_name == 'taobao' else 4
            temp_type = self.one_hot_encoder(np.array([node_type[i] for i in path]), node_classes + 1)
            temp_idx = path
            if temp_type.shape[0] < self.args.max_path_len:
                temp_type = self.pad_along_axis(temp_type, self.args.max_path_len, axis=0).astype(np.float32)
                temp_idx = self.pad_list(temp_idx, self.args.max_path_len)
            walk_data.append((temp_type, temp_idx))
        return walk_data

    class HTNDataset(torch.utils.data.Dataset):
        def __init__(self, data, embs, level_embs):
            super().__init__()
            self.data = data
            self.embs = embs
            self.level_embs = level_embs

        def __getitem__(self, item):
            types, idx = self.data[item]
            return types, self.embs[idx], self.level_embs[idx]

        def __len__(self):
            return len(self.data)

    def one_hot_encoder(self, data, max_value):
        shape = (data.size, max_value)
        one_hot = np.zeros(shape)
        rows = np.arange(data.size)
        one_hot[rows, data] = 1
        return one_hot

    def pad_along_axis(self, array, target_length, axis):
        pad_size = target_length - array.shape[axis]
        if pad_size <= 0:
            return array
        padding = np.zeros([pad_size, array.shape[1]])
        for i in range(len(padding)):
            padding[i][-1] = 1
        return np.concatenate([array, padding], axis=axis)

    def pad_list(self, list, target_leagth):
        pad_size = target_leagth - len(list)
        if pad_size <= 0:
            return list
        for i in range(pad_size):
            list.append(-1)
        return list

    def save_graph(self, path, graph, node_value):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + '.txt', 'w') as f:
            for i, j in zip(graph.nonzero()[0], graph.nonzero()[1]):
                node0 = node_value[i]
                node1 = node_value[j]
                time0 = int(node0.split("_")[1])
                time1 = int(node1.split("_")[1])
                f.write(node0 + " " + node1 + " " + node0[0] + " " + node1[0] + " " + str(max(time0, time1)) + "\n")
        with open(path + '.p', 'wb') as f:
            pickle.dump(graph, f)

    def assemble(self, N, args, device, generator, M, node_type_classified):
        smpls_type, smpls_node = self.generate_walks(N, args, device, generator)
        smpls_type_2 = [i for i in smpls_type if i.shape[0] == 2 and 4 not in i]
        smpls_type_3 = [i for i in smpls_type if i.shape[0] == 3 and 4 not in i]
        smpls_type_4 = [i for i in smpls_type if i.shape[0] == 4 and 4 not in i]
        smpls_type_5 = [i for i in smpls_type if i.shape[0] == 5 and 4 not in i]
        smpls_node_2 = [i for i in smpls_node if i.shape[0] == 2 and N not in i]
        smpls_node_3 = [i for i in smpls_node if i.shape[0] == 3 and N not in i]
        smpls_node_4 = [i for i in smpls_node if i.shape[0] == 4 and N not in i]
        smpls_node_5 = [i for i in smpls_node if i.shape[0] == 5 and N not in i]
        meta_path_freq = self.meta_path_frequency(smpls_type_2, smpls_type_3, smpls_type_4, smpls_type_5)
        score_matrix = self.score_matrix_from_random_walks(smpls_node_2, N)
        if len(smpls_node_3) != 0:
            score_matrix += self.score_matrix_from_random_walks(smpls_node_3, N)
        if len(smpls_node_4) != 0:
            score_matrix += self.score_matrix_from_random_walks(smpls_node_4, N)
        if len(smpls_node_5) != 0:
            score_matrix += self.score_matrix_from_random_walks(smpls_node_5, N)
        score_matrix = score_matrix.tocsr()
        syn_graph = self.heterogeneous_graph_assemble(score_matrix, M, meta_path_freq, node_type_classified)
        return syn_graph

    def generate_walks(self, N, args, device, generator):
        transitions_per_walk = 4 - 1
        transitions_per_iter = 20e4
        eval_transitions = 5e7
        sample_many_count = int(np.round(transitions_per_iter / transitions_per_walk))
        n_eval_walks = eval_transitions / transitions_per_walk
        n_eval_iters = int(np.round(n_eval_walks / sample_many_count))
        smpls_type, smpls_node = [], []
        for i in range(n_eval_iters):
            initial_noise = self.make_noise((args.batch_size, args.noise_dim), args.noise_type).to(device)
            fake_result = generator(initial_noise)
            synthetic_type = fake_result['type_seq'].detach()
            synthetic_node = fake_result['node_seq'].detach()
            synthetic_idx = fake_result['idx']
            synthetic_type = torch.argmax(synthetic_type.cpu(), dim=2).numpy().astype(np.int32)
            synthetic_node = torch.argmax(synthetic_node.cpu(), dim=2).numpy().astype(np.int32)
            smpls_type += self.delete_from_tail(synthetic_type, 4)
            smpls_node += self.delete_from_tail(synthetic_node, N)
            print(f"[step 3/3] [Batch {i+1}/{n_eval_iters}]")
        return smpls_type, smpls_node

    def make_noise(self, shape, type="Gaussian"):
        if type == "Gaussian":
            noise = Variable(torch.randn(shape))
        elif type == 'Uniform':
            noise = Variable(torch.randn(shape).uniform_(-1, 1))
        else:
            print("ERROR: Noise type {} not supported".format(type))
        return noise

    def delete_from_tail(self, array, target):
        updated_array = []
        for i in array:
            temp = i.copy()
            while len(temp) > 0 and temp[-1] == target:
                temp = temp[:-1]
            if len(temp) > 0:
                updated_array.append(temp)
        return updated_array

    def meta_path_frequency(self, smpls_type_2, smpls_type_3, smpls_type_4, smpls_type_5):
        len_2 = self.frequent_meta_path_pattern(smpls_type_2)
        len_3 = self.frequent_meta_path_pattern(smpls_type_3)
        len_4 = self.frequent_meta_path_pattern(smpls_type_4)
        len_5 = self.frequent_meta_path_pattern(smpls_type_5)
        len_2, len_3, len_4, len_5 = len_2[:len(len_2) // 2], len_3[:len(len_3) // 2], len_4[:len(len_4) // 2], len_5[:len(len_5) // 2]
        return dict(len_2 + len_3 + len_5)

    def frequent_meta_path_pattern(self, smpls_type):
        bb = []
        for i in smpls_type:
            bb.append(tuple(i))
        count = Counter(bb)
        return count.most_common()

    def score_matrix_from_random_walks(self, random_walks, N, symmetric=True):
        random_walks = np.array(random_walks)
        bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
        bigrams = np.transpose(bigrams, [0, 2, 1])
        bigrams = bigrams.reshape([-1, 2])
        if symmetric:
            bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))
        mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])), shape=[N, N])
        return mat

    def heterogeneous_graph_assemble(self, scores, n_edges, meta_path_freq, node_type):
        if len(scores.nonzero()[0]) < n_edges:
            return self.symmetric(scores) > 0
        target_g = sp.csr_matrix(scores.shape)
        scores_int = scores.toarray().copy()
        scores_int[np.diag_indices_from(scores_int)] = 0
        degrees_int = scores_int.sum(0)
        N = scores.shape[0]
        node_type_degs = []
        for i in range(len(node_type)):
            node_type_degs.append(dict(zip(node_type[i], [degrees_int[i] for i in node_type[i]])))
        count = 0
        with open("log.txt", "a") as f:
            f.write(str(node_type_degs))
        while target_g.sum() <= n_edges:
            sampled_metapath = self.sample_from_dict(meta_path_freq, 1)[0]
            actual_path = []
            for i in range(len(sampled_metapath) - 1):
                if i == 0:
                    n = self.sample_from_dict(node_type_degs[sampled_metapath[i]], 1)[0]
                    actual_path.append(n)
                else:
                    n = actual_path[-1]
                row = scores_int[n, :].copy()
                next_node_type_degs = list(node_type_degs[sampled_metapath[i + 1]].keys())
                row[self.array_subtraction(range(N), next_node_type_degs)] = 0
                probs = row / row.sum()
                try:
                    target = np.random.choice(N, p=probs)
                except ValueError:
                    print(n)
                target_g[n, target] = 1
                target_g[target, n] = 1
                actual_path.append(target)
                count += 1
            if count % 10000 == 0:
                print("Generating {} of {} edges...".format(target_g.sum(), n_edges))
        target_g = self.symmetric(target_g)
        return target_g

    def sample_from_dict(self, d, sample):
        key = random.choices(list(d.keys()), list(d.values()), k=sample)
        return key

    def symmetric(self, directed_adjacency, clip_to_one=True):
        A_symmetric = directed_adjacency + directed_adjacency.T
        if clip_to_one:
            A_symmetric[A_symmetric > 1] = 1
        return A_symmetric

    def array_subtraction(self, x, y):
        return np.array(list(set(x) - set(y)))

    def gaussian_kernel(self, x, y, sigma=1.0):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        d = ((x - y) ** 2).sum(-1)
        return torch.exp(-d / (2 * sigma ** 2))

    def mmd(self, x, y, sigma=4.0):
        x = x.float()
        y = y.float()
        kxx = self.gaussian_kernel(x, x, sigma).mean()
        kyy = self.gaussian_kernel(y, y, sigma).mean()
        kxy = self.gaussian_kernel(x, y, sigma).mean()
        return kxx + kyy - 2 * kxy

    def statistics_triangle_count(self, A_in):
        triangles = nx.triangles(A_in)
        t = np.sum(list(triangles.values())) / 3
        return int(t)

    def metapath_count(self, A, node_type, dataset):
        if dataset == 'ml-100k':
            metapaths = {2: [[0, 1], [0, 3], [1, 2]], 3: [[0, 1, 2]]}
        elif dataset == 'dblp_kdd':
            metapaths = {2: [[0, 1], [0, 2], [1, 3], [1, 1]], 3: [[0, 1, 3], [0, 1, 1], [1, 1, 3], [1, 1, 1]]}
        elif dataset == 'taobao':
            metapaths = {2: [[0, 1], [1, 2]], 3: [[0, 1, 2]]}
        count = {2: [], 3: []}
        A_A = A @ A
        sum_2 = A.sum()
        sum_3 = A_A.sum()
        for metapath in metapaths[2]:
            cnt = 0
            for i in range(A.shape[0]):
                if node_type[i] == metapath[0]:
                    for j in A[i].nonzero()[-1]:
                        if node_type[j] == metapath[1]:
                            cnt += 1
            if cnt != 0: cnt = cnt / sum_2
            count[2].append(cnt)
        count[2].append(1 - sum(count[2]))
        for metapath in metapaths[3]:
            cnt = 0
            for i in range(A.shape[0]):
                if node_type[i] == metapath[0]:
                    for j in A[i].nonzero()[-1]:
                        if node_type[j] == metapath[1]:
                            for t in A[j].nonzero()[-1]:
                                if node_type[t] == metapath[2]:
                                    cnt += 1
            if cnt != 0: cnt = cnt / sum_3
            count[3].append(cnt)
        count[3].append(1 - sum(count[3]))
        return count

    def statistics_LCC(self, A_in):
        unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
        LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
        return LCC

    def evaluation(self, real_A, syn_A, node_type, timestamp, dataset):
        try:
            real_A = sp.csc_matrix(real_A)
        except:
            pass
        try:
            syn_A = sp.csc_matrix(syn_A)
        except:
            pass
        if real_A.shape[0] > syn_A.shape[0]:
            syn_A = sp.csc_matrix((syn_A.data, (syn_A.nonzero()[0], syn_A.nonzero()[1])), shape=real_A.shape)
        lcc_real = self.statistics_LCC(real_A)
        lcc_syn = self.statistics_LCC(syn_A)
        if dataset == 'taobao':
            if not os.path.exists('{}_{}.p'.format(dataset, timestamp)):
                lcc = random.sample(list(lcc_real), 10000)
                node_type = list(itemgetter(*lcc)(node_type))
                pickle.dump(lcc, open('{}_{}.p'.format(dataset, timestamp), 'wb'))
            else:
                lcc = pickle.load(open('{}_{}.p'.format(dataset, timestamp), 'rb'))
                node_type = list(itemgetter(*lcc)(node_type))
        else:
            lcc = range(real_A.shape[0])
        syn_full, real_full = syn_A, real_A
        orig_G_full = nx.from_scipy_sparse_array(real_full)
        syn_G_full = nx.from_scipy_sparse_array(syn_full)
        real_A = real_A[lcc, :][:, lcc]
        syn_A = syn_A[lcc, :][:, lcc]
        orig_G = nx.from_scipy_sparse_array(real_A)
        syn_G = nx.from_scipy_sparse_array(syn_A)
        results = {}
        results["coef"] = [nx.average_clustering(orig_G), nx.average_clustering(syn_G)]
        results["tc"] = [self.statistics_triangle_count(orig_G), self.statistics_triangle_count(syn_G)]
        results["lcc"] = [len(lcc_real), len(lcc_syn)]
        test_degree_sequence = sorted([d for n, d in orig_G.degree()], reverse=True)
        syn_degree_sequence = sorted([d for n, d in syn_G.degree()], reverse=True)
        test_degree_sequence = torch.tensor(test_degree_sequence).unsqueeze(-1)
        syn_degree_sequence = torch.tensor(syn_degree_sequence).unsqueeze(-1)
        results["deg_mmd"] = self.mmd(test_degree_sequence, syn_degree_sequence, sigma=4.0)
        orig_G_edge = set(orig_G_full.edges())
        syn_G_edge = set(syn_G_full.edges())
        intersecting_edges = orig_G_edge & syn_G_edge
        if len(syn_G_edge) == 0:
            results["eo_rate"] = 0
        else:
            results["eo_rate"] = len(intersecting_edges) / len(syn_G_edge)
        results["meta"] = {}
        real_meta = self.metapath_count(real_A, node_type, dataset)
        syn_meta = self.metapath_count(syn_A, node_type, dataset)
        cross_entropy = CrossEntropyLoss()
        results["meta"][2] = cross_entropy(torch.FloatTensor(syn_meta[2]), torch.FloatTensor(real_meta[2]))
        results["meta"][3] = cross_entropy(torch.FloatTensor(syn_meta[3]), torch.FloatTensor(real_meta[3]))
        return results

    def evaluate(self, dataset):
        if dataset == 'ml-100k':
            timerange = 8
        else:
            timerange = 11
        results = {"coef": [], 'tc': [], "lcc": [], "deg_mmd": [], "eo_rate": [], "meta": {2: [], 3: []}}
        syns = {}
        N = -1
        dir = os.getcwd()
        processed_dir = os.path.join(dir, "data", dataset, "processed")
        generated_dir = os.path.join(dir, "data", dataset, "generated")
        with open(generated_dir + "/" + dataset + "_gen.p", 'rb') as f:
            syn_graph = pickle.load(f)
        with open(processed_dir + "/node_index.p", 'rb') as f:
            node_index = pickle.load(f)
        with open(processed_dir + "/node_value.p", 'rb') as f:
            node_value = pickle.load(f)
        with open(processed_dir + "/node_type.p", 'rb') as f:
            node_type = pickle.load(f)
        with open(processed_dir + "/original_node_index.p", 'rb') as f:
            original_node_index = pickle.load(f)
        with open(processed_dir + "/node_dict.p", 'rb') as f:
            node_dict = pickle.load(f)
        for time in range(0, timerange):
            syns[time] = [[], []]
        for i, j in zip(syn_graph.nonzero()[0], syn_graph.nonzero()[1]):
            node1 = node_value[i]
            node2 = node_value[j]
            type1 = node_type[node_index[node1]]
            type2 = node_type[node_index[node2]]
            node1, time1 = node1.split('_')
            node2, time2 = node2.split('_')
            node1_idx = original_node_index[node1]
            node2_idx = original_node_index[node2]
            syns[int(time1)][0].append(node1_idx)
            syns[int(time1)][1].append(node2_idx)
            syns[int(time2)][0].append(node1_idx)
            syns[int(time2)][1].append(node2_idx)
            N = max(N, node1_idx, node2_idx)
        new_node_type = {}
        edges = {}
        nodes = set()
        for node in node_dict:
            node1, time1 = node.split('_')
            time1 = int(time1)
            if time1 not in edges:
                edges[time1] = set()
            type1 = node_type[node_index[node]]
            for (node2, time2) in node_dict[node]:
                assert time1 == time2
                node_ = '{}_{}'.format(node2, time2)
                type2 = node_type[node_index[node_]]
                node1_idx = original_node_index[node1]
                node2_idx = original_node_index[node2]
                if not (node1_idx, node2_idx) in edges[time1]:
                    edges[time1].add((node1_idx, node2_idx))
                    if node1_idx not in nodes:
                        new_node_type[node1_idx] = type1
                    if node2_idx not in nodes:
                        new_node_type[node2_idx] = type2
        node_type = new_node_type
        syn_0 = sp.coo_matrix((np.ones(len(syns[0][0])), (syns[0][0], syns[0][1])), (N + 1, N + 1))
        for timestamp in range(1, timerange):
            syn = sp.coo_matrix((np.ones(len(syns[timestamp][0])), (syns[timestamp][0], syns[timestamp][1])),(N + 1, N + 1))
            syn = syn + syn_0
            edge_time = np.array(list(edges[timestamp]))
            real = sp.csc_matrix((np.ones(edge_time.shape[1]), (edge_time[0, :], edge_time[1, :])), (N + 1, N + 1))
            result = self.evaluation(real, syn, node_type, timestamp, dataset)
            for metric in results:
                if metric == "meta":
                    results[metric][2].append(result[metric][2])
                    results[metric][3].append(result[metric][3])
                else:
                    results[metric].append(result[metric])
        for metric in ["coef", "tc", "lcc"]:
            result = results[metric]
            result = np.array(result).transpose()
            real_ = torch.tensor(result[0, :]).unsqueeze(-1)
            syn_ = torch.tensor(result[1, :]).unsqueeze(-1)
            results[metric] = self.mmd(real_, syn_, sigma=4.0).item()
        for metric in ['deg_mmd', 'eo_rate']:
            results[metric] = np.array(results[metric]).mean()
        for length in results["meta"]:
            results["meta"][length] = np.array(results["meta"][length]).mean()
        print(results, flush=True)