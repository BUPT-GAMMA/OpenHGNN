import os
from dgl.data.utils import download, extract_archive
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from . import register_dataset, BaseDataset
from ..layers import to_tensor


class QGRLdataset(Dataset):
    prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {

    }

    def __init__(self, name, is_train=True):
        super(QGRLdataset, self).__init__()
        self.name = name
        self.data_path = "./openhgnn/dataset/QGRLData.rar"
        self.is_train = is_train
        self.g = None
        self.meta_paths = None
        self.meta_paths_dict = None
        self.load_data()

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))

    def load_data(self):
        if self.name == 'zoo':
            fea, lab = load_zoo()
            p = pm(fea, lab, no_nom_att=16, no_ord_att=0, no_num_att=0)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'iris':
            fea, lab = load_iris()
            p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=4)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'wine':
            fea, lab = load_wine()
            p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=13)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'heart':
            fea, lab = load_heart()
            p = pm(fea, lab, no_nom_att=5, no_ord_att=0, no_num_att=7)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'ttt':
            fea, lab = load_ttt()
            p = pm(fea, lab, no_nom_att=9, no_ord_att=0, no_num_att=0)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'yeast':
            fea, lab = load_yeast()
            p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=8)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'breast':
            fea, lab = load_BC()
            p = pm(fea, lab, no_nom_att=5, no_ord_att=4, no_num_att=0)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'glass':
            fea, lab = load_glass()
            p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=9)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'aa':
            fea, lab = load_AA()
            p = pm(fea, lab, no_nom_att=7, no_ord_att=0, no_num_att=2)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        elif self.name == 'mm':
            fea, lab = load_mammographic()
            p = pm(fea, lab, no_nom_att=4, no_ord_att=0, no_num_att=1)
            self.fea = p.x
            self.adj = p.adjacent_matrix
            self.lab = p.label
        else:
            # 提供一个玩具数据集作为示例
            fea = np.array([[0, 1, 0.],
                            [1, 0, 0.36],
                            [1, 1, 0.64],
                            [2, 1, 0.61],
                            [2, 2, 1.0]])
            lab = np.array([0, 0, 0, 0, 0])
            p = pm(fea, lab, no_nom_att=1, no_ord_att=1, no_num_att=1)
            self.fea = to_tensor(p.x)
            self.adj = to_tensor(p.adjacent_matrix)
            self.lab = to_tensor(p.label)

    def __getitem__(self, idx):
        return self.fea[idx], self.adj, self.lab[idx]

    def __len__(self):
        return len(self.fea)


@register_dataset("qgrl_dataset")
class qgrl_dataset(BaseDataset):
    def get_data(self, name, is_train):
        return QGRLdataset(name, is_train)


def move_column_to_end(arr, col_index):
    """
    Move a column to the end of a NumPy array.

    Parameters:
    - arr: The input NumPy array.
    - col_index: The index list of the column to be moved to the end.

    Returns:
    - An updated NumPy array with the specified column moved to the end.
    """
    move = []
    for i in col_index:
        column_to_move = arr[:, i]
        move.append(column_to_move)

    move = np.array(move)

    arr = np.delete(arr, col_index, axis=1)

    a = np.insert(arr, arr.shape[1], move, axis=1)

    return a


# ----------------load data-------------------------------

def load_zoo():
    """(16) Column 0-15 (16) is nominal; ordinal is 0; numerical is 0"""
    replace = {'2':'a', '4':'b', '5':'c', '6':'d', '8':'e'}
    path = './openhgnn/dataset/QGRLdata/zoo/'
    data_name = 'zoo'
    data = pd.read_csv("{}{}.data".format(path, data_name), header=None)

    fea = data.iloc[:, 1:-1]
    label = data.iloc[:, -1]

    for i in replace:
        fea[13].replace(i, replace[i], inplace=True)

    feature = np.array(fea, dtype=np.float32)
    labels, uniques = pd.factorize(label)

    return feature, labels


def load_iris():
    """(4) 0 is nominal; 0 is ordinal; numerical is 4 """
    path = './openhgnn/dataset/QGRLdata/iris/'
    data_name = 'iris'
    idx_features = np.genfromtxt("{}{}.data".format(path, data_name), dtype=np.dtype(str))
    fea = []
    label = []
    for i in range(len(idx_features)):
        sub_fea = idx_features[i].split(',')
        fea.append([float(num) for num in sub_fea[:-1]])
        label.append(sub_fea[-1])
    fea = np.array(fea, dtype=np.float32)
    label = np.array(label, dtype=np.str_)
    _, _, labels = np.unique(label, return_index=True, return_inverse=True)
    # labels += 1
    return fea, labels


def load_wine():
    """(13) 0 is nominal;0 is ordinal; numerical is 0-12 """

    path = './openhgnn/dataset/QGRLdata/wine/'
    data_name = 'wine'
    data = pd.read_csv("{}{}.data".format(path, data_name), header=None) #第0列是label
    label = data.iloc[:, 0]

    fea = data.iloc[:, 1:]
    labels = label.values.astype(int) - 1
    fea = fea.values.astype(np.float32)
    return fea, labels


def load_heart():
    """(12) Column 0-4 (5) is nominal; (0) is ordinal; numerical is 0-6(7) [0,2,4,6,7,8,11]"""

    path = './openhgnn/dataset/QGRLdata/heart_failure/'
    data_name = 'heart'
    data = pd.read_csv("{}{}.csv".format(path, data_name)) #第0列是label
    label = data.iloc[:, -1]
    fea = data.iloc[:, :-1]
    fea = fea.values.astype(np.float32)
    label = label.values.astype(int)

    fea = move_column_to_end(fea, [0, 2, 4, 6, 7, 8, 11])

    return fea, label


def load_ttt():
    """(9) Column 0-8 (9) is nominal; 0 is ordinal; numerical is 0 """

    replace = {'x': 0, 'o': 1, 'b': 2}

    path = './openhgnn/dataset/QGRLdata/tic_tac_toe/'
    data_name = 'tic_tac_toe'
    data = pd.read_csv("{}{}.data".format(path, data_name), header=None) #第0列是label

    fea = data.iloc[:, :-1]

    label = data.iloc[:, -1]
    labels, _ = pd.factorize(label)

    for i in replace:
        fea.replace(i, replace[i], inplace=True)

    fea = fea.values.astype(np.float32)
    return fea, labels


def load_yeast():
    """(8) 0 is nominal; 0 is ordinal; 0-7 numerical is 8 """
    path = './openhgnn/dataset/QGRLdata/yeast/'
    data_name = 'yeast'
    data = np.genfromtxt("{}{}.data".format(path, data_name), dtype=np.dtype(str))
    data = pd.DataFrame(data)

    fea = data.iloc[:, 1:9]
    labels, _ = pd.factorize(data.iloc[:, 9])

    fea = fea.values.astype(np.float32)

    return fea, labels


def load_glass():
    """(9); 0 is nominal; 0 is ordinal; 0-8 numerical is 9 """
    path = './openhgnn/dataset/QGRLdata/glass_identification/'
    data_name = 'glass'
    data = pd.read_csv("{}{}.data".format(path, data_name), header=None)

    label = data.iloc[:, -1]
    fea = data.iloc[:, 1:-1]

    fea = fea.values.astype(np.float32)
    labels = label.values.astype(int) - 1
    labels[163:] -= 1
    return fea, labels

def load_BC():
    """This function is the test dataset loader of matlab code to python code!!!"""
    path = './openhgnn/dataset/QGRLdata/BC.csv'
    data = pd.read_csv("{}".format(path), header=None)

    fea = data.iloc[:, :-1]
    label = data.iloc[:, -1]

    fea = fea.values.astype(np.float32) - 1 # 保证从0开始
    label = label.values.astype(int) - 1

    return fea, label


def load_AA():
    """9 0-6 categorical 7,8 numerical"""
    path = './openhgnn/dataset/QGRLdata/Autism-Adolescent/'
    data_name = 'AA'
    data = pd.read_csv("{}{}.csv".format(path, data_name), header=None)

    label = data.iloc[:, -1]
    fea = data.iloc[:, :-1]

    fea = fea.values.astype(np.float32) - 1
    labels = label.values.astype(int) - 1
    return fea, labels


def load_mammographic():
    path = './openhgnn/dataset/QGRLdata/Mammographic/'
    data_name = 'mammographic'
    data = pd.read_csv("{}{}.csv".format(path, data_name), header=None)

    label = data.iloc[:, -1]
    fea = data.iloc[:, :-1]

    fea = pd.get_dummies(fea)

    fea = fea.values.astype(np.float32)
    fea[:, :-1] -= 1

    labels = label.values.astype(int) - 1
    return fea, labels


class pm(object):
    def __init__(self, x, label, no_nom_att, no_ord_att, no_num_att):
        super(pm, self).__init__()
        self.x = x
        self.label = label
        self.no_nom_att = no_nom_att
        self.no_ord_att = no_ord_att
        self.no_num_att = no_num_att
        self.n = x.shape[0]
        self.d = x.shape[1]
        self.k = len(np.unique(label))
        self.no_values = []
        for t in range(0, self.d):
            self.no_values.append(len(np.unique(x[:, t])))
        for i in range(self.d - self.no_num_att, self.d):
            """numerical attributes normalization"""
            self.x[:, i] = ( self.x[:, i] - np.min(self.x[:, i]) ) / ( np.max(self.x[:, i]) - np.min(self.x[:, i]) )
        self.intra_pd = None
        self.cpd = None
        self.dis_matrix = None
        self.adjacent_matrix = None
        self.cw = None
        self.pbr_dis_matrix = None
        self.x_coded = None

        self.bd_computer()
        self.pbr()

        self.graph_based_dissimilarity()
        self.feature_untie_h2h()

    def bd_computer(self):
        x = self.x.copy()

        ia_pd_list = [None for i in range(0, self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            '''intra attribute probability distribution'''
            all_sum = len(x[:, t])
            ia_pd = np.zeros((1, self.no_values[t]))
            for m in range(0, self.no_values[t]):  # for each possible value of a ^ t
                locate_x_tm = x[:, t] == m
                no_x_tm = sum(locate_x_tm)
                ia_pd[0, m] = no_x_tm / all_sum
            ia_pd_list[t] = ia_pd

        dis_matrix = [None for i in range(0, self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            dis_matrix[t] = np.zeros((self.no_values[t], self.no_values[t]))
        cpd = [None for i in range(self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):  # cpd
            '''for each categorical attributes'''
            cpd[t] = [ [None for i in range(self.d)] for j in range(self.no_values[t]) ]
            for m in range(0, self.no_values[t]):  # for each possible value of a ^ t
                locate_x_tm = x[:, t] == m
                no_x_tm = sum(locate_x_tm)  # sum the categorical number in a column; so it need from 0 to values - 1

                for r in range(0, self.d - self.no_num_att):
                    """if a^r is a categorical attribute"""
                    cpd[t][m][r] = np.zeros((1, self.no_values[r]))
                    for g in range(0, self.no_values[r]):
                        cpd[t][m][r][0, g] = sum(x[locate_x_tm, r] == g) # has locate_x_tm(a) and g(b) values p(a, b)
                    cpd[t][m][r] /= no_x_tm # p(a|b) = p(a, b) / p(a)

                for r in range(self.d - self.no_num_att, self.d):
                    '''if a^r is a numerical attribute'''
                    cpd[t][m][r] = np.zeros((1, 5))
                    for s in [0.0, 0.2, 0.4, 0.6]:
                        cpd[t][m][r][0, int(s * 5)] = np.sum( (x[locate_x_tm, r] >= s - 0.0) * (x[locate_x_tm, r] < s + 0.2))
                    cpd[t][m][r][0, 4] = no_x_tm - np.sum(cpd[t][m][r])
                    cpd[t][m][r] /= no_x_tm

        for t in range(0, self.no_nom_att):  # nominal distance
            for m in range(0, self.no_values[t] - 1):
                for h in range(m + 1, self.no_values[t]):  # h = m + 1
                    cost_relate = np.zeros((1, self.d))
                    for r in range(0, self.no_nom_att):  # nominal attributes
                        diff_relate = cpd[t][h][r] - cpd[t][m][r]
                        cost_relate[0, r] = np.sum(np.abs(diff_relate)) / 2

                    for r in range(self.no_nom_att, self.d - self.no_num_att):  # ordinal attributes
                        diff_relate = cpd[t][h][r] - cpd[t][m][r]
                        for s in range(0, self.no_values[r] - 1):
                            cost_relate[0, r] += np.abs(diff_relate[0, s])
                            diff_relate[0, s+1] = diff_relate[0, s] + diff_relate[0, s+1]  # the next diff accumulate the latest
                        cost_relate[0, r] /= (self.no_values[r] - 1)

                    for r in range(self.d - self.no_num_att, self.d):
                        diff_relate = cpd[t][h][r] - cpd[t][m][r]
                        for s in range(0, 4):
                            cost_relate[0, r] += np.abs(diff_relate[0, s])
                            diff_relate[0, s+1] = diff_relate[0, s] + diff_relate[0, s+1]  # the same as ordinal
                        cost_relate[0, r] /= 4

                    dis_matrix[t][m][h] = np.mean(cost_relate)
                    dis_matrix[t][h][m] = dis_matrix[t][m][h]

        for t in range(self.no_nom_att, self.d - self.no_num_att):  # ordinal distance
            dist_vct = np.zeros((1, self.no_values[t] - 1))
            for m in range(0, self.no_values[t] - 1):
                cost_relate = np.zeros((1, self.d))
                for r in range(0, self.no_nom_att):
                    """nominal"""
                    diff_relate = cpd[t][m+1][r] - cpd[t][m][r]
                    cost_relate[0, r] = np.sum(np.abs(diff_relate)) / 2

                for r in range(self.no_nom_att, self.d - self.no_num_att):
                    """ordinal"""
                    diff_relate = cpd[t][m+1][r] - cpd[t][m][r]
                    for s in range(0, self.no_values[r] - 1):
                        cost_relate[0, r] += np.abs(diff_relate[0, s])
                        diff_relate[0, s+1] += diff_relate[0, s]
                    cost_relate[0, r] /= (self.no_values[r] - 1)

                for r in range(self.d - self.no_num_att, self.d):
                    diff_relate = cpd[t][m+1][r] - cpd[t][m][r]
                    for s in range(0, 4):
                        cost_relate[0, r] += np.abs(diff_relate[0, s])
                        diff_relate[0, s+1] += diff_relate[0, s]
                    cost_relate[0, r] /= 4

                dist_vct[0, m] = np.mean(cost_relate)

            for m in range(0, self.no_values[t] - 1):
                for h in range(m+1, self.no_values[t]):
                    dis_matrix[t][m][h] = np.sum(dist_vct[0, m:h])
                    dis_matrix[t][h][m] = dis_matrix[t][m][h]
            dis_matrix[t] /= np.max(dis_matrix[t])

        # ------------two priority--------------
        self.intra_pd = ia_pd_list
        self.cpd = cpd
        self.dis_matrix = dis_matrix

    def pbr(self):
        bd_mtx = self.dis_matrix.copy()
        x = self.x.copy()
        self.cw = np.zeros((1, self.d))
        for i in range(0, self.no_nom_att):
            self.cw[0, i] = np.max(bd_mtx[i])
        self.cw[0, self.no_nom_att:-1] = 1
        pbr_list = (np.arange(0, self.no_nom_att) + 1) * (np.array(self.no_values[0:self.no_nom_att]) > 2)
        pbr_list = (pbr_list[pbr_list != 0] - 1).tolist()

        npbr_list = np.setdiff1d(np.arange(0, self.d - self.no_num_att), pbr_list).tolist()# 245678
        num_pbr_att = len(pbr_list)
        num_npbr_att = len(npbr_list)
        pbr_dis = [None] * num_pbr_att
        for r in range(0, num_pbr_att):
            num_att_val = self.no_values[pbr_list[r]]
            Cn2 = ((num_att_val * (num_att_val - 1)) // 2)
            pbr_dis[r] = [None] * Cn2
            for k in range(Cn2):
                pbr_dis[r][k] = np.zeros((num_att_val, num_att_val))

            num_new_att = -1
            for v1 in range(0, self.no_values[pbr_list[r]] - 1):
                for v2 in range(v1+1, self.no_values[pbr_list[r]]):
                    num_new_att += 1
                    d12 = bd_mtx[pbr_list[r]][v1, v2]
                    plist = np.setdiff1d(np.arange(0, self.no_values[pbr_list[r]]), [v1, v2])
                    pval = np.zeros((1, self.no_values[pbr_list[r]]))
                    pval[0, v1] = 0
                    pval[0, v2] = d12
                    for vm in plist:
                        d1m = bd_mtx[int(pbr_list[r])][v1, vm]
                        d2m = bd_mtx[int(pbr_list[r])][v2, vm]
                        if d1m > d2m:
                            e = (d1m**2-d2m**2+d12**2) / (2*d12)
                            pval[0, vm] = e
                        else:
                            e = (d2m**2-d1m**2+d12**2) / (2*d12)
                            pval[0, vm] = bd_mtx[pbr_list[r]][v1, v2] - e
                    pval -= np.min(pval)
                    for vv1 in range(0, self.no_values[pbr_list[r]] - 1):
                        for vv2 in range(vv1+1, self.no_values[pbr_list[r]]):
                            pbr_dis[r][num_new_att][vv1, vv2] = np.abs(pval[0, vv1] - pval[0, vv2])
                            pbr_dis[r][num_new_att][vv2, vv1] = pbr_dis[r][num_new_att][vv1, vv2]
        x_pbr = np.zeros((self.n, 0))
        dis_mtx_pbr = []
        no_val_pbr = np.zeros((0))
        pbr_rcd = np.zeros((0))
        for r in range(0, num_pbr_att):
            num_new_att = self.no_values[pbr_list[r]] * (self.no_values[pbr_list[r]] - 1) // 2
            expand_x = np.tile(x[:, pbr_list[r]:pbr_list[r]+1], (1, int(num_new_att)))
            x_pbr = np.hstack((x_pbr, expand_x))
            dis_mtx_pbr.append(pbr_dis[r])
            no_val_pbr = np.hstack((no_val_pbr, (np.ones(int(num_new_att)) * self.no_values[pbr_list[r]])))
            pbr_rcd = np.hstack((pbr_rcd, (np.ones(int(num_new_att)) * r)))

        num_list = [i for i in range(self.d-self.no_num_att, self.d)]
        column = npbr_list + num_list
        x_ncoded = x[:, column]
        x_coded = np.hstack((x_pbr, x_ncoded))

        dis_mtx = [None] * (self.d - self.no_num_att)
        for i in range(0, self.d-self.no_num_att):
            for j in range(len(dis_mtx_pbr)):
                dis_mtx[pbr_list[j]] = dis_mtx_pbr[j]
            for j in range(len(npbr_list)):
                dis_mtx[npbr_list[j]] = bd_mtx[npbr_list[j]]

        # ---------------------------------------------------------
        self.pbr_dis_matrix = dis_mtx
        self.x_coded = x_coded

    def graph_based_dissimilarity(self):
        x = self.x.copy()
        numerical_values = []
        for i in range(self.d - self.no_num_att, self.d):
            """numerical attributes normalization"""
            x[:, i] = ( x[:, i] - np.min(x[:, i]) ) / ( np.max(x[:, i]) - np.min(x[:, i]) )

        for t in range(self.d - self.no_num_att, self.d):
            numerical_values.append(np.unique(x[:, t]))

        cpd = [None for i in range(self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            '''for each categorical attributes'''
            cpd[t] = [ [None for i in range(self.d)] for j in range(self.no_values[t]) ]
            for m in range(0, self.no_values[t]):  # for each possible value of a ^ t
                locate_x_tm = x[:, t] == m
                no_x_tm = sum(locate_x_tm)  # sum the categotical number in a column; so it need from 0 to values - 1

                for r in range(0, self.d - self.no_num_att):
                    """if a^r is a categorical attribute"""
                    cpd[t][m][r] = np.zeros((1, self.no_values[r]))
                    for g in range(0, self.no_values[r]):
                        cpd[t][m][r][0, g] = sum(x[locate_x_tm, r] == g) # has locate_x_tm(a) and g(b) values p(a, b)
                    cpd[t][m][r] /= no_x_tm # p(a|b) = p(a, b) / p(a)

                for r in range(self.d - self.no_num_att, self.d):
                    '''if a^r is a numerical attribute'''
                    cpd[t][m][r] = np.zeros((1, self.no_values[r]))
                    for g in range(0, self.no_values[r]):
                        a = x[locate_x_tm, r]
                        b = numerical_values[r-self.no_nom_att-self.no_ord_att][g]
                        cpd[t][m][r][0, g]=sum(a==b)
                        # has locate_x_tm(a) and g(b) values p(a, b)
                    cpd[t][m][r] /= no_x_tm  # p(a|b) = p(a, b) / p(a)

        diff_matrix = [[None for i in range(self.d)] for i in range(0, self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            for r in range(0, self.d):
                diff_matrix[t][r] = np.zeros((self.no_values[t], self.no_values[t]))
        for d in range(0, self.no_nom_att):
            for r in range(0, self.d):
                for m in range(0, self.no_values[d] - 1):
                    for h in range(m + 1, self.no_values[d]):  # h = m + 1
                        diff_relate = cpd[d][m][r] - cpd[d][h][r]
                        phi = None
                        if 0 <= r < self.no_nom_att:  # based on nominal
                             phi = np.abs(np.sum(diff_relate[np.where(diff_relate > 0)] * 1))
                        elif self.no_nom_att <= r < self.d:  # based on numerical or ordinal
                            max_tag = np.where(diff_relate==np.max(diff_relate))[1].tolist()

                            t_vector = None
                            if self.no_nom_att <= r < self.d-self.no_num_att:
                                ordinal_vector = np.linspace(0, 1, self.no_values[r])
                                t_vector = ordinal_vector
                            elif self.d - self.no_num_att <= r < self.d:
                                numerical_vector = np.array(numerical_values[r-self.no_nom_att-self.no_ord_att])
                                t_vector = numerical_vector
                            min_phi = float('inf')
                            for tag in max_tag:
                                t = np.abs(t_vector - t_vector[tag])
                                phi = np.sum(np.abs(diff_relate) * t)
                                if phi < min_phi:
                                    min_phi = phi
                            phi = min_phi
                        diff_matrix[d][r][m][h] = phi
                        diff_matrix[d][r][h][m] = diff_matrix[d][r][m][h]

        for d in range(self.no_nom_att, self.d-self.no_num_att):
            for r in range(0, self.d):
                for m in range(0, self.no_values[d] - 1):
                    for h in range(m + 1, self.no_values[d]):  # h = m + 1
                        for v_mh in range(m, h):
                            diff_relate = cpd[d][v_mh][r] - cpd[d][v_mh+1][r]
                            phi = None
                            if 0 <= r < self.no_nom_att:  # based on nominal
                                phi = np.abs(np.sum(diff_relate[np.where(diff_relate > 0)] * 1))
                            elif self.no_nom_att <= r < self.d:  # based on numerical or ordinal
                                max_tag = np.where(diff_relate==np.max(diff_relate))[1].tolist()
                                t_vector = None
                                if self.no_nom_att <= r < self.d-self.no_num_att:
                                    ordinal_vector = np.linspace(0, 1, self.no_values[r])
                                    t_vector = ordinal_vector
                                elif self.d - self.no_num_att <= r < self.d:
                                    numerical_vector = np.array(numerical_values[r-self.no_nom_att-self.no_ord_att])
                                    t_vector = numerical_vector
                                min_phi = float('inf')
                                for tag in max_tag:
                                    t = np.abs(t_vector - t_vector[tag])
                                    phi = np.sum(np.abs(diff_relate) * t)
                                    if phi < min_phi:
                                        min_phi = phi
                                phi = min_phi
                            diff_matrix[d][r][m][h] += phi
                            diff_matrix[d][r][h][m] = diff_matrix[d][r][m][h]

        # compute w
        w_list = [[None for i in range(self.d)] for i in range(0, self.d - self.no_num_att)]

        for d in range(self.d-self.no_num_att):
            if d < self.no_nom_att:
                for r in range(0, self.d):
                    w_list[d][r] = (np.sum(diff_matrix[d][r]) / 2) / (self.no_values[d] * (self.no_values[d] - 1) / 2)
            elif self.no_nom_att <= d < self.d-self.no_num_att:
                for r in range(0, self.d):
                    sum_diff = 0
                    for i in range(0, self.no_values[d] - 1):
                        sum_diff += diff_matrix[d][r][i, i+1]
                    w_list[d][r] = sum_diff / (self.no_values[d] - 1)

        # compute distance
        dis_matrix = [None for i in range(0, self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            dis_matrix[t] = np.zeros((self.no_values[t], self.no_values[t]))

        for d in range(0, self.d-self.no_num_att):
            for s in range(0, self.d):
                dis_matrix[d] += w_list[d][s] * diff_matrix[d][s]

        for d in range(0, self.d-self.no_num_att):
            if d < self.no_nom_att:
                dis_matrix[d] /= self.d
            elif self.no_nom_att <= d < self.d-self.no_num_att:
                dis_matrix[d] /= np.max(dis_matrix[d])
        # ------------adj--------------------
        connect_matrix = np.zeros((self.n, self.n))

        for i in range(0, self.n):
            for j in range(i, self.n):
                diff_vector = np.zeros((1, self.d))
                for r in range(0, self.d - self.no_num_att):
                    """nominal and ordinal"""
                    ai = int(x[i, r])
                    aj = int(x[j, r])
                    diff_vector[0, r] = dis_matrix[r][ai, aj]
                for r in range(self.d - self.no_num_att, self.d):
                    """numerical"""
                    ai = x[i, r]
                    aj = x[j, r]
                    diff_vector[0, r] = aj - ai
                distance = np.linalg.norm(diff_vector, ord=2)  # ord is the 1,2,...
                connect_matrix[i, j] = distance
                connect_matrix[j, i] = connect_matrix[i, j]
        # -------adj--------------------
        self.adjacent_matrix = connect_matrix

    def feature_untie_h2h(self):
        """intra_pd + cpd + pbr + numerical"""
        intra_pd = self.intra_pd.copy()
        cpd = self.cpd.copy()
        pbr_dis_matrix = self.pbr_dis_matrix.copy()
        x = self.x.copy()
        feature_matrix = []
        for i in range(self.d - self.no_num_att, self.d):
            '''numerical attributes normalization'''
            x[:, i] = ( x[:, i] - np.min(x[:, i]) ) / ( np.max(x[:, i]) - np.min(x[:, i]) )

        for t in range(0, self.n):
            expand_feature_vector = np.zeros((1, 0))
            for r in range(0, self.d - self.no_num_att):
                values = int(x[t, r])
                expand_feature_vector = np.hstack((expand_feature_vector, intra_pd[r][0:1, values:values+1]))  # ia_pd
                for i in range(0, self.d - self.no_num_att):  # cpd
                    expand_feature_vector = np.hstack((expand_feature_vector, cpd[r][values][i]))

                if r < (self.d-self.no_num_att-self.no_ord_att):  # pbr
                    if isinstance(pbr_dis_matrix[r], list):
                        for j in range(0, len(pbr_dis_matrix[r])):
                            expand_feature_vector = np.hstack(
                                (expand_feature_vector, pbr_dis_matrix[r][j][values:values+1, :]))
                    else:
                        expand_feature_vector = np.hstack(
                            (expand_feature_vector, pbr_dis_matrix[r][values:values+1, :]))
                elif (self.d-self.no_num_att-self.no_ord_att) <= r < (self.d-self.no_num_att):
                    expand_feature_vector = np.hstack((expand_feature_vector, pbr_dis_matrix[r][values:values+1, :]))

            for r in range(self.d - self.no_num_att, self.d):  # numerical
                expand_feature_vector = np.hstack((expand_feature_vector, x[t:t+1, r:r+1]))

            feature_matrix.append(expand_feature_vector[0, :])
        feature_matrix = np.array(feature_matrix)

        self.x = feature_matrix


def MLload_data(name):
    '''zoo, iris, wine, car, heart, ttt, lymphography, yeast, breast, soybean, hayes'''
    if name.lower() == 'zoo':
        fea, lab = load_zoo()
        p = pm(fea, lab, no_nom_att=16, no_ord_att=0, no_num_att=0)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'iris':
        fea, lab = load_iris()
        p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=4)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() =='wine':
        fea, lab = load_wine()
        p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=13)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'heart':
        fea, lab = load_heart()
        p = pm(fea, lab, no_nom_att=5, no_ord_att=0, no_num_att=7)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'ttt':
        fea, lab = load_ttt()
        p = pm(fea, lab, no_nom_att=9, no_ord_att=0, no_num_att=0)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'yeast':
        fea, lab = load_yeast()
        p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=8)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'breast':
        fea, lab = load_BC()
        p = pm(fea, lab, no_nom_att=5, no_ord_att=4, no_num_att=0)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'glass':
        fea, lab = load_glass()
        p = pm(fea, lab, no_nom_att=0, no_ord_att=0, no_num_att=9)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'aa':
        fea, lab = load_AA()
        p = pm(fea, lab, no_nom_att=7, no_ord_att=0, no_num_att=2)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    elif name.lower() == 'mm':
        fea, lab = load_mammographic()
        p = pm(fea, lab, no_nom_att=4, no_ord_att=0, no_num_att=1)
        adj = p.adjacent_matrix
        fea = p.x
        return fea, adj, lab
    else:
        """toy example test"""
        fea = np.array([[0, 1, 0.],
                        [1, 0, 0.36],
                        [1, 1, 0.64],
                        [2, 1, 0.61],
                        [2, 2, 1.0]])
        lab = np.array([0, 0, 0, 0, 0])
        p = pm(fea, lab, no_nom_att=1, no_ord_att=1, no_num_att=1)
        return


'''{'zoo':7 36, 'iris':3 4, 'wine':3 13, 'car':4 21, 'heart': 2 17, 'ttt': 2 17, 
\\'yeast': 10 8, 'breast':2 43 , 'hayes': 3 15 , 'glass': 6, 'lymphography': 4 9}'''

