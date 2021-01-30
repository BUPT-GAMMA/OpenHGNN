import scipy.io as scio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class evaluation:
    def __init__(self, seed):
        self.seed = seed

    def cluster(self, n, X, Y):

        Y_pred = KMeans(n, random_state=self.seed).fit(np.array(X)).predict(X)
        nmi = normalized_mutual_info_score(np.array(Y), Y_pred)
        ari = adjusted_rand_score(np.array(Y), Y_pred)
        return nmi, ari

    def classification(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)
        LR = LogisticRegression()
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)

        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1

def evaluate_acm(seed, X, Y, n):
    _evaluation = evaluation(seed)
    NMI, ARI = _evaluation.cluster(n, X, Y)
    micro, macro = _evaluation.classification(X, Y)

    print('<Cluster>        NMI = %.4f, ARI = %.4f' % (NMI, ARI))

    print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro, macro))