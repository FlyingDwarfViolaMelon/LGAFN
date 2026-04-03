import torch
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import Dataset
import opt


def setup():
    """
    setup
    - name: the name of dataset
    - device: CPU / GPU
    - seed: random seed
    - n_clusters: num of cluster
    - n_input: dimension of feature
    - lr: learning rate
    Return: None

    """
    print("setting:")
    if opt.args.name == 'acm': 
        opt.args.n_clusters = 3
        opt.args.n_components = 100
        opt.args.lr = 5*1e-5
        opt.args.lambda1 = 1
        opt.args.lambda2 = 1
        if opt.args.pca_status:
            opt.args.n_input1 = 100
            opt.args.hidden_gsa_dim = [128, 256, 512]
            opt.args.encoder_dim = [128, 256, 512]
            opt.args.decoder_dim = [512, 256, 128]
        else:
            opt.args.n_input1 = 1870
            opt.args.hidden_gsa_dim = [500, 500, 2000]

    else:
        print("error!")
        print("please add the new dataset's parameters")
        print("------------------------------")
        print("dataset       : ")
        print("device        : ")
        print("clusters      : ")
        print("learning rate : ")
        print("n_input1 : ")
        print("------------------------------")
        exit(0)

    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("clusters      : {}".format(opt.args.n_clusters))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("n_input1      : {}".format(opt.args.n_input1))
    print("lambda1       : {}".format(opt.args.lambda1))
    print("lambda2       : {}".format(opt.args.lambda2))
    print("hidden_gsa_dim      : {}".format(opt.args.hidden_gsa_dim))
    print("------------------------------")


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

