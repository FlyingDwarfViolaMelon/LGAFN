import torch
from LGAFN import LGAFN
import numpy as np
from opt import args
from sklearn.decomposition import PCA
from utils import setup, LoadDataset
from train import train
from module.load_graph import load_graph
import scipy.io as sio
import random
import hdf5storage
from time import *

setup()

print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

args.data_path = 'data/{}/{}.mat'.format(args.name, args.name) 
args.label_path = 'data/{}_label.txt'.format(args.name)
args.adj_path = 'data/{}_graph.txt'.format(args.name)

print("Data: {}".format(args.data_path))
print("Label: {}".format(args.label_path))
print("Adj: {}".format(args.adj_path))

# X = hdf5storage.loadmat(args.data_path)
X = sio.loadmat(args.data_path)
X_dict = dict(X)
X = X_dict['X']
x1 = X[0][0]
x1 = np.float32(x1)
x2 = X[0][1]
x2 = np.float32(x2)

y = np.loadtxt(args.label_path, dtype=int)
adj = load_graph(args.name, 5)
n, m = x2.shape
args.n_input2 = m

NMI = []
ARI = []
ACC = []
F1 = []
tol_pred = {}
tol_embedd = {}

if args.pca_status:
    pca = PCA(n_components=args.n_components)
    X1_pca = pca.fit_transform(x1)
    X2_pca = pca.fit_transform(x2)
else:
    X1_pca = x1
    X2_pca = x2

dataset1 = LoadDataset(X1_pca)
dataset2 = LoadDataset(X2_pca)

# pred_save_path = 'result/{}_pred.mat'.format(args.name)
# embedd_save_path = 'result/{}_embed.mat'.format(args.name)
runtime = 1

for iter_number in range(runtime):
    begin_time = time()
    model = LGAFN(n_input1=args.n_input1, n_input2=args.n_input2,
                     hidden_gsa_dim=args.hidden_gsa_dim,
                     encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim,
                     n_z=args.n_z,n_clusters=args.n_clusters,
                     n_node=n, v=1.0).to(device)

    metrics, best_embedding, final_pred = train(model, adj, dataset1, dataset2, y, device, iter_number)
    
    end_time = time()
    run_time = end_time - begin_time
    print('Time:' + str(run_time))
    acc = metrics[0]
    nmi = metrics[1]
    ari = metrics[2]
    f1 = metrics[3]
    ACC.append(acc)
    NMI.append(nmi)
    ARI.append(ari)
    F1.append(f1)
    pred_name = f'ypred_{iter_number + 1}'
    tol_pred[pred_name] = final_pred
    embedd_name = f'embedd_{iter_number + 1}'
    

NMI_mean = np.mean(NMI)
NMI_std = np.std(NMI)

ARI_mean = np.mean(ARI)
ARI_std = np.std(ARI)

ACC_mean = np.mean(ACC)
ACC_std = np.std(ACC)

F1_mean = np.mean(F1)
F1_std = np.std(F1)

print('ACC_mean:' + str(ACC_mean) + ',' + 'ACC_std:' + str(ACC_std))
print('NMI_mean:' + str(NMI_mean) + ',' + 'NMI_std:' + str(NMI_std))
print('ARI_mean:' + str(ARI_mean) + ',' + 'ARI_std:' + str(ARI_std))
print('F1_mean:' + str(F1_mean) + ',' + 'F1_std:' + str(F1_std))

