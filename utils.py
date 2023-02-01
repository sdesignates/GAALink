import pandas as pd
import torch
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

"""
Construct a graph based on the cell features
"""




"""
Load scRNA-seq data set and perfrom preprocessing
"""


def load_data(data_path, dataset_str, PCA_dim,  n_clusters=20):
    # Get data
    DATA_PATH = data_path

    data = pd.read_csv(DATA_PATH, index_col=0, sep='\t')
    cells = data.columns.values
    genes = data.index.values
    features = data.values.T

    # Preprocess features
    features = normalization(features)

    # Construct graph
    N = len(cells)
    avg_N = N // n_clusters
    K = avg_N // 10
    K = min(K, 20)
    K = max(K, 6)

    L = 0

    method = 'pearson'
    adj = getGraph(dataset_str, features, L, K, method)

    # feature tranformation
    if features.shape[0] > PCA_dim and features.shape[1] > PCA_dim:
        pca = PCA(n_components=PCA_dim)
        features = pca.fit_transform(features)
    else:
        var = np.var(features, axis=0)
        min_var = np.sort(var)[-1 * PCA_dim]
        features = features.T[var >= min_var].T
        features = features[:, :PCA_dim]
    print('Shape after transformation:', features.shape)

    features = (features - np.mean(features)) / (np.std(features))

    return adj, features, cells, genes


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor





def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # y_p = torch.argmax(y_pred,dim=1)
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()


    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR, AUPR_norm




def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr



def Network_Statistic(data_type,net_scale,net_type):

    if net_type =='STRING Dataset':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale



    elif net_type == 'Non-Specific Dataset':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific Dataset':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof Dataset':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError

