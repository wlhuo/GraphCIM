# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/11/27 15:03:30
@Author  :   wlhuo 
'''
import numpy as np
import torch
import copy
import random

import scipy
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, accuracy_score
from matplotlib.colors import ListedColormap, Normalize


def sample_n(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def prepare_inputs(adj, features):
    adj_norm = preprocess_graph(adj)
    adj_norm = scipy.sparse.coo_matrix((adj_norm[1], (adj_norm[0][:, 0], adj_norm[0][:, 1])),
                                       shape=adj_norm[2]).toarray()
    adj_norm = torch.FloatTensor(adj_norm)

    pos_weight_node = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm_node = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    features = torch.FloatTensor(np.array(features.todense()))
    # features_nonzero = torch.where(features == 1)[0].shape[0]
    features_nonzero = features.sum().item()

    pos_weight_attr = torch.tensor(
        float(features.shape[0] * features.shape[1] - features_nonzero) / features_nonzero)
    norm_attr = features.shape[0] * features.shape[1] / float(
        (features.shape[0] * features.shape[1] - features_nonzero) * 2)

    return adj_norm, pos_weight_node, norm_node, features, pos_weight_attr, norm_attr

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])


    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def get_rec_loss(norm, pos_weight, pred, labels, loss_type='bce'):
    if loss_type == 'bce':
        return norm * torch.mean(
            F.binary_cross_entropy_with_logits(input=pred, target=labels, reduction='none', pos_weight=pos_weight),
            dim=[0, 1])
    elif loss_type == 'sce':
        return norm * sce_loss(pred, labels)
    elif loss_type == 'mse':
        return norm * torch.mean(
            F.mse_loss(input=pred, target=labels, reduction='none'),
            dim=[0, 1]
        )
    else:
        assert loss_type == 'sig'
        return norm * sig_loss(pred, labels)

def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    # loss = loss.mean()
    loss = torch.mean(loss, dim=[0, 1])
    return loss


def adj_augment(adj_mat, aug_prob):
    # adj_mat: scipy sparse matrix
    # Symmetric Matrices

    # change inplace
    # copy is very important
    # aug_prob /= 2
    adj_mat = adj_mat.copy()
    xrow, yrow = adj_mat.nonzero()
    low_tri = xrow > yrow
    xrow = xrow[low_tri]
    yrow = yrow[low_tri]
    num_indices = len(xrow)
    selected_idx = random.sample(range(num_indices), int(num_indices * aug_prob))
    selected_idx.sort()
    xrow_ = xrow[selected_idx]
    yrow_ = yrow[selected_idx]
    adj_mat[xrow_, yrow_] = 0
    adj_mat[yrow_, xrow_] = 0
    adj_mat.eliminate_zeros()
    return adj_mat


def attr_augment(attr_mat, aug_prob):
    # attr_mat: scipy dense matrix

    # change inplace
    # copy is very important
    attr_mat = attr_mat.copy()
    xrow, yrow = attr_mat.nonzero()
    num_indices = len(xrow)
    selected_idx = random.sample(range(num_indices), int(num_indices * aug_prob))
    selected_idx.sort()
    xrow_ = xrow[selected_idx]
    yrow_ = yrow[selected_idx]
    attr_mat[xrow_, yrow_] = 0
    return attr_mat


def get_roc_score_node(edges_pos, edges_neg, emb, adj):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score_attr(feas_pos, feas_neg, logits_attr, features_orig):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    # fea_rec = np.dot(emb_node, emb_attr.T)
    fea_rec = logits_attr
    preds = []
    pos = []
    for e in feas_pos:
        preds.append(sigmoid(fea_rec[e[0], e[1]][0]))
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(sigmoid(fea_rec[e[0], e[1]][0]))
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_mse_attr(feas_pos, feas_neg, fea_rec, features_orig):
    # Predict on test set of edges
    preds = []
    pos = []
    for e in feas_pos:
        preds.append(fea_rec[e[0], e[1]][0].item())  # Note this [0].item
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(fea_rec[e[0], e[1]][0].item())
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([pos, neg])

    try:
        mse = mean_squared_error(labels_all, preds_all)
    except Exception as e:
        print(e)
        mse = 1e8
    return mse



def get_recovered(edges_pos, edges_neg, recovere):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    recovered = sigmoid(np.dot(recovere, recovere.T))
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(recovered[e[0], e[1]])

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(recovered[e[0], e[1]])

    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    all_acc_score = {}
    max_acc_score = 0
    optimal_threshold = 0
    for threshold in np.arange(0.01,1,0.005):
        preds_all = np.hstack([preds, preds_neg])
        preds_all = (preds_all>threshold).astype('int')
        acc_score = accuracy_score(labels_all, preds_all)
        all_acc_score[threshold] = acc_score
        if acc_score > max_acc_score:
            max_acc_score = acc_score
            optimal_threshold = threshold

    for i in range(0, recovered.shape[0]):
        recovered[i,i] = 0

    adj_rec_1 = copy.deepcopy(recovered)
    adj_rec_1 = (adj_rec_1>optimal_threshold).astype('int')
    for j in range(0, adj_rec_1.shape[0]):
        adj_rec_1[j,j] = 0

    def add_limit(adj_rec, adj_rec_1, top_num):
        adj_rec_new_tmp = copy.deepcopy(adj_rec)
        for z in range(0, adj_rec_new_tmp.shape[0]):
            tmp = adj_rec_new_tmp[z,:]
            adj_rec_new_tmp[z,:] = (adj_rec_new_tmp[z,:] >= np.sort(tmp)[-top_num]).astype('int')
        adj_rec_new = adj_rec_1 + adj_rec_new_tmp
        adj_rec_new = (adj_rec_new == 2).astype('int')
        adj_rec_new = adj_rec_new + adj_rec_new.T
        adj_rec_new = (adj_rec_new != 0).astype('int')
        return adj_rec_new

    adj_rec_new = add_limit(recovered, adj_rec_1, 5)

    return recovered, adj_rec_new


def write_csv_matrix(matrix, filename, ifindex=False, ifheader=True, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames
        ifindex, ifheader = ifheader, ifindex

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename+'.csv', index=ifindex, header=ifheader)


def map_cell_communication(cell_pixel, cell_type_pd, coord, recovered, cell_id):
    cell_type  = {}
    select_cell = recovered[104]
    origin_cell_type = cell_id["Cell Type"].values
    cell_cluster = cell_type_pd["Cell_class_id"].values
    for i, cell_t in enumerate(origin_cell_type):
        cell_type[cell_t] = cell_cluster[i]

    idx = np.where(select_cell == 1)
    select_cell_coord = coord.values[104]
    connect_cells_coord = coord.values[idx]

    lines = []
    for connect_cell in connect_cells_coord:
        lines.append((select_cell_coord,connect_cell))

    merged = np.zeros((1600,1600))
    with open(cell_pixel, 'r') as file:
        for line in file:
            pixel, type_value = line.split()
            x_, y_ = map(int, pixel.split(':')) 
            type_value = int(float(type_value)) 
            if type_value not in cell_type.keys():
                continue
            t = cell_type[type_value]
            x = int(float(x_))
            y = int(float(y_))
            if x >=1600 or y >=1600:
                continue
            merged[(x,y)] = t

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']


    cmap = ListedColormap(colors)
    norm = Normalize(vmin=0, vmax=len(colors) - 1)

    plt.figure(figsize=(12, 12))

    plt.imshow(merged, cmap=cmap, norm=norm, interpolation='nearest')

    cbar = plt.colorbar(label='Cell type')
    cbar.set_ticks(np.arange(len(colors)))
    cbar.set_ticklabels(['Kupffercells', 'TransitioningMonocytes', 'Hepatocyrtes', 'Cholangiocyte', 'CapsuleMacrophages', 'Endothelialcell']) 


    for start, end in lines:
        plt.plot([start[1], end[1]], [start[0], end[0]], color='black', linewidth=1, linestyle='--')

    plt.title('Cell communication network')
    plt.grid(False)  
    plt.show()
    plt.show()
    plt.savefig("cell_communication.png")


def map_special_cell(cell_pixel, cell_type_pd, coord, recovered, cell_id):
    cell_type  = {}
    cell_label = cell_type_pd.values[:,1]
    cell_interaction_count =  np.sum(recovered,axis=1)
    cell_coord = coord.values[np.where(cell_interaction_count > 200)]
    select_cell_coord = cell_coord[(cell_coord[:, 0] <= 1600) & (cell_coord[:, 1] <= 1600)]

    cell_cluster = cell_type_pd["Cell_class_id"].values
    for i, cell_t in enumerate(cell_id["Cell Type"].values):
        cell_type[cell_t] = cell_cluster[i]

    merged = np.zeros((1600,1600))
    with open(cell_pixel, 'r') as file:
        for line in file:
            # 解析行
            pixel, type_value = line.split()
            x_, y_ = map(int, pixel.split(':'))  
            type_value = int(float(type_value))  
            if type_value not in cell_type.keys():
                continue
            t = cell_type[type_value]
            x = int(float(x_))
            y = int(float(y_))
            if x >=1600 or y >= 1600:
                continue
            merged[(x,y)] = t


    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    cmap = ListedColormap(colors)
    norm = Normalize(vmin=0, vmax=len(colors) - 1)

    plt.figure(figsize=(12, 12))
    plt.imshow(merged, cmap=cmap, norm=norm, interpolation='nearest')

    cbar = plt.colorbar(label='Cell type')
    cbar.set_ticks(np.arange(len(colors))) 
    cbar.set_ticklabels(['Kupffercells', 'TransitioningMonocytes', 'Hepatocyrtes', 'Cholangiocyte', 'CapsuleMacrophages', 'Endothelialcell']) 


    plt.scatter(select_cell_coord[:, 0], select_cell_coord[:, 1], color='black', label='Highlighted Points', s=5,zorder=5)
    plt.title('Special Cell Map')

    plt.grid(False) 
    plt.show()
    plt.show()
    plt.savefig("special_cell_map.png")

def map_cell(cell_pixel, cell_type_pd, coord, recovered, cell_id):

    map_cell_communication(cell_pixel, cell_type_pd, coord, recovered, cell_id)
    map_special_cell(cell_pixel, cell_type_pd, coord, recovered, cell_id)




    

