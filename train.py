# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/11/27 15:03:20
@Author  :   wlhuo 
'''
import argparse
import numpy as np
import pandas as pd
from torch import optim

from utils import *
from model import GraphCIM, GraphCIM_B





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, help='TODO: Input gene expression data path')
    parser.add_argument('--adj', '-a', type=str, help='Input adjacency matrix data path')
    parser.add_argument('--cell_pixel', type=str, help='Input pixel of cell data path')
    parser.add_argument('--cell_type', type=str, help='Input cell type data path')
    parser.add_argument('--coord', type=str, help='Input coord data path')
    parser.add_argument('--seed', type=int, nargs="+", default=0, help='Random seed.')
    parser.add_argument('--epoch', type=int, nargs="+", default=30, help='Number of epochs to train.')
    parser.add_argument('--K', type=int, default=1,
                        help='Number of samples for importance re-weighting.')
    parser.add_argument('--J', type=int, default=1,
                        help='Number of samples for variational distribution q.')
    parser.add_argument('--eps', type=float, default=1e-7,
                        help='Eps')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument('--batch', type=int, default=0, help='Use batch processing.')
    parser.add_argument('--filename', type=str, default='result.txt', help='Filename for recording results.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = f'cuda:{args.gpu_id}' if args.cuda else 'cpu'

    return args


def create_batches(adj_norm, features, labels, batch_size):
    num_nodes = features.shape[1]
    # indices = np.random.permutation(num_nodes)
    indices = np.arange(0,num_nodes)
    for start in range(0, num_nodes, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        batch_adj_norm = adj_norm[start:start + batch_size, start:start + batch_size]
        batch_features = features[start:start + batch_size]

        batch_labels = labels[start:start + batch_size, start:start + batch_size]
        yield {'adj_norm':batch_adj_norm,
                'features':batch_features,
               'adj_label':batch_labels,
               'n_nodes':len(batch_indices)
        }



def cycle_operation(adj_train, fea_train, adj_label, num_nodes, num_features, features_test, adj_norm_test, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig, args):

    best_roc_val = 0
    best_ap_val = 0
    best_roc_test = 0
    best_ap_test = 0

    model = GraphCIM_B(input_dim=num_features,
                        num_hidden=512,
                        out_dim=512,
                        noise_dim=5,
                        dropout=0,
                        K=args.K,
                        J=args.J,
                        device=args.device)

    model.to(args.device)
    optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=2e-5)
    for epoch in range(1, args.epoch + 1):
        print("epoch:",epoch)
        # Graph augmentation

        adj_train_aug = adj_augment(adj_mat=adj_train, aug_prob=0)
        fea_train_aug = attr_augment(attr_mat=fea_train, aug_prob=0)

        adj_norm, pos_weight, norm, features, pos_weight_a, norm_a = prepare_inputs(adj=adj_train_aug,
                                                                                    features=fea_train_aug)
        adj_norm = adj_norm.to(args.device)
        features = features.to(args.device)
        warmup = np.min([epoch / 300., 1.])

        for batch in create_batches(adj_norm, features, adj_label, batch_size=args.batch):
            model.train()
            optimizer.zero_grad()
            merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
            merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
            reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(x=batch['features'],
                                                                                                        adj=batch['adj_norm'])

            node_attr_mu = torch.cat((merged_node_mu, merged_attr_mu), 0)
            node_attr_sigma = torch.cat((merged_node_sigma, merged_attr_sigma), 0)
            node_attr_z_samples = torch.cat((merged_node_z_samples, merged_attr_z_samples), 0)
            node_attr_logv_iw = torch.cat((node_logv_iw, attr_logv_iw), 0)

            ker = torch.exp(
                -0.5 * (torch.sum(
                    torch.square(node_attr_z_samples - node_attr_mu) / torch.square(node_attr_sigma + args.eps), 3)))

            log_H_iw_vec = torch.log(torch.mean(ker, 2) + args.eps) - 0.5 * torch.sum(node_attr_logv_iw, 2)
            log_H_iw = torch.mean(log_H_iw_vec, 0)

            adj_orig_tile = batch['adj_label'].unsqueeze(-1).expand(-1, -1, args.K)  # adj matrix
            log_lik_iw_node = -1 * get_rec_loss(norm=norm,
                                                pos_weight=pos_weight,
                                                pred=reconstruct_node_logits,
                                                labels=adj_orig_tile,
                                                loss_type="bce")

            node_log_prior_iw_vec = -0.5 * torch.sum(torch.square(node_z_samples_iw), 2)
            node_log_prior_iw = torch.mean(node_log_prior_iw_vec, 0)

            features_tile = batch['features'].unsqueeze(-1).expand(-1, -1, args.K)  # feature matrix
            log_lik_iw_attr = -1 * get_rec_loss(norm=norm_a,
                                                pos_weight=pos_weight_a,
                                                pred=reconstruct_attr_logits,
                                                labels=features_tile,
                                                loss_type="mse") # "mse"

            attr_log_prior_iw_vec = -0.5 * torch.sum(torch.square(attr_z_samples_iw), 2)
            attr_log_prior_iw = torch.mean(attr_log_prior_iw_vec, 0)

            loss = - torch.logsumexp(
                log_lik_iw_node +
                log_lik_iw_attr +
                node_log_prior_iw * warmup / num_nodes +
                attr_log_prior_iw * warmup / num_features -
                log_H_iw * warmup / (num_nodes + num_features), dim=0) + np.log(args.K)
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()
            threshold = 0

        if epoch > threshold:
            with torch.no_grad():
                model.eval()
                merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
                merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
                reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(
                    x=features_test,
                    adj=adj_norm_test)
            roc_curr_val, ap_curr_val = get_roc_score_node(emb=node_mu_iw_vec.detach().cpu().numpy(),
                                                            edges_pos=val_edges,
                                                            edges_neg=val_edges_false,
                                                            adj=adj_orig)

            roc_curr_test, ap_curr_test = get_roc_score_node(emb=node_mu_iw_vec.detach().cpu().numpy(),
                                                                edges_pos=test_edges,
                                                                edges_neg=test_edges_false,
                                                                adj=adj_orig)

            print("Epoch:", '%04d' % epoch, "val_ap=", "{:.5f}".format(ap_curr_val))
            print("Epoch:", '%04d' % epoch, "val_roc=", "{:.5f}".format(roc_curr_val))
            print("Epoch:", '%04d' % epoch, "test_ap=", "{:.5f}".format(ap_curr_test))
            print("Epoch:", '%04d' % epoch, "test_roc=", "{:.5f}".format(roc_curr_test))
            print('--------------------------------')

            if roc_curr_val > best_roc_val and ap_curr_val > best_ap_val:
                best_roc_val = roc_curr_val
                best_ap_val = ap_curr_val
                best_roc_test = roc_curr_test
                best_ap_test = ap_curr_test
                # torch.save(model,"model.pth")
                latent_feature = copy.deepcopy(reconstruct_attr_logits)

    print("val_roc:", '{:.5f}'.format(best_roc_val), "val_ap=", "{:.5f}".format(best_ap_val))
    print("test_roc:", '{:.5f}'.format(best_roc_test), "test_ap=", "{:.5f}".format(best_ap_test))

    recovered = latent_feature.detach().cpu().numpy()[:,:,0]

    recovered, recovered_new = get_recovered(test_edges, test_edges_false, recovered)

    return recovered_new




def normal_operation(adj_train, fea_train, adj_label, num_nodes, num_features, features_test, adj_norm_test, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig,args):


    best_roc_val = 0
    best_ap_val = 0
    best_roc_test = 0
    best_ap_test = 0

    model = GraphCIM(num_nodes=num_nodes,
                                input_dim=num_features,
                                num_hidden=512,
                                out_dim=512,
                                noise_dim=5,
                                dropout=0,
                                K=args.K,
                                J=args.J,
                                device=args.device)

    model.to(args.device)
    optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=2e-5)
    for epoch in range(1, args.epoch + 1):
        print("epoch:",epoch)
        # Graph augmentation
        adj_train_aug = adj_augment(adj_mat=adj_train, aug_prob=0)
        fea_train_aug = attr_augment(attr_mat=fea_train, aug_prob=0)

        adj_norm, pos_weight, norm, features, pos_weight_a, norm_a = prepare_inputs(adj=adj_train_aug,
                                                                                    features=fea_train_aug)
        adj_norm = adj_norm.to(args.device)
        features = features.to(args.device)

        warmup = np.min([epoch / 300., 1.])
        model.train()
        optimizer.zero_grad()
        merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
        merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
        reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(x=features,
                                                                                                    adj=adj_norm)

        node_attr_mu = torch.cat((merged_node_mu, merged_attr_mu), 0)
        node_attr_sigma = torch.cat((merged_node_sigma, merged_attr_sigma), 0)
        node_attr_z_samples = torch.cat((merged_node_z_samples, merged_attr_z_samples), 0)
        node_attr_logv_iw = torch.cat((node_logv_iw, attr_logv_iw), 0)

        ker = torch.exp(
            -0.5 * (torch.sum(
                torch.square(node_attr_z_samples - node_attr_mu) / torch.square(node_attr_sigma + args.eps), 3)))

        log_H_iw_vec = torch.log(torch.mean(ker, 2) + args.eps) - 0.5 * torch.sum(node_attr_logv_iw, 2)
        log_H_iw = torch.mean(log_H_iw_vec, 0)

        adj_orig_tile = adj_label.unsqueeze(-1).expand(-1, -1, args.K)  # adj matrix
        log_lik_iw_node = -1 * get_rec_loss(norm=norm,
                                            pos_weight=pos_weight,
                                            pred=reconstruct_node_logits,
                                            labels=adj_orig_tile,
                                            loss_type="bce")

        node_log_prior_iw_vec = -0.5 * torch.sum(torch.square(node_z_samples_iw), 2)
        node_log_prior_iw = torch.mean(node_log_prior_iw_vec, 0)

        features_tile = features.unsqueeze(-1).expand(-1, -1, args.K)  # feature matrix
        log_lik_iw_attr = -1 * get_rec_loss(norm=norm_a,
                                            pos_weight=pos_weight_a,
                                            pred=reconstruct_attr_logits,
                                            labels=features_tile,
                                            loss_type="mse") # "mse"

        attr_log_prior_iw_vec = -0.5 * torch.sum(torch.square(attr_z_samples_iw), 2)
        attr_log_prior_iw = torch.mean(attr_log_prior_iw_vec, 0)

        loss = - torch.logsumexp(
            log_lik_iw_node +
            log_lik_iw_attr +
            node_log_prior_iw * warmup / num_nodes +
            attr_log_prior_iw * warmup / num_features -
            log_H_iw * warmup / (num_nodes + num_features), dim=0) + np.log(args.K)
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()
        threshold = 0

        if epoch > threshold:
            with torch.no_grad():
                model.eval()
                merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
                merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
                reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(
                    x=features_test,
                    adj=adj_norm_test)
            roc_curr_val, ap_curr_val = get_roc_score_node(emb=node_mu_iw_vec.detach().cpu().numpy(),
                                                            edges_pos=val_edges,
                                                            edges_neg=val_edges_false,
                                                            adj=adj_orig)

            roc_curr_test, ap_curr_test = get_roc_score_node(emb=node_mu_iw_vec.detach().cpu().numpy(),
                                                                edges_pos=test_edges,
                                                                edges_neg=test_edges_false,
                                                                adj=adj_orig)

            print("Epoch:", '%04d' % epoch, "val_ap=", "{:.5f}".format(ap_curr_val))
            print("Epoch:", '%04d' % epoch, "val_roc=", "{:.5f}".format(roc_curr_val))
            print("Epoch:", '%04d' % epoch, "test_ap=", "{:.5f}".format(ap_curr_test))
            print("Epoch:", '%04d' % epoch, "test_roc=", "{:.5f}".format(roc_curr_test))
            print('--------------------------------')

            if roc_curr_val > best_roc_val and ap_curr_val > best_ap_val:
                best_roc_val = roc_curr_val
                best_ap_val = ap_curr_val
                best_roc_test = roc_curr_test
                best_ap_test = ap_curr_test
                # torch.save(model,"model.pth")
                latent_feature = copy.deepcopy(reconstruct_attr_logits)

    print("val_roc:", '{:.5f}'.format(best_roc_val), "val_ap=", "{:.5f}".format(best_ap_val))    
    print("test_roc:", '{:.5f}'.format(best_roc_test), "test_ap=", "{:.5f}".format(best_ap_test))

    recovered = latent_feature.detach().cpu().numpy()[:,:,0]

    recovered, recovered_new = get_recovered(test_edges, test_edges_false, recovered)
    return recovered_new

def main(args):
    seed = args.seed
    set_random_seed(seed=seed)
    exp_df = pd.read_csv(open(args.exp))
    adj_df = pd.read_csv(open(args.adj))
    cell_pixel = args.cell_pixel
    cell_type = pd.read_csv(open(args.cell_type))
    cell_id = exp_df.iloc[:,:1]
    coord = pd.read_csv(open(args.coord))
    batch = args.batch

    exp_df = exp_df.iloc[:, 1:]
    features, adj = exp_df.values, adj_df.values

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj) # 转换为稀疏矩阵

    num_nodes, num_features = features.shape

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    fea_train = features

    adj_norm_test, pos_weight_test, norm_test, features_test, pos_weight_a_test, norm_a_test = prepare_inputs(
        adj=adj_train, features=fea_train)
    adj_norm_test = adj_norm_test.to(args.device)
    features_test = features_test.to(args.device)

    adj_label = adj_train + sp.eye(adj_train.shape[0])  # self-loop
    adj_label = torch.FloatTensor(adj_label.toarray()).to(args.device)
    set_random_seed(seed=seed)
    if batch == 0:
        recovered = normal_operation(adj_train, fea_train, adj_label, num_nodes, num_features, features_test, adj_norm_test, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig, args)
    else:
        recovered = cycle_operation(adj_train, fea_train, adj_label, num_nodes, num_features, features_test, adj_norm_test, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig, args)

    map_cell(cell_pixel, cell_type, coord, recovered.values, cell_id)



if __name__ == '__main__':
    args = get_args()
    main(args)
