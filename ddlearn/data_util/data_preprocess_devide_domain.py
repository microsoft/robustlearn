# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from raw_aug_loader import set_param
from main import args_parse
import utils
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


def merge_split_dsads(seed, root_path='/home/data/process/dsads/dsads_processwin.npz', n_domain=4, save_file='/home/data/process/dsads/dsads_subject_final.pkl'):
    d = np.load(root_path)
    x, y, s = d['x'], (d['y']-1).reshape(-1,), (d['s']-1).reshape(-1,)
    data_lst = []
    for i in range(n_domain):
        data_i = []
        d_index = np.argwhere((s == 2*i) | (s == 2*i+1)).reshape(-1,)
        x_i = x[d_index, :, :]
        y_i = y[d_index]
        data_i.append(x_i)
        data_i.append(y_i)
        data_lst.append(data_i)

    devide_train_val_test(data_lst, n_domain, save_file, seed)


def devide_train_val_test(data_lst, n_domain, save_file, seed):
    domain_dic = []
    for i in range(n_domain):
        train_i, val_i, test_i, dic_i = [], [], [], []
        x_i, y_i = data_lst[i][0], data_lst[i][1]
        x_train_i, x_tmp_i, y_train_i, y_tmp_i = train_test_split(
            x_i, y_i, test_size=0.4, random_state=seed, stratify=y_i)
        x_val_i, x_test_i, y_val_i, y_test_i = train_test_split(
            x_tmp_i, y_tmp_i, test_size=0.5, random_state=seed, stratify=y_tmp_i)
        train_i.append(x_train_i)
        train_i.append(y_train_i)

        val_i.append(x_val_i)
        val_i.append(y_val_i)

        test_i.append(x_test_i)
        test_i.append(y_test_i)

        dic_i.append(train_i)
        dic_i.append(val_i)
        dic_i.append(test_i)
        domain_dic.append(dic_i)
    with open(save_file, 'wb') as f:
        pickle.dump(domain_dic, f)


if __name__ == "__main__":
    args = args_parse()
    for dataset in ['dsads']:
        for args.seed in range(1, 4, 1):
            utils.set_random_seed(args.seed)
            root_path = '/home/data/process/' + \
                f'{dataset}/{dataset}_processwin.npz'
            save_file = '/home/data/process/' + \
                f'{dataset}/{dataset}_subject_final_seed{args.seed}.pkl'
            n_domain = set_param(dataset)
            if dataset == 'dsads':
                merge_split_dsads(args.seed, root_path, n_domain, save_file)
            else:
                print('error')
