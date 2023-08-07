# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from main import args_parse
import utils
import numpy as np
from data_augment import DA_Jitter, DA_Scaling, DA_MagWarp, DA_TimeWarp, DA_Rotation, DA_Permutation, DA_RandSampling
import pickle
from sklearn.model_selection import train_test_split


def raw_to_aug(seed, root_path, save_path, dataset, aug_num=7, remain_data_rate=0.2):
    data = np.load(
        root_path+f'{dataset}/{dataset}_subject_final_seed{seed}.pkl', allow_pickle=True)
    splitrate = str(remain_data_rate).split('.')
    rate_name = splitrate[0]+splitrate[1]
    aug_all = []
    raw_all = []
    for i in range(len(data)):
        tr_aug, va_aug, te_aug, domain_aug = [], [], [], []
        tr_raw, va_raw, te_raw, domain_raw = [], [], [], []
        x_train_i, y_train_i = pick_data(data, 'train', src=i)
        x_val_i, y_val_i = pick_data(data, 'val', src=i)
        x_test_i, y_test_i = pick_data(data, 'test', src=i)

        if remain_data_rate == 1.0:
            x_train_i_remain, y_train_i_remain = x_train_i, y_train_i
        else:
            x_train_i_remain, x_train_i_reduce, y_train_i_remain, y_train_i_reduce = train_test_split(
                x_train_i, y_train_i, test_size=1-remain_data_rate, random_state=seed, stratify=y_train_i)
        x_tr_aug, y_tr_aug, aug_label_tr = aug_per_sample_sensor(
            x_train_i_remain, y_train_i_remain, aug_num)
        x_va_aug, y_va_aug, aug_label_va = aug_per_sample_sensor(
            x_val_i, y_val_i, aug_num)
        x_te_aug, y_te_aug, aug_label_te = aug_per_sample_sensor(
            x_test_i, y_test_i, aug_num)
        tr_aug.append(x_tr_aug)
        tr_aug.append(y_tr_aug)
        tr_aug.append(aug_label_tr)

        va_aug.append(x_va_aug)
        va_aug.append(y_va_aug)
        va_aug.append(aug_label_va)

        te_aug.append(x_te_aug)
        te_aug.append(y_te_aug)
        te_aug.append(aug_label_te)

        domain_aug.append(tr_aug)
        domain_aug.append(va_aug)
        domain_aug.append(te_aug)

        aug_all.append(domain_aug)
        tr_raw.append(x_train_i_remain)
        tr_raw.append(y_train_i_remain)

        va_raw.append(x_val_i)
        va_raw.append(y_val_i)

        te_raw.append(x_test_i)
        te_raw.append(y_test_i)

        domain_raw.append(tr_raw)
        domain_raw.append(va_raw)
        domain_raw.append(te_raw)

        raw_all.append(domain_raw)

    print(len(aug_all))
    if save_path is not None:
        aug_save_path = save_path + \
            f'{dataset}/rawaug_fix_valtest/{dataset}_cross_subject_augment_rate{rate_name}_seed{seed}.pkl'
        raw_save_path = save_path + \
            f'{dataset}/rawaug_fix_valtest/{dataset}_cross_subject_raw_rate{rate_name}_seed{seed}.pkl'
        with open(aug_save_path, 'wb') as f:
            pickle.dump(aug_all, f)
        with open(raw_save_path, 'wb') as f:
            pickle.dump(raw_all, f)
    return raw_all, aug_all


def aug_per_sample_sensor(x, y, aug_num):
    colomn_num = x.shape[2]
    col3 = int(colomn_num/3)
    x_aug, y_aug, aug_label = [], [], []
    for i in range(len(x)):
        x_i = x[i, :, :]
        y_i = y[i]
        for j in range(col3):
            x_sensor_i = x_i[:, j*3:(j+1)*3]
            x_aug_col, y_aug_col, aug_label_col = augment(
                x_sensor_i, y_i, aug_num)
            if j == 0:
                x_aug_i, y_aug_i, aug_label_i = x_aug_col, y_aug_col, aug_label_col
            else:
                for type in range(len(x_aug_i)):
                    x_aug_i[type] = np.hstack((x_aug_i[type], x_aug_col[type]))

        for aug_type in range(aug_num):
            x_aug.append(x_aug_i[aug_type])
            y_aug.append(y_aug_i[aug_type][0])
            aug_label.append(aug_label_i[aug_type][0])
    return np.array(x_aug), np.array(y_aug), np.array(aug_label)


def augment(x, y, aug_num):
    aug_all_x, aug_all_y, aug_label = [], [], []
    jit_x = DA_Jitter(x, sigma=0.05)
    sca_x = DA_Scaling(x, sigma=0.1)
    mag_x = DA_MagWarp(x, sigma=0.2)
    tim_x = DA_TimeWarp(x, sigma=0.2)
    rot_x = DA_Rotation(x)
    per_x = DA_Permutation(x, nPerm=4, minSegLength=10)
    ran_x = DA_RandSampling(x, nSample_rate=0.4)
    aug_all_x.append(jit_x)
    aug_all_x.append(sca_x)
    aug_all_x.append(mag_x)
    aug_all_x.append(tim_x)
    aug_all_x.append(rot_x)
    aug_all_x.append(per_x)
    aug_all_x.append(ran_x)
    for aug in range(aug_num):
        aug_all_y.append(np.ones((x.shape[0])) * y)
        aug_label.append(np.ones((x.shape[0])) * aug)
    return aug_all_x, aug_all_y, aug_label


def pick_data(data, data_name, src):
    x, y = [], []
    if data_name == 'train':
        x = data[src][0][0]
        y = data[src][0][1]
    elif data_name == 'val':
        x = data[src][1][0]
        y = data[src][1][1]
    elif data_name == 'test':
        x = data[src][2][0]
        y = data[src][2][1]
    else:
        print('Error pick data!')
    y = y.reshape(-1, 1)
    return x, y


if __name__ == "__main__":
    args = args_parse()
    utils.set_random_seed(args.seed)
    raw_all, aug_all = raw_to_aug(seed=args.seed, root_path="/home/data/process/",
                                  save_path="/home/data/process/", dataset='dsads', aug_num=7, remain_data_rate=0.8)
