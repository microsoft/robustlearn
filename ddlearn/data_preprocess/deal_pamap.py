# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import os
from math import isnan


def load_data(root_path, winsize, overlap):
    file_list = os.listdir(root_path)
    list_len = len(file_list)
    x_all, y_all, s_all = [], [], []
    for filenum in range(list_len):
        data_i = []
        filename = file_list[filenum]
        data_i = np.loadtxt(os.path.join(root_path, filename))
        subject = int(filename.split('0')[1].split('.')[0])
        x_i = np.hstack((data_i[:, 4:7], data_i[:, 7:13], data_i[:, 21:24],
                         data_i[:, 27:33], data_i[:, 38:41], data_i[:, 44:50]))
        y_i = data_i[:, 1]
        tx, ty, ts = getwin_replace(x_i, y_i, subject,
                                    winsize=winsize, overlap=overlap)
        if filenum == 0:
            x_all, y_all, s_all = tx, ty, ts
        else:
            x_all = np.vstack((x_all, tx))
            y_all = np.vstack((y_all, ty))
            s_all = np.vstack((s_all, ts))
        print('a')
    return x_all, y_all, s_all


def getwin_replace(x, y, s, winsize, overlap):
    data_num = len(x)
    overlap_size = int(winsize*overlap)
    stepsize = winsize-overlap_size
    head, tail = 0, winsize
    xx, yy = [], []
    while tail <= data_num:
        ry = np.unique(y[head:tail])
        if len(ry) == 1:
            x_win = x[head:tail, :]
            x_new = replace_nan(x_win)
            xx.append(x_new)
            yy.append(y[head])
            head += stepsize
            tail += stepsize
        else:
            head = tail-1
            while y[head] == y[head-1]:
                head -= 1
            tail = head + winsize
    ss = np.ones(len(yy)) * s
    return np.array(xx), np.array(yy).reshape(-1, 1), np.array(ss).reshape(-1, 1)


def replace_nan(x_win):
    x_new = []
    for col in range(x_win.shape[1]):
        x_col = x_win[:, col]
        x_col_mean = calculate_mean_value(x_col)
        index_nan = np.argwhere(np.isnan(x_col))
        x_col[index_nan] = x_col_mean
        if col == 0:
            x_new = x_col.reshape(-1, 1)
        else:
            x_new = np.hstack((x_new, x_col.reshape(-1, 1)))
    return x_new


def calculate_mean_value(x):
    x_new = []
    for x_i in x:
        if isnan(x_i):
            continue
        else:
            x_new.append(x_i)
    x_mean = np.mean(np.array(x_new), axis=0)
    return x_mean


def get_pamap_npy(root_path, save_path, winsize, overlap):
    if os.path.exists(save_path+'pamap_processwin.npz'):
        pass
    else:
        x, y, s = load_data(root_path, winsize, overlap)
        np.savez(save_path+'pamap_processwin.npz', x=x, y=y, s=s)


if __name__ == '__main__':
    root_path = '/home/data/process/raw/PAMAP/PAMAP2_Dataset/Protocol/'
    save_path = '/home/data/process/pamap/'
    winsize = 512
    overlap = 0.5
    get_pamap_npy(root_path, save_path, winsize, overlap)
