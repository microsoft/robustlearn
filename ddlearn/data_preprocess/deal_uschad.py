# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import scipy.io
import os


def getwin(x, y, s, winsize, overlapsize):
    l = len(x)
    stepsize = winsize-overlapsize
    h, t = 0, winsize
    xx, yy, ss = [], [], []
    while t <= l:
        ry = np.unique(y[h:t])
        rs = np.unique(s[h:t])
        if len(ry) == 1 and len(rs) == 1:
            xx.append(x[h:t, :])
            yy.append(y[h])
            ss.append(s[h])
        else:
            print("error!")
        h += stepsize
        t += stepsize
    return np.array(xx), np.array(yy).reshape(-1, 1), np.array(ss).reshape(-1, 1)


def get_npy(root_path, save_path, winsize, overlapsize):
    if os.path.exists(save_path+'uschad_processwin.npz'):
        pass
    else:
        x, y, s = get_raw_data_deal(root_path, winsize, overlapsize)
        np.savez(save_path+'uschad_processwin.npz', x=x, y=y, s=s)


def get_raw_data_deal(root_path, winsize, overlapsize):
    file_name = os.listdir(root_path)
    x_all, y_all, s_all = np.zeros(
        (1, winsize, 6)), np.zeros((1, 1)), np.zeros((1, 1))
    sub_folder_list = []
    for i in file_name:
        if i == 'Readme.txt' or i == 'displayData_acc.m' or i == 'displayData_gyro.m':
            continue
        else:
            sub_folder_list.append(i)
    for subfolder in sub_folder_list:
        sub = subfolder.split('t')[1]
        path = os.path.join(root_path, subfolder)
        file_list = os.listdir(path)
        for file in file_list:
            data = scipy.io.loadmat(os.path.join(path, file))
            x, act_num = data['sensor_readings'], data['activity_number'] if 'activity_number' in data else data['activity_numbr']
            y = np.ones(x.shape[0]) * int(act_num[0])
            s = np.ones(x.shape[0]) * int(sub)
            tx, ty, ts = getwin(x, y, s, winsize, overlapsize)
            x_all, y_all, s_all = np.vstack((x_all, tx)), np.vstack(
                (y_all, ty)), np.vstack((s_all, ts))
    x_all, y_all, s_all = x_all[1:], y_all[1:], s_all[1:]
    return x_all, y_all, s_all


if __name__ == '__main__':
    winsize = 500
    overlap = 0.5
    overlapsize = int(winsize*overlap)
    root_path = '/home/data/usc-had/raw/USC-HAD/'
    save_path = '/home/data/process/uschad/'
    x, y, s = get_raw_data_deal(root_path, winsize, overlapsize)
    get_npy(root_path, save_path, winsize, overlapsize)
