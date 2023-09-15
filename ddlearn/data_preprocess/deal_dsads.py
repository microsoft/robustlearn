# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import os
# dsads sampling rate is 25Hz, the time window size is 5s, thus, a window consists of 125 readings
# merge data from a subject and sliding window, overlap=50%


def load_data(root_path, winsize, overlapsize):
    act_file_list = os.listdir(root_path)
    act_num = len(act_file_list)
    # merge data/ a01-a19
    x_all, y_all, s_all = [], [], []
    for act in range(act_num):
        act_file = act_file_list[act]
        aa = act_file.split('a')[1]
        if aa[0] == '0':
            activity = int(aa[1])
        else:
            activity = int(aa)
        ac_path = os.path.join(root_path, act_file)
        sub_file_list = os.listdir(ac_path)
        sub_num = len(sub_file_list)
        # merge a/ p1-p8
        x_a, y_a, s_a = [], [], []
        for subnum in range(sub_num):
            sub = sub_file_list[subnum]
            subject = int(sub.split('p')[1])
            sub_path = os.path.join(ac_path, sub)
            subname_list = os.listdir(sub_path)
            #  merge p/ s01-s60 as data_sub
            data_sub = np.zeros((1, 45))
            for j in range(len(subname_list)):
                data_i = []
                if j < 9:
                    name = '0' + str(j+1)
                else:
                    name = str(j+1)
                data_i = np.loadtxt(os.path.join(
                    sub_path, 's'+name+'.txt'), delimiter=',')
                data_sub = np.vstack((data_sub, data_i))
            data_sub = data_sub[1:]
            x_s, y_s, s_s = getwin(
                data_sub, activity, subject, winsize, overlapsize)
            if subnum == 0:
                x_a, y_a, s_a = x_s, y_s, s_s
            else:
                x_a, y_a, s_a = np.vstack((x_a, x_s)), np.vstack(
                    (y_a, y_s)), np.vstack((s_a, s_s))
        if act == 0:
            x_all, y_all, s_all = x_a, y_a, s_a
        else:
            x_all, y_all, s_all = np.vstack((x_all, x_a)), np.vstack(
                (y_all, y_a)), np.vstack((s_all, s_a))
    return x_all, y_all, s_all


def getwin(x, y, s, winsize=125, overlapsize=63):
    l = len(x)
    stepsize = winsize-overlapsize
    h, t = 0, winsize
    xx, yy = [], []
    while t <= l:
        xx.append(x[h:t, :])
        yy.append(y)
        h += stepsize
        t += stepsize
    ss = np.ones(len(yy)) * s
    return np.array(xx), np.array(yy).reshape(-1, 1), np.array(ss).reshape(-1, 1)


if __name__ == '__main__':
    root_path = "/home/data/process/raw/dsads/data/"
    save_path = '/home/data/process/'
    winsize = 125
    overlapsize = 63
    x, y, s = load_data(root_path, winsize, overlapsize)
    if os.path.exists(save_path+'dsads_processwin.npz'):
        pass
    else:
        np.savez(save_path+'dsads_processwin.npz', x=x, y=y, s=s)
