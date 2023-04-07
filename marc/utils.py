# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import importlib
from torch.optim.lr_scheduler import _LRScheduler
import pdb
import math
import torch.nn as nn


class CosineAnnealingLRWarmup(_LRScheduler):
    """
    Cosine Annealing with Warm Up.
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=5, base_lr=0.05, warmup_lr=0.1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch)

    def get_cos_lr(self):
        return [self.eta_min + (self.warmup_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_warmup_lr(self):
        return [((self.warmup_lr - self.base_lr) / (self.warmup_epochs - 1) * (self.last_epoch - 1)
                 + self.base_lr) / self.base_lr * base_lr
                for base_lr in self.base_lrs]

    def get_lr(self):
        assert self.warmup_epochs >= 2
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_lr()
        else:
            return self.get_cos_lr()


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def print_write(print_str, log_file):
    print(*print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(*print_str, file=f)


def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights)
    return model


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def weighted_shot_acc(preds, labels, ws, train_data, many_shot_thr=100, low_shot_thr=20):
    training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(ws[labels == l].sum())
        class_correct.append(((preds[labels == l] == labels[labels == l]) * ws[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def F_measure(preds, labels, openset=False, theta=None):
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.

        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 and preds[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')


def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                        + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def weighted_mic_acc_cal(preds, labels, ws):
    acc_mic_top1 = ws[preds == labels].sum() / ws.sum()
    return acc_mic_top1


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


# def dataset_dist (in_loader):

#     """Example, dataset_dist(data['train'][0])"""

#     label_list = np.array([x[1] for x in in_loader.dataset.samples])
#     total_num = len(data_list)

#     distribution = []
#     for l in np.unique(label_list):
#         distribution.append((l, len(label_list[label_list == l])/total_num))

#     return distribution


# New Added
def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x


def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score


def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce


def get_priority(ptype, logits, labels):
    if ptype == 'score':
        ws = 1 - logits2score(logits, labels)
    elif ptype == 'entropy':
        ws = logits2entropy(logits)
    elif ptype == 'CE':
        ws = logits2CE(logits, labels)

    return ws


def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv


def calculate_prior(num_classes, img_max=None, prior=None, prior_txt=None, reverse=False, return_num=False):
    if prior_txt:
        labels = []
        with open(prior_txt) as f:
            for line in f:
                labels.append(int(line.split()[1]))
        occur_dict = dict(Counter(labels))
        img_num_per_cls = [occur_dict[i] for i in range(num_classes)]
    else:
        img_num_per_cls = []
        for cls_idx in range(num_classes):
            if reverse:
                num = img_max * (prior ** ((num_classes - 1 - cls_idx) / (num_classes - 1.0)))
            else:
                num = img_max * (prior ** (cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    img_num_per_cls = torch.Tensor(img_num_per_cls)

    if return_num:
        return img_num_per_cls
    else:
        return img_num_per_cls / img_num_per_cls.sum()
