# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import sys
import torchvision


def test_model(model, test_loader, model_file=None, batch_size=128):
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    test_ori_loader, test_aug_loader = test_loader
    iter_test_ori, iter_test_aug = iter(test_ori_loader), iter(test_aug_loader)
    model.eval()
    test_batch_num = len(test_ori_loader.dataset)//batch_size
    last_sample = len(test_ori_loader.dataset) % batch_size
    if last_sample > 0:
        test_batch_num = test_batch_num + 1
    with torch.no_grad():
        correct_act, total_act = 0, 0
        for iter_n in range(test_batch_num):
            x_ori, label_ori, auglabel_ori = next(iter_test_ori)
            x_aug, label_aug, auglabel_aug = next(iter_test_aug)
            if (iter_n == test_batch_num-1) and (last_sample > 0):
                x_aug, label_aug, auglabel_aug = x_aug[:x_ori.shape[0]
                                                       ], label_aug[:x_ori.shape[0]], auglabel_aug[:x_ori.shape[0]]
            x_ori, label_ori, auglabel_ori = x_ori.unsqueeze(2).permute(
                0, 3, 2, 1).cuda().float(), label_ori.cuda().long(), auglabel_ori.cuda().long()
            x_aug, label_aug, auglabel_aug = x_aug.unsqueeze(2).permute(
                0, 3, 2, 1).cuda().float(), label_aug.cuda().long(), auglabel_aug.cuda().long()
            act_label_p, _ = model.test_predict(x_ori, x_aug)
            _, predict_act = torch.max(act_label_p.data, 1)
            correct_act += (predict_act == label_ori).sum()
            total_act += label_ori.size(0)

    act_acc_test = float(correct_act) * 100 / total_act
    return act_acc_test


def write_file(file_name, content):
    with open(file_name, 'a+') as fp:
        fp.write(content + '\n')


def set_random_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))


def param_init(args):
    dataset = args.dataset
    if dataset == 'dsads':
        args.n_act_class = 19
        args.n_domain = 4
    elif dataset == 'pamap':
        args.n_act_class = 8
        args.n_domain = 4
    elif dataset == 'uschad':
        args.n_act_class = 12
        args.n_domain = 5
    else:
        print('No matching dataset.')
    return args


def record_losses(total_loss, loss):
    total_loss.append(loss.item())
    return total_loss


def record_acc(correct_act, total_act):
    acc_train_cls = float(correct_act) * 100.0 / total_act
    return acc_train_cls


def record_trainingacc_labels(labels_true, labels_p, correct_act, total_act):
    actlabel_ori, actlabel_aug = labels_true
    actlabel_p = labels_p
    _, predict_act = torch.max(actlabel_p.data, 1)
    act_label_all = torch.cat((actlabel_ori, actlabel_aug), dim=0)
    correct_act += (predict_act == act_label_all).sum()
    total_act += (actlabel_ori.size(0)+actlabel_aug.size(0))
    return correct_act, total_act
