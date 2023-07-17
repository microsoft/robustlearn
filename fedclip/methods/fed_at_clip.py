# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.config import img_param_init, set_random_seed
import utils.clip_util as clu
from utils.prepare_data_dg_clip import *
import copy
import argparse
from nets.models import ClipModelat
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

def toeval(model):
    model.model.eval()
    model.img_adap.eval()


def totrain(model):
    model.model.train()
    model.img_adap.train()


def train(args, model, data_loader, optimizer, device):
    totrain(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    for batch in data_loader:

        image, text, label = batch

        if len(text) > 1:
            image = image.to(device)
            text = text.to(device)

            image_features = model.model.encode_image(image).float()
            text_features = model.model.encode_text(text).float()
            image_features_att = model.img_adap(image_features)
            image_features = torch.mul(image_features_att, image_features)

            image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

            logit_scale = model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(
                len(image), dtype=torch.long, device=device)

            loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(args, model, data_loader, device):
    toeval(model)
    total = 0
    correct = 0
    texts = model.labels
    text_features = clu.get_text_features_list(texts, model.model).float()

    with torch.no_grad():
        for batch in data_loader:

            image, _, label = batch
            image = image.to(device)
            label = label.to(device)
            image_features = clu.get_image_features(
                image, model.model, model.preprocess).float()
            image_features_attn = model.img_adap(image_features)
            image_features = torch.mul(
                image_features_attn, image_features).detach()
            similarity = clu.get_similarity(image_features, text_features)
            _, indices = similarity.topk(1)
            total += len(label)
            pred = torch.squeeze(indices)
            res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            res = res.cpu().numpy()
            correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

        return correct/total


def communication(args, server_model, models, client_weights):
    client_num = len(models)
    with torch.no_grad():
        for key in server_model.img_adap.state_dict().keys():
            if 'num_batches_tracked' in key or 'bert' in key:
                server_model.img_adap.state_dict()[key].data.copy_(
                    models[0].img_adap.state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.img_adap.state_dict()[
                                        key], dtype=torch.float32)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        temp += client_weights[client_idx] * \
                            models[client_idx].img_adap.state_dict()[key]
                server_model.img_adap.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    if client_idx not in args.test_envs:
                        models[client_idx].img_adap.state_dict()[key].data.copy_(
                            server_model.img_adap.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pacs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='../../../data/')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAtImg')
    parser.add_argument('--net', type=str, default='ViT-B/32',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)
    args.n_clients = 4
    args = img_param_init(args)
    os.makedirs('../data/', exist_ok=True)

    server_model = ClipModelat(
        args.net, imgadpy=True, freezepy=True)

    train_loaders, val_loaders, test_loaders = get_data(
        args.dataset)(args, server_model)

    server_model.initdgatal(test_loaders[0])

    client_num = len(test_loaders)
    sclient_num = client_num-len(args.test_envs)
    client_weights = [1/sclient_num for i in range(client_num)]
    models = [copy.deepcopy(server_model)for idx in range(client_num)]
    for i in range(client_num):
        models[i].model.to(device)
        models[i].img_adap.to(device)
    best_changed = False

    best_acc = [0. for j in range(client_num)]
    finalrecord = ''
    logrecord = ''

    for a_iter in range(args.iters):
        optimizers = [optim.Adam(params=[{'params': models[idx].img_adap.parameters()}], lr=args.lr, betas=(
            args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(
                wi + a_iter * args.wk_iters))
            logrecord += 'Train epoch:%d\n' % (wi + a_iter * args.wk_iters)
            for client_idx, model in enumerate(models):
                if client_idx in args.test_envs:
                    pass
                else:
                    train(
                        args, model, train_loaders[client_idx], optimizers[client_idx], device)
        with torch.no_grad():
            server_model, models = communication(
                args, server_model, models, client_weights)

            val_acc_list = [0. for j in range(client_num)]
            for client_idx, model in enumerate(models):
                if client_idx in args.test_envs:
                    pass
                else:
                    train_acc = test(
                        args, model, train_loaders[client_idx], device)
                    print(' Site-{:d}| Train Acc: {:.4f}'.format(
                        client_idx, train_acc))
                    logrecord += ' Site-{:d}| Train Acc: {:.4f}\n'.format(
                        client_idx, train_acc)

                    val_acc = test(
                        args, model, val_loaders[client_idx], device)
                    val_acc_list[client_idx] = val_acc
                    print(' Site-{:d}| Val  Acc: {:.4f}'.format(
                        client_idx, val_acc), flush=True)
                    logrecord += ' Site-{:d}| Val  Acc: {:.4f}\n'.format(
                        client_idx, val_acc)

            test_acc_list = [0. for j in range(client_num)]
            for client_idx in range(client_num):
                if client_idx in args.test_envs:
                    test_acc = test(args, server_model,
                                    test_loaders[client_idx], device)
                else:
                    test_acc = test(
                        args, models[client_idx], test_loaders[client_idx], device)
                print(
                    ' Test site-{:d}| Test Acc: {:.4f}'.format(client_idx, test_acc))
                logrecord += ' Test site-{:d}| Test Acc: {:.4f}'.format(
                    client_idx, test_acc)
                test_acc_list[client_idx] = test_acc

            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    if client_idx in args.test_envs:
                        pass
                    else:
                        best_acc[client_idx] = val_acc_list[client_idx]
                        best_epoch = a_iter
                        best_changed = True
            if best_changed:
                finalrecord = finalrecord+str(a_iter)+','
                for item in test_acc_list:
                    finalrecord = finalrecord+str(item)+','
                best_changed = False
    print('best epoch:%d\n' % (best_epoch))
    logrecord += '\n best epoch:%d\n' % (best_epoch)
    rec = finalrecord.split(',')[-5:-1]
    ts = ''
    for item in rec:
        ts += '%.4f ' % float(item)
    print('best test acc: '+ts)
    logrecord += 'best test acc: '+ts
    filename = 'results/clip_'+args.dataset+'_'+str(args.datapercent)+'/'+str(
        args.test_envs[0])+'_'+str(args.iters)+'-'+str(args.wk_iters)+'-'+args.mode+'_'+str(args.lr)
    filename = filename+'_'+args.net
    os.makedirs(filename, exist_ok=True)
    with open(filename+'/output.txt', 'w') as f:
        f.write(finalrecord)
    with open(filename+'/log.txt', 'w') as f:
        f.write(logrecord)
