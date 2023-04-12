# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def get_params(alg, args, nettype):
    init_lr = args.lr
    if nettype == 'Diversify-adv':
        params = [
            {'params': alg.dbottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.dclassifier.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.ddiscriminator.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
        return params
    elif nettype == 'Diversify-cls':
        params = [
            {'params': alg.bottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.discriminator.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
        return params
    elif nettype == 'Diversify-all':
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.abottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
            {'params': alg.aclassifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
        return params


def get_optimizer(alg, args, nettype):
    params = get_params(alg, args, nettype=nettype)
    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, 0.9))
    return optimizer
