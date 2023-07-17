# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
import torch


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    else:
        print('No such dataset exists!')
    args.domains = domains
    if args.dataset == 'pacs':
        args.num_classes = 7
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
