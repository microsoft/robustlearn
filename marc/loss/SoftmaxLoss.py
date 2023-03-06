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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math


class CrossEntropyLoss(_Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.centers = None
    def forward(self, logits, y, features, classifier):
        loss = F.cross_entropy(logits, y, reduction='mean')
        return loss



def create_loss():
    print('Loading Softmax Loss.')
    return CrossEntropyLoss()
