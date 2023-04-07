# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
import json


class GRWCrossEntropyLoss(_WeightedLoss):
    """
    Generalized Reweight Loss, introduced in
    Distribution Alignment: A Unified Framework for Long-tail Visual Recognition
    https://arxiv.org/abs/2103.16370
    """
    __constants__ = ['ignore_index', 'reduction']

    def _init_weights(self, freq_path, num_classes=1000, exp_scale=1.2):
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        assert len(freq) > 0, "num_samples_list is empty"

        num_shots = np.array(freq)
        ratio_list = num_shots / np.sum(num_shots)
        exp_reweight = 1 / (ratio_list ** exp_scale)
        exp_reweight = exp_reweight / np.sum(exp_reweight)
        exp_reweight = exp_reweight * num_classes
        exp_reweight = torch.tensor(exp_reweight).float()
        return exp_reweight

    def __init__(
            self,
            freq_path=None,
            exp_scale=1.2,
            size_average=None,
            ignore_index=-100,
            reduce=None,
            reduction='mean',
            num_classes=1000
    ):
        weights_init = self._init_weights(
            freq_path=freq_path,
            num_classes=num_classes,
            exp_scale=exp_scale)
        super(GRWCrossEntropyLoss, self).__init__(weights_init, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target, features, classifier):
        if self.weight.device != input.device:
            self.weight.to(input.device)
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )


def create_loss(exp_scale=0, num_classes=-1, freq_path=None):
    print('Loading GRW Loss.')
    return GRWCrossEntropyLoss(freq_path, exp_scale, num_classes=num_classes)
