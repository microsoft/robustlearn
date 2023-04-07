# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
from utils import *
from os import path
import json


class MARCLinear(nn.Module):
    """
    A wrapper for nn.Linear with support of MARC method.
    """

    def __init__(self, in_features, out_features, cls_freq):
        super().__init__()
        with open(cls_freq, 'r') as fd:
            freq = json.load(fd)
        self.freq = torch.tensor(freq)
        self.fc = nn.Linear(in_features, out_features)
        self.a = torch.nn.Parameter(torch.ones(1, out_features))
        self.b = torch.nn.Parameter(torch.zeros(1, out_features))
    
    def forward(self, input, *args):
        # logit_before = F.linear(input, self.weight, self.bias)
        with torch.no_grad():
            logit_before = self.fc(input)
            w_norm = torch.norm(self.fc.weight, dim=1)
        logit_after = self.a * logit_before + self.b * w_norm
        return logit_after, None


def create_model(feat_dim, num_classes=1000, stage1_weights=False,
                 model_dir=None, test=False, cls_freq=None, *args):
    print('Loading MARC Classifier.')
    clf = MARCLinear(feat_dim, num_classes, cls_freq)

    if not test:
        if stage1_weights:
            #assert (dataset)
            print('==> Loading classifier weights from %s' % model_dir)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=path.join(model_dir, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
