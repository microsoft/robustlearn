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

import torch.nn as nn
from utils import *
from os import path


class DotProduct_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, use_route=False, *args):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = nn.Linear(feat_dim, num_classes)
        self.use_route = use_route


    def forward(self, x, *args):
        x = self.fc(x)
        if self.use_route:
            return x, x
        else:
            return x, None


def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None,
                 log_dir=None, test=False, use_route=False, model_dir=None, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim, use_route)

    if not test:
        if stage1_weights:
            assert (dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            print('==> Loading classifier weights from %s' % model_dir)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=path.join(model_dir, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
