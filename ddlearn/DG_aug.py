# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from loss_dp import DPLoss
import network as net
from contrastive_loss_m import SupConLoss_m


class DDLearn(nn.Module):
    def __init__(self, n_feature, n_act_class, n_aug_class, dataset, dp):
        super(DDLearn, self).__init__()
        self.n_feature = n_feature
        self.n_act_class = n_act_class
        self.n_aug_class = n_aug_class
        self.dataset = dataset
        self.dp = dp
        self.feature_module = net.Network(n_feature, dataset)
        self.act_cls = nn.Linear(n_feature, n_act_class)
        self.aug_cls = nn.Linear(n_feature, n_aug_class)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_a = nn.CrossEntropyLoss()
        self.con = SupConLoss_m(contrast_mode='all')

        self.params = [
            {'params': self.feature_module.parameters()},
            {'params': self.act_cls.parameters()},
            {'params': self.aug_cls.parameters()},
        ]

    def forward(self, x_ori, x_onlyaug, labels):
        """the forward of model
        """
        actlabel_ori, actlabel_aug, auglabel_ori, auglabel_aug = labels
        feature_ori = self.feature_module(x_ori)
        auglabel_true = torch.cat((auglabel_ori, auglabel_aug), dim=0)
        feature_aug = self.feature_module(x_onlyaug)
        feature_aug_task = torch.cat((feature_ori, feature_aug), dim=0)
        auglabel_p = self.predict_aug(feature_aug_task)
        feature_act_task = feature_aug_task
        actlabel_true = torch.cat((actlabel_ori, actlabel_aug), dim=0)
        actlabel_p = self.predict_act(feature_act_task)

        loss_c = self.criterion(actlabel_p, actlabel_true)

        loss_selfsup = self.criterion_a(auglabel_p, auglabel_true)
        loss_dp = torch.zeros(1).cuda()
        if self.dp != 'no':
            dp_layer = DPLoss(
                loss_type=self.dp, input_dim=self.n_feature)
            loss_dp = dp_layer.compute(feature_ori, feature_aug)

        con_loss = self.con(torch.cat([feature_ori.unsqueeze(1), feature_aug.unsqueeze(
            1)], dim=1), torch.cat([actlabel_ori, actlabel_aug]))
        return actlabel_p, loss_c, loss_selfsup, loss_dp, con_loss

    def test_predict(self, x_ori, x_aug):
        actlabel_p = self.act_cls(self.feature_module(x_ori))
        auglabel_p = self.aug_cls(self.feature_module(
            torch.cat((x_ori, x_aug), dim=0)))
        return actlabel_p, auglabel_p

    def predict_act(self, feature):
        act_predict = self.act_cls(feature)
        return act_predict

    def predict_aug(self, feature):
        aug_predict = self.aug_cls(feature)
        return aug_predict
