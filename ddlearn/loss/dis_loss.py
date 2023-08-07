# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, out_dim=1):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x


def dis(source, target, input_dim=256, hidden_dim=512, dis_net=None):
    """discrimination loss for two domains.
    source and target are features. 
    """
    domain_loss = nn.BCELoss()
    dis_net = Discriminator(input_dim, hidden_dim).cuda()
    domain_src = torch.ones(len(source)).cuda()
    domain_tar = torch.zeros(len(target)).cuda()
    domain_src, domain_tar = domain_src.view(
        domain_src.shape[0], 1), domain_tar.view(domain_tar.shape[0], 1)
    pred_src = dis_net(source)
    pred_tar = dis_net(target)
    loss_s, loss_t = domain_loss(
        pred_src, domain_src), domain_loss(pred_tar, domain_tar)
    loss = loss_s + loss_t
    return loss
