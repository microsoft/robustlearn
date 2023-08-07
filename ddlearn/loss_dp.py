# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from loss import dis_loss


class DPLoss(object):
    def __init__(self, loss_type='dis', input_dim=512):
        self.loss_type = loss_type
        self.input_dim = input_dim

    def compute(self, X, Y):
        if self.loss_type == 'dis':
            loss = dis_loss.dis(X, Y, input_dim=self.input_dim, hidden_dim=60)
        return loss
