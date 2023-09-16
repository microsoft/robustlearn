# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn


class Network(nn.Module):
    def __init__(self, n_feature=64, dataset='dsads'):
        super(Network, self).__init__()
        self.dataset = dataset
        self.var_size = {
            'dsads': {
                'in_size': 45,
                'ker_size': 9,
                'fc_size': 32*25
            },
            'pamap': {
                'in_size': 27,
                'ker_size': 9,
                'fc_size': 32*122
            },
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.var_size[self.dataset]['in_size'], out_channels=16, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=self.var_size[self.dataset]['fc_size'], out_features=n_feature),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(-1, self.var_size[self.dataset]['fc_size'])
        feature = self.fc1(x)
        return feature


class Network_usc(nn.Module):
    def __init__(self, n_feature=64, dataset='uschad'):
        super(Network_usc, self).__init__()
        self.dataset = dataset
        self.var_size = {
            'uschad': {
                'in_size': 6,
                'ker_size': 6,
                'fc_size': 64*58
            },
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.var_size[self.dataset]['in_size'], out_channels=16, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
                1, self.var_size[self.dataset]['ker_size'])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=self.var_size[self.dataset]['fc_size'], out_features=n_feature),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, self.var_size[self.dataset]['fc_size'])
        feature = self.fc1(x)
        return feature
