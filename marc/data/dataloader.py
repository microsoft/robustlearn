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

import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100


# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std':[0.229, 0.224, 0.225]
    }
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, dataset, transform=None, meta=False):
        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        # save the class frequency
        if 'train' in txt and not meta:
            if not os.path.exists('cls_freq'):
                os.makedirs('cls_freq')
            freq_path = os.path.join('cls_freq', dataset + '.json')
            self.img_num_per_cls = [0 for _ in range(max(self.labels)+1)]
            for cls in self.labels:
                self.img_num_per_cls[cls] += 1
            with open(freq_path, 'w') as fd:
                json.dump(self.img_num_per_cls, fd)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index


# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, cifar_imb_ratio=None, meta=False):
    if phase == 'train_plain':
        txt_split = 'train'
    elif phase == 'train_val':
        txt_split = 'val'
        phase = 'train'
    else:
        txt_split = phase
    txt = '/media/Cygnus/haoc/longtail/%s/%s_%s.txt'%(dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))


    if dataset == 'iNaturalist18':
        print('===> Loading iNaturalist18 statistics')
        key = 'iNaturalist18'
    else:
        key = 'default'

    if dataset == 'CIFAR10_LT':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    elif dataset == 'CIFAR100_LT':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    else:
        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
        if phase not in ['train', 'val']:
            transform = get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key)

        print('Use data transformation:', transform)
    
        set_ = LT_Dataset(data_root, txt, dataset, transform, meta)

    print(len(set_))

    if sampler_dic and phase == 'train' and sampler_dic.get('batch_sampler', False):
        print('Using sampler: ', sampler_dic['sampler'])
        return DataLoader(dataset=set_,
                           batch_sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)

    elif sampler_dic and (phase == 'train' or meta):
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
