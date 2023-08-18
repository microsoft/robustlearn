# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd

import random

import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class TinyImageNet(Dataset):
    def __init__(self, dataset_type, transform=None):
        self.root = "./data/tiny-imagenet-200/"
        data_path = os.path.join(self.root, dataset_type)

        self.dataset = torchvision.datasets.ImageFolder(root=data_path)

        self.transform = transform

    def __getitem__(self, index):
        img, targets = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return self.dataset.__len__()


class TinyImageNetC(Dataset):
    def __init__(self, name, data_dir="./data/Tiny-ImageNet-C", level=1):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost"
        ]

        assert name in self.corruptions
        self.root = data_dir
        data_path = os.path.join(self.root, name+"/"+str(level))

        self.dataset = torchvision.datasets.ImageFolder(root=data_path)
        # print(self.dataset.__len__())
        # print(len(self.dataset.classes))
        # print(self.dataset.classes)
        # print(self.dataset.class_to_idx)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        img, targets = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return self.dataset.__len__()


class CIFAR100C(Dataset):
    def __init__(self, name, data_dir="./data/CIFAR-100-C"):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost"
        ]

        assert name in self.corruptions
        self.root = data_dir
        data_path = os.path.join(self.root, name + '.npy')
        target_path = os.path.join(self.root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)


class CIFAR10C(Dataset):
    def __init__(self, name, data_dir="./data/CIFAR-10-C"):
        self.corruptions = [
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost"
        ]

        assert name in self.corruptions
        self.root = data_dir
        data_path = os.path.join(self.root, name + '.npy')
        target_path = os.path.join(self.root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)


class adv_dataset(Dataset):
    def __init__(self):
        self.images = None
        self.labels = None

    def append_data(self, images, labels):
        if self.images is None:
            self.images = images
            self.labels = labels
        else:
            # print(images.shape)
            # print(self.images.shape)

            self.images = torch.cat((self.images, images), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        # print(img.shape)
        return img, label

    def __len__(self):
        return self.images.shape[0]


class mini_imagenet_dataset(Dataset):
    def __init__(self, csv_dir, transform=None):
        img_label_pairs = pd.read_csv(csv_dir)

        self.imgs = img_label_pairs["filename"].values
        self.labels = img_label_pairs["label"].values

        self.transform = transform

        label_set = sorted(set(img_label_pairs["label"].drop_duplicates().values))
        self.label2idx = {}
        self.idx2label = {}

        for v, k in enumerate(label_set):
            self.label2idx[k] = v
            self.idx2label[v] = k

    def __getitem__(self, item):
        img = Image.open("./data/mini-imagenet/images/" + self.imgs[item][:9] + "/" + self.imgs[item]).convert("RGB")
        label = self.label2idx[self.labels[item]]
        return self.transform(img), torch.tensor(label)

    def __len__(self):
        return len(self.imgs)


def create_dataloader(dataset, batch_size, use_val=True, transform_dict=None, resize=None):
    if dataset == "TinyImageNet":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        train_dataset = TinyImageNet("train", transform_train)
        testset = TinyImageNet("val", transform_test)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        valloader = None
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)        

    if dataset == "CIFAR10":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

        if use_val:

            valid_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

            train_idx = np.loadtxt("./data/train_idx.txt", dtype=int)
            valid_idx = np.loadtxt("./data/val_idx.txt", dtype=int)

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=True,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
            valloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            valloader = None
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=8)

    if dataset == "CIFAR100":
        if transform_dict is not None:
            transform_train, transform_test = transform_dict["train"], transform_dict["test"]
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
        
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train, )

        # load the dataset
        if use_val:

            valid_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test, )

            indices = list(range(50000))
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[:45000], indices[45000:]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            trainloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=True,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
            valloader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                num_workers=8, worker_init_fn=seed_worker, generator=g
            )
        
        else:
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            valloader = None

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, valloader, testloader