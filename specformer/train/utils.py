import torch
from torchvision import datasets, transforms
import torch
import logging
import os
from collections import OrderedDict
from torch.utils.data.sampler import SubsetRandomSampler


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def normalize(args, X):
    if args.dataset=="cifar10":
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif args.dataset=="cifar100":
        mu = torch.tensor(cifar100_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar100_std).view(3, 1, 1).cuda()
    elif args.dataset=="imagenette" or args.dataset=="imagenet":
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    return (X - mu) / std


def get_loaders(args):
    if args.dataset=="cifar10":
        mean =cifar10_mean
        std = cifar10_std
    elif args.dataset=="cifar100":
        mean =cifar100_mean
        std = cifar100_std
    elif args.dataset=="imagenette" or args.dataset=="imagenet":
        mean = imagenet_mean
        std = imagenet_std
    train_list = [
        transforms.Resize([args.resize,args.resize]),
        transforms.RandomCrop(args.crop, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    train_list.append(transforms.ToTensor())
    train_list.append(transforms.Normalize(mean, std))
    train_transform = transforms.Compose(train_list)
    test_transform = transforms.Compose([
        transforms.Resize([args.resize,args.resize]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    num_workers = args.num_workers
    if args.dataset=="cifar10":
        train_dataset = datasets.CIFAR10(
        args.data_dir, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
        args.data_dir, train=False, transform=test_transform, download=True)
    if args.dataset=="cifar100":
        train_dataset = datasets.CIFAR100(
        args.data_dir, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
        args.data_dir, train=False, transform=test_transform, download=True)
    if args.dataset=="imagenette":
        train_dataset = datasets.ImageFolder(args.data_dir+"train/",train_transform)
        test_dataset = datasets.ImageFolder(args.data_dir+"val/",test_transform)
    if args.dataset == "imagenet":
        train_dataset = datasets.ImageFolder(args.data_dir+"train/",train_transform)
        test_dataset = datasets.ImageFolder(args.data_dir+"val/",test_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size // args.accum_steps*2,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

_logger = logging.getLogger(__name__)