# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import logging

import torch
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchattacks
from dataloader import CIFAR10C, CIFAR100C, TinyImageNetC


def interpolation(args, logger, init_sd, ft_sd, model, dataloader, criterion, save_dir, eval_robustness_func):
    alphas = np.arange(0, 1.1, 0.1)

    # alphas = np.arange(0, 0.21, 0.01)

    records = []

    for alpha in alphas:
        model_dict = {}
        for name, _ in init_sd.items():
            model_dict[name] = alpha * ft_sd[name] + (1 - alpha) * init_sd[name]

        torch.save(model_dict, save_dir + "finetune_{:.3f}_params.pth".format(alpha))

        model.load_state_dict(model_dict)
        test_loss, test_acc = evaluate(args, model, dataloader, criterion)
        test_robust_acc = eval_robustness_func(args, model)

        logger.info("==> Alpha: {:.2f}, test acc: {:.2f}%, test robust acc: {:.2f}%".format(alpha, test_acc, test_robust_acc))
        records.append((test_acc, test_robust_acc))

    return records


def load_sd(model_path):
    sd = torch.load(model_path)
    if "net" in sd.keys():
        sd = sd["net"]
    elif "state_dict" in sd.keys():
        sd = sd["state_dict"]
    elif "model" in sd.keys():
        sd = sd["model"]

    return sd


def evaluate_cifar_robustness(args, model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])

    model = nn.Sequential(norm_layer, model).to(args.device)
    atk_model = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)

    model.eval()

    total = 0
    correct = 0
    for images, labels in dataloader:

        images = images.to(args.device)
        labels = labels.to(args.device)

        adv_images = atk_model(images, labels)
        outputs = model(adv_images)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return correct / total * 100


def evaluate_tiny_robustness(args, model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = nn.Sequential(norm_layer, model).to(args.device)
    atk_model = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)

    model.eval()

    total = 0
    correct = 0
    for images, labels in dataloader:

        images = images.to(args.device)
        labels = labels.to(args.device)

        adv_images = atk_model(images, labels)
        outputs = model(adv_images)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return correct / total * 100


def evaluate_tiny_corruption(args, model, data_dir="./data/Tiny-ImageNet-C", level=1):
    model.eval()

    avg_acc = 0.0

    corruptions = [
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

    corruption_acc_dict = {}
    for cname in corruptions:
        correct = 0
        total = 0
        dataset = TinyImageNetC(cname, data_dir=data_dir, level=level)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        corruption_acc_dict[cname] = acc
        avg_acc += acc

    corruption_acc_dict["avg"] = avg_acc / len(corruptions)

    return corruption_acc_dict


def evaluate_cifar_corruption(args, model, data_dir="./data/CIFAR-100-C"):
    model.eval()

    avg_acc = 0.0

    corruptions = [
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

    corruption_acc_dict = {}
    for cname in corruptions:
        if args.dataset == "CIFAR10":
            dataset = CIFAR10C(cname, data_dir=data_dir)
        elif args.dataset == "CIFAR100":
            # dataset = CIFAR100C(cname, data_dir=data_dir)
            dataset = CIFAR100C(cname)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        corruption_acc_dict[cname] = acc
        avg_acc += acc

    corruption_acc_dict["avg"] = avg_acc / len(corruptions)

    return corruption_acc_dict


def evaluate(args, model, dataloader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return loss, acc


class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to("cuda"))
        self.register_buffer('std', torch.Tensor(std).to("cuda"))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_all_trained_model_params(path):

	trained_params_list = []

	for (root, dirs, files) in os.walk(path):
		# print(root, dirs, files)
		if len(files) > 0:
			for my_file in files:
				if my_file.find(".pth") != -1:
					trained_params_list.append(root+"/"+my_file)
	# print(trained_params_list)
	# exit()
	return trained_params_list