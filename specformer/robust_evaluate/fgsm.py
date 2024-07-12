import sys
from tqdm import tqdm
import torch.nn as nn
from autoattack import AutoAttack
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from sys import exit
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import pandas as pd
import csv
import random
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, TypeVar
from torch import Tensor
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def validate_fgsm(args,model,logger,test_loader, device,eps):
    # Attack amount
    color_value = 255.0
    eps /= color_value
    logger.info(f"FGSM attack with capacity {eps}")
    
        
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for i, (input, target) in enumerate(test_loader):
        
        input = input.to(device)
        target = target.to(device)
        
        orig_input = input.clone()

        invar = torch.autograd.Variable(input, requires_grad=True)
        output = model(invar)
        ascend_loss = criterion(output, target)
        ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
        pert = fgsm(ascend_grad, eps)
        # Apply purturbation
        input += pert.data
        input = torch.max(orig_input-eps, input)
        input = torch.min(orig_input+eps, input)
        input.clamp_(0, 1.0)
        
        # input.sub_(dmean[None,:,None,None]).div_(dstd[None,:,None,None])
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if i == 0 or (i + 1) % 10 == 0:
                logger.info('FGSM Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(test_loader), loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    logger.info(' FGSM Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg
