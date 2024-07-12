from train.utils import *
import torch.nn.functional as F
import numpy as np
import random
import torch.backends.cudnn as cudnn


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = CW_loss(model(X+delta), y, reduction=False).detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def evaluate_pgd(args, model, test_loader, eval_steps=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    print('Evaluating with PGD {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset=="cifar10":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="cifar100":
        mu = torch.tensor(cifar100_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar100_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet" :
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)), 
                pgd_loss/n, pgd_acc/n)
    return pgd_loss/n, pgd_acc/n

def evaluate_CW(args, model, test_loader, eval_steps=None):
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    cw_loss = 0
    cw_acc = 0
    n = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    model.eval()
    print('Evaluating with CW {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset=="cifar10":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="cifar100":
        mu = torch.tensor(cifar100_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar100_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet":
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = CW_loss(output, y)
            cw_loss += loss.item() * y.size(0)
            cw_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)),
                cw_loss/n, cw_acc/n)
    return cw_loss/n, cw_acc/n


def CW_loss(x, y, reduction=True, num_cls=10, threshold=10,):
    batch_size = x.shape[0]
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    logit_mc = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1. - ind)
    logit_gt = x[np.arange(batch_size), y]
    loss_value_ori = -(logit_gt - logit_mc)
    loss_value = torch.maximum(loss_value_ori, torch.tensor(-threshold).cuda())
    if reduction:
        return loss_value.mean()
    else:
        return loss_value



