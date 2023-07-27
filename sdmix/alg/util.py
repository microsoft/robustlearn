# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch


class avgmeta():
    def __init__(self):
        self.count = 0
        self.avg = None

    def update(self, tcount, tx):
        if self.count == 0:
            self.count = tcount
            self.avg = torch.mean(tx, dim=0)
        else:
            self.avg = self.avg*self.count+torch.sum(tx, dim=0)
            self.count += tcount
            self.avg = self.avg/self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


def get_dist(x, y, dist_type='1-norm'):
    if dist_type == '1-norm':
        return torch.sum(torch.abs(x-y))
    elif dist_type == '2-norm':
        return torch.sqrt(torch.sum(torch.square(x-y)))
    elif dist_type == 'cos':
        t1, t2 = torch.norm(x), torch.norm(y)
        if t1 == 0:
            if t2 == 0:
                return 0
            else:
                tx = torch.ones_like(x.shape).to(x.device)
            return 1-torch.sum(tx*y)/(torch.norm(tx)*t2)
        elif t2 == 0:
            ty = torch.ones_like(x.shape).to(x.device)
            return 1-torch.sum(x*ty)/(torch.norm(ty)*t1)
        else:
            return 1-torch.sum(x*y)/(t1*t2)
    else:
        return 0


def get_maxdist(x, y, dist_type='1-norm'):
    maxd = 0
    for i in range(x.shape[0]):
        td = get_dist(x[i], y, dist_type)
        if td > maxd:
            maxd = td.detach()
    return maxd


def get_avgdist(x, y, dist_type='1-norm'):
    sum = 0
    for i in range(x.shape[0]):
        td = get_dist(x[i], y, dist_type)
        sum += td
    if x.shape[0] == 0:
        return 0
    return (sum/x.shape[0])


def get_domain_r(feanet, botnet, train_loaders, rstyle='max', disttype='1-norm'):
    l = len(train_loaders)
    avglist = [avgmeta() for i in range(l)]
    feanet.eval()
    botnet.eval()
    with torch.no_grad():
        for i in range(l):
            for j, data in enumerate(train_loaders[i]):
                x, y = data[0], data[1]
                x = x.cuda()
                tcount = x.size(0)
                tx = botnet(feanet(x))
                avglist[i].update(tcount, tx)
    rlist = [0 for i in range(l)]
    with torch.no_grad():
        for i in range(l):
            for j, data in enumerate(train_loaders[i]):
                x, y = data[0], data[1]
                x = x.cuda()
                tx = botnet(feanet(x))
                if rstyle == 'max':
                    td = get_maxdist(tx, avglist[i].get_avg(), disttype)
                else:
                    td = get_avgdist(tx, avglist[i].get_avg(), disttype)
                if td > rlist[i]:
                    rlist[i] = td
    feanet.train()
    botnet.train()
    return rlist


def get_class_r(feanet, botnet, train_loaders, num_classes, rstyle='max', disttype='1-norm'):
    l = len(train_loaders)
    avglist = [[avgmeta() for j in range(num_classes)] for i in range(l)]
    rlist = [[0 for j in range(num_classes)] for i in range(l)]
    feanet.eval()
    botnet.eval()
    with torch.no_grad():
        for i in range(l):
            for j, data in enumerate(train_loaders[i]):
                x, y = data[0], data[1]
                x = x.cuda()
                tx = botnet(feanet(x))
                for k in range(num_classes):
                    index = torch.where(y == k)[0]
                    tcount = len(index)
                    if tcount > 0:
                        avglist[i][k].update(tcount, tx[index])
        for i in range(l):
            for j, data in enumerate(train_loaders[i]):
                x, y = data[0], data[1]
                x = x.cuda()
                tx = botnet(feanet(x))
                for k in range(num_classes):
                    index = torch.where(y == k)
                    if rstyle == 'max':
                        td = get_maxdist(
                            tx[index], avglist[i][k].get_avg(), disttype)
                    else:
                        td = get_avgdist(
                            tx[index], avglist[i][k].get_avg(), disttype)
                    if td > rlist[i][k]:
                        rlist[i][k] = td
    feanet.train()
    botnet.train()
    return rlist
