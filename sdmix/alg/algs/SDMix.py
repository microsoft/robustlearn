# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn as nn

from datautil.util import random_pairs_of_minibatches_by_domainperm
from alg.algs.base import Algorithm
from alg.modelopera import get_fea
from network import common_network
from alg.util import get_class_r, get_domain_r
from loss.margin_loss import LargeMarginLoss


class SDMix(Algorithm):
    def __init__(self, args):

        super(SDMix, self).__init__(args)
        self.class_r = None
        self.domain_r = None
        self.args = args
        self.featurizer = get_fea(args)
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)

        self.network = nn.Sequential(
            self.featurizer, self.bottleneck, self.classifier)

    def update_r(self, train_loaders, num_classes):
        self.class_r = get_class_r(self.featurizer, self.bottleneck, train_loaders,
                                   num_classes, self.args.normstyle, self.args.disttype)
        self.domain_r = get_domain_r(
            self.featurizer, self.bottleneck, train_loaders, self.args.normstyle, self.args.disttype)

    def update(self, minibatches, opt, sch):
        objective = 0

        criterion = LargeMarginLoss(self.args.mixup_ld_margin, top_k=self.args.top_k,
                                    loss_type=self.args.ldmarginlosstype, reduce='none')

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches_by_domainperm(minibatches):
            xi, yi, di, xj, yj, dj = xi.cuda().float(), yi.cuda().long(), di.cuda(
            ).long(), xj.cuda().float(), yj.cuda().long(), dj.cuda().long()
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
            r1 = torch.tensor([self.class_r[int(di[bi])][int(yi[bi])]
                               for bi in range(len(di))])
            r2 = torch.tensor([self.class_r[int(dj[bi])][int(yj[bi])]
                               for bi in range(len(dj))])
            x = lam * xi + (1 - lam) * xj

            fea = self.bottleneck(self.featurizer(x))
            predictions = self.classifier(fea)
            t1 = lam*r1
            t2 = (1-lam)*r2
            t = t1+t2
            t1 = (t1/t).detach().to(yi.device)
            t2 = (t2/t).detach().to(yj.device)

            objective += torch.mean(t1*criterion(predictions, yi, [fea]))
            objective += torch.mean(t2*criterion(predictions, yj, [fea]))

        objective /= len(minibatches)
        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': objective.item()}

    def predict(self, x):
        return self.network(x)
