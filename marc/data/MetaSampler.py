from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax
from queue import Queue
import numpy as np


def invert_sigmoid(x):
    return x.log() - (1-x).log()


class SampleLearner(nn.Module):
    """
    Sample Learner
    """

    def __init__(self, num_classes, init_pow=0., freq_path=None):
        super(SampleLearner, self).__init__()
        self.num_classes = num_classes
        self.init_pow = init_pow
        self.freq_path = freq_path

        self.fc = nn.Sequential(
            nn.Linear(num_classes, 1, bias=False),
            nn.Sigmoid()
        )

        # register intermediate variable between sampler and BP process
        self.sample_memory = Queue()

    def init_learner(self, img_num_per_cls):
        self.sample_per_class = torch.tensor(img_num_per_cls).float()
        self.sample_per_class = (self.sample_per_class / self.sample_per_class.sum()).cuda()
        self.fc.apply(self.init_weights_sampler)

    def init_weights_sampler(self, m):
        sample_per_class = 1. / self.sample_per_class.pow(self.init_pow)
        sample_per_class = sample_per_class / (sample_per_class.min() + sample_per_class.max())
        sample_per_class = invert_sigmoid(sample_per_class)
        if type(m) == nn.Linear:
            nn.init._no_grad_zero_(m.weight)
            with torch.no_grad():
                m.weight.add_(sample_per_class)

    def forward(self, onehot_targets, batch_size):
        """
        To be called in the sampler
        """
        weighted_onehot = self.fc(onehot_targets).squeeze(-1).unsqueeze(0)
        weighted_onehot = weighted_onehot.expand(batch_size, -1)
        weighted_onehot = gumbel_softmax(weighted_onehot.log(), hard=True, dim=-1)
        self.sample_memory.put(weighted_onehot.clone())
        return weighted_onehot.detach().nonzero()[:, 1]

    def forward_loss(self, x):
        """
        To be called when computing the meta loss
        """
        assert not self.sample_memory.empty()
        curr_sample = self.sample_memory.get()
        x = x.unsqueeze(-1) * curr_sample.detach() * curr_sample
        x = x.sum(-1).mean()
        return x


class MetaSampler(Sampler):
    def __init__(self, data_source, batch_size, meta_learner):
        num_classes = len(np.unique(data_source.labels))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
        self.num_samples = len(data_source.labels)

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.meta_learner = meta_learner
        self.indices = list(range(len(data_source.labels)))

        targets = []
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
            targets.append(label)

        targets = torch.tensor(targets)
        self.targets_onehot = nn.functional.one_hot(targets, num_classes).float().cuda()
        self.meta_learner.init_learner(data_source.img_num_per_cls)

    def __iter__(self):
        for _ in range(self.num_samples // self.batch_size):
            g = self.meta_learner(self.targets_onehot, self.batch_size)
            batch = [self.indices[i] for i in g]
            yield batch

    def __len__(self):
        return self.num_samples // self.batch_size


def get_sampler():
    return MetaSampler


def get_learner():
    return SampleLearner
