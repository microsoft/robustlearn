import torch
import torch.nn as nn

class CrossEntropyWrapper(nn.Module):
    def __init__(self, model):
        super(CrossEntropyWrapper, self).__init__()
        self.model = model

    def forward(self, x, labels):
        y = self.model(x)
        logits = y - torch.gather(y, dim=-1, index=labels.unsqueeze(-1))
        return torch.exp(logits).sum(dim=-1, keepdim=True)

class CrossEntropyWrapperMultiInput(nn.Module):
    def __init__(self, model):
        super(CrossEntropyWrapperMultiInput, self).__init__()
        self.model = model

    def forward(self, labels, *x):
        y = self.model(*x)
        logits = y - torch.gather(y, dim=-1, index=labels.unsqueeze(-1))
        return torch.exp(logits).sum(dim=-1, keepdim=True)