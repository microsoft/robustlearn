Code for the paper "Margin Calibration for Long-Tailed Visual Recognition".

**[Margin Calibration for Long-Tailed Visual Recognition](https://arxiv.org/abs/2112.07225)**  
Yidong Wang, Bowen Zhang, Wenxin Hou, Zhen Wu, Jindong Wang, Takahiro Shinozaki
ACML 2022

## Snapshot
```python

class MARCLinear(nn.Module):
    """
    A wrapper for nn.Linear with support of MARC method.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.a = torch.nn.Parameter(torch.ones(1, out_features))
        self.b = torch.nn.Parameter(torch.zeros(1, out_features))
    
    def forward(self, input, *args):
        with torch.no_grad():
            logit_before = self.fc(input)
            w_norm = torch.norm(self.fc.weight, dim=1)
        logit_after = self.a * logit_before + self.b * w_norm
        return logit_after


```

## Data preparation

Please follow [classifier-balancing](https://github.com/facebookresearch/classifier-balancing)

## Example of Training
- Base model (Representation Learning)
```bash
python main.py --cfg ./config/CIFAR10_LT/softmax_100.yaml
```
- MARC
```bash
python main.py --cfg ./config/CIFAR10_LT/MARC_100.yaml
```

## Experiment Results
The logs of training can be found at [logs link](https://1drv.ms/u/s!At10qerm7Tcdg25ROuGeKE644w81?e=J5fgfg).
We download Stage 1 model weights of Places and iNaturalist18 from [classifier-balancing](https://github.com/facebookresearch/classifier-balancing) for quick reproduction.
The results could be slightly different from the results reported in the paper, since we originally used an internal repository for the experiments in the paper.


## Cite BALMS
```bibtex
@article{wang2022margin},
  title={Margin calibration for long-tailed visual recognition},
  author={Wang, Yidong and Zhang, Bowen and Hou, Wenxin and Wu, Zhen and Wang, Jindong and Shinozaki, Takahiro},
  booktitle={Asian Conference on Machine Learning (ACML)},
  year={2022}
}
```

## Contact
Yidong Wang (yidongwang37@gmail.com)
Qiang Heng (qheng@ncsu.edu) Thanks for helping reproduction!
Jindong Wang (jindong.wang@micorosoft.com)

## Reference 
- The code is based on [balanced meta softmax](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification).

