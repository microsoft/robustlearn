import torch
import torch.nn as nn
from torch import Tensor as Tensor

import torch._C as _C

class BoundedTensor(Tensor): 
    @staticmethod
    # We need to override the __new__ method since Tensor is a C class
    def __new__(cls, x, ptb, *args, **kwargs):
        if isinstance(x, Tensor):
            tensor = super().__new__(cls, [], *args, **kwargs)
            tensor.data = x.data
            tensor.requires_grad = x.requires_grad
            return tensor
        else:
            return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x, ptb):
        self.ptb = ptb

    def __repr__(self):
        if hasattr(self, 'ptb') and self.ptb is not None:
            return '<BoundedTensor: {}, {}>'.format(super().__repr__(), self.ptb.__repr__())
        else:
            return '<BoundedTensor: {}, no ptb>'.format(super().__repr__())

    def clone(self, *args, **kwargs):
        tensor = BoundedTensor(super().clone(*args, **kwargs), self.ptb)
        return tensor

    def _func(self, func, *args, **kwargs):
        temp = func(*args, **kwargs)
        new_obj = BoundedTensor([], self.ptb)
        new_obj.data = temp.data
        new_obj.requires_grad = temp.requires_grad
        return new_obj

    # Copy to other devices with perturbation
    def to(self, *args, **kwargs):
        return self._func(super().to, *args, **kwargs)

    @classmethod
    def _convert(cls, ret):
        if cls is Tensor:
            return ret

        if isinstance(ret, Tensor):
            if True: 
                # The current implementation does not seem to need non-leaf BoundedTensor      
                return ret
            else:
                # Enable this branch if non-leaf BoundedTensor should be kept
                ret = ret.as_subclass(cls)

        if isinstance(ret, tuple):
            ret = tuple(cls._convert(r) for r in ret)

        return ret 

    if torch.__version__ >= '1.7':
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            if not all(issubclass(cls, t) for t in types):
                return NotImplemented

            with _C.DisableTorchFunction():
                ret = func(*args, **kwargs)
                return cls._convert(ret)


class BoundedParameter(nn.Parameter):
    def __new__(cls, data, ptb, requires_grad=True):
        return BoundedTensor._make_subclass(cls, data, requires_grad)
    
    def __init__(self, data, ptb, requires_grad=True):
        self.ptb = ptb
        self.requires_grad = requires_grad

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.ptb, self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return 'BoundedParameter containing:\n{}\n{}'.format(
            self.data.__repr__(), self.ptb.__repr__())

    def __reduce_ex__(self, proto):
        raise NotImplementedError

