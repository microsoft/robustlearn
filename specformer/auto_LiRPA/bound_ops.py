import copy
import os
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d, \
    AdaptiveAvgPool2d, AvgPool2d, Tanh
import math

from auto_LiRPA.perturbations import Perturbation, PerturbationLpNorm, PerturbationSynonym, PerturbationL0Norm
from auto_LiRPA.utils import eyeC, LinearBound, user_data_dir, lockutils, isnan, Patches, logger

epsilon = 1e-12

"""Interval object. Used for interval bound propagation."""


class Interval(tuple):
    # Subclassing tuple object so that all previous code can be reused.
    def __new__(self, lb=None, ub=None, ptb=None):
        # FIXME: for the issue in model loading
        if ub is None:
            assert (isinstance(lb, tuple))
            lb, ub = lb

        return tuple.__new__(Interval, (lb, ub))

    def __init__(self, lb, ub, ptb=None):
        if ptb is None:
            # We do not perturb this interval. It shall be treated as a constant and lb = ub.
            self.ptb = None
            # To avoid mistakes, in this case the caller must make sure lb and ub are the same object.
            assert lb is ub
        else:
            if not isinstance(ptb, Perturbation):
                raise ValueError("ptb must be a Perturbation object or None. Got type {}".format(type(ptb)))
            else:
                self.ptb = ptb

    def __str__(self):
        return "({}, {}) with ptb={}".format(self[0], self[1], self.ptb)

    def __repr__(self):
        return "Interval(lb={}, ub={}, ptb={})".format(self[0], self[1], self.ptb)

    """Checking if the other interval is tuple, keep the perturbation."""

    @staticmethod
    def make_interval(lb, ub, other):
        if isinstance(other, Interval):
            return Interval(lb, ub, other.ptb)
        else:
            return (lb, ub)

    """Given a tuple or Interval object, returns the norm and eps."""

    @staticmethod
    def get_perturbation(interval):
        if isinstance(interval, Interval):
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            elif isinstance(interval.ptb, PerturbationSynonym):
                return np.inf, 1.0
            elif isinstance(interval.ptb, PerturbationL0Norm):
                return 0, interval.ptb.eps, interval.ptb.ratio
            elif interval.ptb is None:
                raise RuntimeError("get_perturbation() encountered an interval that is not perturbed.")
            else:
                raise RuntimeError("get_perturbation() does not know how to handle {}".format(type(interval.ptb)))
        else:
            # Tuple object. Assuming L infinity norm lower and upper bounds.
            return np.inf, np.nan

    """Checking if a Interval or tuple object has perturbation enabled."""

    @staticmethod
    def is_perturbed(interval):
        if isinstance(interval, Interval) and interval.ptb is None:
            return False
        else:
            return True


class Bound(nn.Module):
    def __init__(self, input_name, name, ori_name, attr={}, inputs=[], output_index=0, options={}, device=None):
        super().__init__()
        self.output_name = []
        self.input_name, self.name, self.ori_name, self.attr, self.inputs, self.output_index, self.options, self.device = \
            input_name, name, ori_name, attr, inputs, output_index, options, device
        self.fv = None
        self.from_input = False
        self.bounded = False
        self.IBP_rets = None
        # Determine if this node has a perturbed output or not. The function BoundedModule._mark_perturbed_nodes() will set this property.
        self.perturbed = False
        if options is not None and 'loss_fusion' in options:
            self.loss_fusion = options['loss_fusion']
        else:
            self.loss_fusion = False

    """Check if the i-th input is with perturbation or not."""

    def is_input_perturbed(self, i=0):
        return self.inputs[i].perturbed

    def forward(self, *x):
        raise NotImplementedError

    def interval_propagate(self, *v):
        assert (len(v) == 1)
        # unary monotonous functions only 
        h_L, h_U = v[0]
        return Interval.make_interval(self.forward(h_L), self.forward(h_U), v[0])

    def bound_forward(self, dim_in, last):
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def infer_batch_dim(self, batch_size, *x):
        print(self, self.name, self.fv.shape)
        raise NotImplementedError

    def broadcast_backward(self, A, x):
        shape = x.default_shape
        batch_dim = max(self.batch_dim, 0)
                
        if isinstance(A, torch.Tensor):
            if x.batch_dim == -1:
                # final shape of input
                shape = torch.Size([A.shape[batch_dim + 1]] + list(shape))
                dims = []
                cnt_sum = A.ndim - len(shape) - 1
                for i in range(1, A.ndim): # merge the output dimensions?
                    if i != self.batch_dim + 1 and cnt_sum > 0:
                        dims.append(i)
                        cnt_sum -= 1
                if dims:
                    A = torch.sum(A, dim=dims)
            else:
                dims = list(range(1, 1 + A.ndim - 1 - len(shape)))
                if dims:
                    A = torch.sum(A, dim=dims)
            dims = []
            for i in range(len(shape)):
                if shape[i] == 1 and A.shape[i + 1] != 1:
                    dims.append(i + 1)
            if dims:
                A = torch.sum(A, dim=dims, keepdim=True)
            assert (A.shape[1:] == shape)
        elif type(A) == Patches:
            pass
            
        return A

    @staticmethod
    def broadcast_forward(dim_in, x, shape_res):
        lw, lb, uw, ub = x.lw, x.lb, x.uw, x.ub
        shape_x, shape_res = list(x.lb.shape), list(shape_res)
        if lw is None:
            lw = uw = torch.zeros(dim_in, *shape_x, device=lb.device)
            has_batch_size = False
        else:
            has_batch_size = True
        while len(shape_x) < len(shape_res):
            if not has_batch_size:
                lw, uw = lw.unsqueeze(0), uw.unsqueeze(0)
                lb, ub = lb.unsqueeze(0), ub.unsqueeze(0)
                shape_x = [1] + shape_x
                has_batch_size = True
            else:
                lw, uw = lw.unsqueeze(2), uw.unsqueeze(2)
                lb, ub = lb.unsqueeze(1), ub.unsqueeze(1)
                shape_x = [shape_x[0], 1] + shape_x[1:]
        repeat = [(shape_res[i] // shape_x[i]) for i in range(len(shape_x))]
        lb, ub = lb.repeat(*repeat), ub.repeat(*repeat)
        repeat = repeat[:1] + [1] + repeat[1:]
        lw, uw = lw.repeat(*repeat), uw.repeat(*repeat)
        return lw, lb, uw, ub

    def get_bias(self, A, bias):
        if A is None:
            return 0
        assert not isnan(A)
        assert not isnan(bias)

        if isinstance(A, torch.Tensor):
            if torch.norm(A, p=1) < epsilon:
                return 0
            output_dim = A.shape[0]
            if self.batch_dim != -1:
                batch_size = A.shape[self.batch_dim + 1]
                A_shape = [
                    A.shape[0],
                    np.prod(A.shape[1:self.batch_dim + 1]).astype(np.int32),
                    batch_size,
                    np.prod(A.shape[self.batch_dim + 2:]).astype(np.int32)
                ]
                A = A.reshape(*A_shape).permute(2, 0, 1, 3).reshape(batch_size, output_dim, -1)
                bias = bias.reshape(*A_shape[1:]).transpose(0, 1).reshape(batch_size, -1, 1)
                # FIXME avoid .transpose(0, 1) back
                bias_new = A.matmul(bias).squeeze(-1).transpose(0, 1)
            else:
                batch_size = A.shape[1]
                A = A.view(output_dim, batch_size, -1)
                bias_new = A.matmul(bias.view(-1))
            if isnan(bias_new):
                # NaN can be caused by 0 * inf, if 0 appears in `A` and inf appears in `bias`.
                # Force the whole bias to be 0, to avoid gradient issues. 
                # FIXME maybe find a more robust solution.
                return 0
            else:
                return bias_new
        elif type(A) == Patches:
            if torch.norm(A.patches, p = 1) < epsilon:
                return 0

            # the shape of A.patches is [batch, L, out_c, in_c, K, K]
            
            if self.batch_dim != -1:
                batch_size = bias.shape[0]
                bias = F.unfold(bias, kernel_size=A.patches.size(-1), stride=A.stride, padding=A.padding).transpose(-2, -1).unsqueeze(-2)
                # Here the size of bias is [batch_size, L, 1, in_c * K * K]
                L = bias.size(1)

                # now the shape is [batch_size, L, out_c, in_c * K * K]
                patches = A.patches.view(A.patches.size(0), A.patches.size(1), A.patches.size(-4), A.patches.size(-1) * A.patches.size(-2) * A.patches.size(-3))
                prod = bias * patches
                bias_new = prod.sum(-1).transpose(-2, -1)
                bias_new = bias_new.view(batch_size, bias_new.size(-2), int(math.sqrt(bias_new.size(-1))), int(math.sqrt(bias_new.size(-1))))
            else:
                # Similar to BoundConstant
                patches = A.patches
                patches_reshape = torch.sum(patches, dim=(-1, -2, -3)) * bias.to(self.device)
                patches_reshape = patches_reshape.transpose(-1, -2)
                return patches_reshape.view(patches_reshape.size(0), patches_reshape.size(1), int(math.sqrt(patches_reshape.size(2))), -1).transpose(0, 1)

            return bias_new
        else:
            return NotImplementedError()


class BoundReshape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x, shape):
        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = np.prod(x.shape) // int(np.prod(shape[:i]) * np.prod(shape[(i + 1):]))
        self.input_shape, self.shape = x.size(), shape
        return x.reshape(shape)

    def bound_backward(self, last_lA, last_uA, x, shape):
        def _bound_oneside(A):
            if A is None:
                return None
            return A.reshape(A.shape[0], *self.input_shape)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, shape):
        batch_size = x.lw.shape[0]
        lw = x.lw.reshape(batch_size, dim_in, *self.shape[1:])
        uw = x.uw.reshape(batch_size, dim_in, *self.shape[1:])
        lb = x.lb.reshape(batch_size, *self.shape[1:])
        ub = x.ub.reshape(batch_size, *self.shape[1:])
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        return Interval.make_interval(v[0][0].reshape(*v[1][0]), v[0][1].reshape(*v[1][0]), v[0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif self.input_shape[x[0]] == self.shape[x[0]]:
            return x[0]
        raise NotImplementedError('input shape {}, new shape {}, input batch dim {}'.format(
            self.input_shape, self.shape, x[0]
        ))


class BoundInput(Bound):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name)
        self.value = value
        self.perturbation = perturbation
        self.from_input = True

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        # Update perturbed property based on the perturbation set.
        if key == "perturbation":
            if self.perturbation is not None:
                self.perturbed = True
            else:
                self.perturbed = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            if len(prefix.split('.')) == 2:
                key = prefix + name
            else:
                # change key to prefix + self.ori_name when calling load_state_dict()
                key = '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if param.ndim == 0 and input_param.ndim == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'
                                      .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if len(prefix.split('.')) == 2:
                    destination[self.ori_name] = param if keep_vars else param.detach()
                else:
                    # change parameters' name to self.ori_name when calling state_dict()
                    destination[
                        '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None:
                if len(prefix.split('.')) == 2:
                    destination[self.ori_name] = buf if keep_vars else buf.detach()
                else:
                    # change buffers' name to self.ori_name when calling state_dict()
                    destination[
                        '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name] = buf if keep_vars else buf.detach()

    def forward(self):
        return self.value

    def bound_forward(self, dim_in):
        assert 0

    def bound_backward(self, last_lA, last_uA):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

    def interval_propagate(self, *v):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

    def infer_batch_dim(self, batch_size, *x):
        shape = self.fv.shape
        for i in range(len(shape)):
            if shape[i] == batch_size:
                return i
        return -1


class BoundParams(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_parameter('param', value)
        self.from_input = False
        self.initializing = False

    """Override register_parameter() hook to register only needed parameters."""

    def register_parameter(self, name, param):
        if name == 'param':
            # self._parameters[name] = param  # cannot contain '.' in name, it will cause error when loading state_dict
            return super().register_parameter(name, param)
        else:
            # Just register it as a normal property of class.
            object.__setattr__(self, name, param)

    def init(self, initializing=False):
        self.initializing = initializing

    def forward(self):
        if self.initializing:
            return self.param_init
        else:
            return self.param

    def infer_batch_dim(self, batch_size, *x):
        return -1


class BoundBuffers(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_buffer('buffer', value.clone().detach())

    def forward(self):
        return self.buffer


class BoundLinear(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        # Gemm:
        # A = A if transA == 0 else A.T
        # B = B if transB == 0 else B.T
        # C = C if C is not None else np.array(0)
        # Y = alpha * np.dot(A, B) + beta * C
        # return Y

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        if attr is not None:
            # assumption: using it as a linear layer now
            assert 'transA' not in attr
            assert 'transB' not in attr or attr['transB'] == 1
            assert 'alpha' not in attr or attr['alpha'] == 1.0
            assert 'beta' not in attr or attr['beta'] == 1.0

        self.opt_matmul = options.get('matmul')

    def forward(self, x, w, b=None):
        self.input_shape = self.x_shape = x.shape
        self.y_shape = w.t().shape
        res = x.matmul(w.t())
        if b is not None:
            res += b
        return res

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2 or len(x) == 3
        has_bias = len(x) == 3
        # x[0]: input node, x[1]: weight, x[2]: bias
        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0
        batch_size = last_lA.shape[1] if isinstance(last_lA, torch.Tensor) else last_uA.shape[1]

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            # If last_lA and last_uA are indentity matrices.
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                # Not perturbed, so we can use either lower or upper.
                lA_x = uA_x = x[1].lower.unsqueeze(1).repeat([1, batch_size] + [1] * (x[1].lower.ndim - 1))
                # Bias will be directly added to output.
                if has_bias:
                    lbias = ubias = x[2].lower.unsqueeze(1).repeat(1, batch_size)
            else:
                def _bound_oneside(last_A):
                    if last_A is None:
                        return None, 0
                    # Just multiply this layer's weight into bound matrices, and produce biases.
                    next_A = last_A.matmul(x[1].lower)
                    sum_bias = last_A.matmul(x[2].lower) if has_bias else 0.0
                    return next_A, sum_bias

                lA_x, lbias = _bound_oneside(last_lA)
                uA_x, ubias = _bound_oneside(last_uA)

        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            # Obtain relaxations for matrix multiplication.
            [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias = self.bound_backward_with_weight(last_lA, last_uA, x[0], x[1])
            if has_bias:
                if x[2].perturbation is not None:
                    # Bias is also perturbed. Since bias is directly added to the output, in backward mode it is treated
                    # as an input with last_lA and last_uA as associated bounds matrices.
                    # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
                    lA_bias = last_lA
                    uA_bias = last_uA
                else:
                    # Bias not perturbed, so directly adding the bias of this layer to the final bound bias term.
                    if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                        # Bias will be directly added to output.
                        lbias += x[2].lower.unsqueeze(1).repeat(1, batch_size)
                        ubias += x[2].lower.unsqueeze(1).repeat(1, batch_size)
                    else:
                        if last_lA is not None:
                            lbias += last_lA.matmul(x[2].lower)
                        if last_uA is not None:
                            ubias += last_uA.matmul(x[2].lower)
            # If not has_bias, no need to compute lA_bias and uA_bias

        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                lA_x = uA_x = x[1].lower.unsqueeze(1).repeat([1, batch_size] + [1] * (x[1].lower.ndim - 1))
            else:
                lA_x = last_lA.matmul(x[1].lower)
                uA_x = last_uA.matmul(x[1].lower)
            # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
            lA_bias = last_lA
            uA_bias = last_uA

        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def _reshape(self, x_l, x_u, y_l, y_u):
        x_shape, y_shape = self.x_shape, self.y_shape

        # (x_1, x_2, ..., x_{n-1}, y_2, x_n)
        x_l = x_l.unsqueeze(-2)
        x_u = x_u.unsqueeze(-2)

        if len(x_shape) == len(y_shape):
            # (x_1, x_2, ..., x_{n-1}, y_n, y_{n-1})
            shape = x_shape[:-1] + (y_shape[-1], y_shape[-2])
            y_l = y_l.unsqueeze(-3)
            y_u = y_u.unsqueeze(-3)
        elif len(y_shape) == 2:
            # (x_1, x_2, ..., x_{n-1}, y_2, y_1)
            shape = x_shape[:-1] + y_shape[1:] + y_shape[:1]
            y_l = y_l.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
            y_u = y_u.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
        return x_l, x_u, y_l, y_u

    def _relax(self, x, y):
        return BoundMul.get_bound_mul(*self._reshape(x.lower, x.upper, y.lower, y.upper))

    def bound_backward_with_weight(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(x, y)

        alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
        beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)
        self.x_shape = x.lower.size()
        self.y_shape = y.lower.size()
        gamma_l = torch.sum(gamma_l, dim=-1).reshape(self.x_shape[0], -1, 1)
        gamma_u = torch.sum(gamma_u, dim=-1).reshape(self.x_shape[0], -1, 1)

        if len(x.default_shape) != 2 and len(x.default_shape) == len(y.default_shape):
            # assert (self.x_shape[0] == self.y_shape[0] == 1)
            dim_y = [-3]
        elif len(y.default_shape) == 2:
            dim_y = list(range(2, 2 + len(self.x_shape) - 2))
        else:
            raise NotImplementedError

        def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos, alpha_neg, beta_neg, gamma_neg):
            if last_A is None:
                return None, None, 0
            if isinstance(last_A, eyeC):
                A_x = alpha_pos.squeeze(0).permute(1, 0, 2).repeat(1, last_A.shape[1], 1)
                A_y = beta_pos * torch.eye(last_A.shape[2], device=last_A.device) \
                    .view((last_A.shape[2], 1, last_A.shape[2], 1))
                if len(dim_y) != 0:
                    A_y = torch.sum(beta_pos, dim=dim_y)
                bias = gamma_pos.transpose(0, 1)
            else:
                # last_uA has size (batch, spec, output)
                last_A_pos = last_A.clamp(min=0).unsqueeze(-1)
                last_A_neg = last_A.clamp(max=0).unsqueeze(-1)
                # alpha_u has size (batch, spec, output, input)
                # uA_x has size (batch, spec, input).
                # uA_x = torch.sum(last_uA_pos * alpha_u + last_uA_neg * alpha_l, dim=-2)
                A_x = (alpha_pos.transpose(-1, -2).matmul(last_A_pos) + \
                       alpha_neg.transpose(-1, -2).matmul(last_A_neg)).squeeze(-1)
                # beta_u has size (batch, spec, output, input)
                # uA_y is for weight matrix, with parameter size (output, input)
                # uA_y has size (batch, spec, output, input). This is an element-wise multiplication.
                A_y = last_A_pos * beta_pos + last_A_neg * beta_neg
                if len(dim_y) != 0:
                    A_y = torch.sum(A_y, dim=dim_y)
                # last_uA has size (batch, spec, output)
                _last_A_pos = last_A_pos.reshape(last_A.shape[0], last_A.shape[1], -1)
                _last_A_neg = last_A_neg.reshape(last_A.shape[0], last_A.shape[1], -1)
                # gamma_u has size (batch, output, 1)
                # ubias has size (batch, spec, 1)
                bias = _last_A_pos.transpose(0, 1).matmul(gamma_pos).transpose(0, 1) + \
                       _last_A_neg.transpose(0, 1).matmul(gamma_neg).transpose(0, 1)
            bias = bias.squeeze(-1)
            return A_x, A_y, bias

        lA_x, lA_y, lbias = _bound_oneside(last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
        uA_x, uA_y, ubias = _bound_oneside(last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def _propogate_Linf(h_L, h_U, w):
        mid = (h_L + h_U) / 2
        diff = (h_U - h_L) / 2
        w_abs = w.abs()
        if mid.ndim == 2 and w.ndim == 3:
            center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
            deviation = torch.bmm(diff.unsqueeze(1), w_abs.transpose(-1, -2)).squeeze(1)
        else:
            center = mid.matmul(w.transpose(-1, -2))
            deviation = diff.matmul(w_abs.transpose(-1, -2))
        return center, deviation

    def interval_propagate(self, *v, C=None, w=None):
        has_bias = len(v) == 3
        if w is None and self is None:
            # Use C as the weight, no bias.
            w, lb, ub = C, torch.tensor(0., device=C.device), torch.tensor(0., device=C.device)
        else:
            if w is None:
                # No specified weight, use this layer's weight.
                if self.is_input_perturbed(1):  # input index 1 is weight.
                    # w is a perturbed tensor. Use IBP with weight perturbation.
                    # C matrix merging not supported.
                    assert C is None
                    l, u = self.interval_propagate_with_weight(*v)
                    if has_bias:
                        return l + v[2][0], u + v[2][1]
                    else:
                        return l, u
                else:
                    # Use weight 
                    w = v[1][0]
            if has_bias:
                lb, ub = v[2]
            else:
                lb = ub = 0.0

            if C is not None:
                w = C.matmul(w)
                lb = C.matmul(lb) if not isinstance(lb, float) else lb
                ub = C.matmul(ub) if not isinstance(ub, float) else ub

        # interval_propagate() of the Linear layer may encounter input with different norms.
        norm = Interval.get_perturbation(v[0])
        norm = norm[0]
        if norm == np.inf:
            norm, eps = Interval.get_perturbation(v[0])
            h_L, h_U = v[0]

            if hasattr(self, 'detach_ibp_pre') and self.detach_ibp_pre:
                h_L, h_U = h_L.detach(), h_U.detach()

            center, deviation = BoundLinear._propogate_Linf(h_L, h_U, w)
        elif norm > 0:
            # General Lp norm.
            norm, eps = Interval.get_perturbation(v[0])
            mid = v[0][0]
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if w.ndim == 3:
                # Extra batch dimension.
                # mid has dimension [batch, input], w has dimension [batch, output, input].
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                # mid has dimension [batch, input], w has dimension [output, input].
                center = mid.matmul(w.t())
            deviation = w.norm(dual_norm, dim=-1) * eps
        else: # here we calculate the L0 norm IBP bound of Linear layers, using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = v[0][0]
            weight_abs = w.abs()
            if w.ndim == 3:
                # Extra batch dimension.
                # mid has dimension [batch, input], w has dimension [batch, output, input].
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                # mid has dimension [batch, input], w has dimension [output, input].
                center = mid.matmul(w.t())
            # L0 norm perturbation
            k = int(eps)
            deviation = torch.sum(torch.topk(weight_abs, k)[0], dim=1) * ratio

        lower, upper = center - deviation + lb, center + deviation + ub

        return lower, upper

    def interval_propagate_with_weight(self, *v):
        input_norm, input_eps = Interval.get_perturbation(v[0])
        weight_norm, weight_eps = Interval.get_perturbation(v[1])
        self.x_shape = v[0][0].shape
        self.y_shape = v[1][0].shape

        if input_norm == np.inf and weight_norm == np.inf:
            # A memory-efficient implementation without expanding all the elementary multiplications
            if self.opt_matmul == 'economic':
                x_l, x_u = v[0][0], v[0][1]
                y_l, y_u = v[1][0].transpose(-1, -2), v[1][1].transpose(-1, -2)

                dx, dy = F.relu(x_u - x_l), F.relu(y_u - y_l)
                base = x_l.matmul(y_l)

                mask_xp, mask_xn = (x_l > 0).float(), (x_u < 0).float()
                mask_xpn = 1 - mask_xp - mask_xn
                mask_yp, mask_yn = (y_l > 0).float(), (y_u < 0).float()
                mask_ypn = 1 - mask_yp - mask_yn

                lower, upper = base.clone(), base.clone()

                lower += dx.matmul(y_l.clamp(max=0)) - (dx * mask_xn).matmul(y_l * mask_ypn)
                upper += dx.matmul(y_l.clamp(min=0)) + (dx * mask_xp).matmul(y_l * mask_ypn)

                lower += x_l.clamp(max=0).matmul(dy) - (x_l * mask_xpn).matmul(dy * mask_yn)
                upper += x_l.clamp(min=0).matmul(dy) + (x_l * mask_xpn).matmul(dy * mask_yp)

                lower += (dx * mask_xn).matmul(dy * mask_yn)
                upper += (dx * (mask_xpn + mask_xp)).matmul(dy * (mask_ypn + mask_yp))
            else:
                # Both input data and weight are Linf perturbed (with upper and lower bounds).
                # We need a x_l, x_u for each row of weight matrix.
                x_l, x_u = v[0][0].unsqueeze(-2), v[0][1].unsqueeze(-2)
                y_l, y_u = v[1][0].unsqueeze(-3), v[1][1].unsqueeze(-3)
                # Reuse the multiplication bounds and sum over results.
                lower, upper = BoundMul.interval_propagate(*[(x_l, x_u), (y_l, y_u)])
                lower, upper = torch.sum(lower, -1), torch.sum(upper, -1)

            return lower, upper
        elif input_norm == np.inf and weight_norm == 2:
            # This eps is actually the epsilon per row, as only one row is involved for each output element.
            eps = weight_eps
            # Input data and weight are Linf perturbed (with upper and lower bounds).
            h_L, h_U = v[0]
            # First, handle non-perturbed weight with Linf perturbed data.
            center, deviation = BoundLinear._propogate_Linf(h_L, h_U, v[1][0])
            # Compute the maximal L2 norm of data. Size is [batch, 1].
            max_l2 = torch.max(h_L.abs(), h_U.abs()).norm(2, dim=-1).unsqueeze(-1)
            # Add the L2 eps to bounds.
            lb, ub = center - deviation - max_l2 * eps, center + deviation + max_l2 * eps
            return lb, ub
        else:
            raise NotImplementedError(
                "Unsupported perturbation combination: data={}, weight={}".format(input_norm, weight_norm))

    # w: an optional argument which can be utilized by BoundMatMul
    def bound_forward(self, dim_in, x, w=None, b=None, C=None):
        has_bias = b is not None

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            if isinstance(w, LinearBound):
                w = w.lower
            if isinstance(b, LinearBound):
                b = b.lower
            if C is not None:
                w = C.matmul(w).transpose(-1, -2)
                if b is not None:
                    b = C.matmul(b)
                w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
                lb = (x.lb.unsqueeze(1).matmul(w_pos) + x.ub.unsqueeze(1).matmul(w_neg)).squeeze(1)
                ub = (x.ub.unsqueeze(1).matmul(w_pos) + x.lb.unsqueeze(1).matmul(w_neg)).squeeze(1)
            else:               
                w = w.t()
                w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
                lb = x.lb.matmul(w_pos) + x.ub.matmul(w_neg)
                ub = x.ub.matmul(w_pos) + x.lb.matmul(w_neg)
            lw = x.lw.matmul(w_pos) + x.uw.matmul(w_neg)
            uw = x.uw.matmul(w_pos) + x.lw.matmul(w_neg)
            if b is not None:
                lb += b
                ub += b
        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            if C is not None:
                raise NotImplementedError
            res = self.bound_forward_with_weight(dim_in, x, w)
            if has_bias:
                raise NotImplementedError
            lw, lb, uw, ub = res.lw, res.lb, res.uw, res.ub
        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            raise NotImplementedError

        return LinearBound(lw, lb, uw, ub)

    def bound_forward_with_weight(self, dim_in, x, y):
        x_unsqueeze = LinearBound(
            x.lw.unsqueeze(-2),
            x.lb.unsqueeze(-2),
            x.uw.unsqueeze(-2),
            x.ub.unsqueeze(-2),
            x.lower.unsqueeze(-2),
            x.upper.unsqueeze(-2),
        )
        y_unsqueeze = LinearBound(
            y.lw.unsqueeze(-3),
            y.lb.unsqueeze(-3),
            y.uw.unsqueeze(-3),
            y.ub.unsqueeze(-3),
            y.lower.unsqueeze(-3),
            y.upper.unsqueeze(-3),
        )
        res_mul = BoundMul.bound_forward(dim_in, x_unsqueeze, y_unsqueeze)
        return LinearBound(
            res_mul.lw.sum(dim=-1) if res_mul.lw is not None else None,
            res_mul.lb.sum(dim=-1),
            res_mul.uw.sum(dim=-1) if res_mul.uw is not None else None,
            res_mul.ub.sum(dim=-1)
        )

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundBatchNormalization(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device, training):

        self.num_features = inputs[2].param.shape[0]
        self.eps = attr['epsilon']
        self.momentum = round(1 - attr['momentum'], 5)  # take care!
        self.affine = True
        self.track_running_stats = True
        self.mode = options.get("conv_mode", "matrix")
        # self.num_batches_tracked = 0 # not support yet

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        self.to(device)
        self.training = training

    def forward(self, x, w, b, m, v):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # n = x.numel() / x.size(1)
            self.current_mean = x.mean([0, 2, 3])
            self.current_var = x.var([0, 2, 3], unbiased=False)
            # with torch.no_grad():
            #     m.data.copy_(m.data * round(1. - exponential_average_factor, 5) + self.current_mean * exponential_average_factor)
            #     v.data.copy_(v.data * round(1. - exponential_average_factor, 5) + self.current_var * exponential_average_factor * n / (n - 1))
            # output = (x - self.current_mean[None, :, None, None]) / (torch.sqrt(self.current_var[None, :, None, None] + self.eps))
            # if self.affine:
            #     output = output * w[None, :, None, None] + b[None, :, None, None]
            output = F.batch_norm(x, m, v, w, b, self.training or not self.track_running_stats, exponential_average_factor,
                                  self.eps)

        else:
            self.current_mean = m.data.clone()
            self.current_var = v.data.clone()
            output = F.batch_norm(x, m, v, w, b, self.training or not self.track_running_stats,
                                  exponential_average_factor, self.eps)

        return output

    def bound_backward(self, last_lA, last_uA, *x):
        # TODO: Weight perturbation not support yet
        # x[0]: input, x[1]: weight, x[2]: bias, x[3]: running_mean, x[4]: running_var
        weight, bias = x[1].param, x[2].param
        current_mean, current_var = self.current_mean.to(weight.device), self.current_var.to(weight.device)

        tmp_bias = bias - current_mean / torch.sqrt(current_var + self.eps) * weight
        tmp_weight = weight / torch.sqrt(current_var + self.eps)

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) == torch.Tensor:
                next_A = last_A * tmp_weight.view(1, 1, -1, 1, 1)
                sum_bias = (last_A.sum((3, 4)) * tmp_bias).sum(2)
            elif type(last_A) == Patches:
                if last_A.identity == 0:
                    patches = last_A.patches
                    patches = patches * tmp_weight.view(-1, 1, 1)
                    next_A = Patches(patches, last_A.stride, last_A.padding, last_A.shape, identity=0)
                    sum_bias = (last_A.patches.sum((-1, -2)) * tmp_bias).sum(-1).transpose(-1, -2)
                    sum_bias = sum_bias.view(sum_bias.size(0), sum_bias.size(1), int(math.sqrt(sum_bias.size(2))), int(math.sqrt(sum_bias.size(2)))).transpose(0, 1)
                else:
                    # we should create a real identity Patch
                    num_channel = tmp_weight.view(-1).size(0)
                    patches = (torch.eye(num_channel, device=tmp_weight.device) * tmp_weight.view(-1)).unsqueeze(0).unsqueeze(0).unsqueeze(4).unsqueeze(5) # now [1 * 1 * in_C * in_C * 1 * 1]
                    next_A = Patches(patches, 1, 0, [1, 1, num_channel, 1, 1])
                    sum_bias = tmp_bias.unsqueeze(1).unsqueeze(2).unsqueeze(3) # squeezing batch dim, now [C * 1 * 1 * 1]
            else:
                raise NotImplementedError()
            return next_A, sum_bias

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)

        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], lbias, ubias

    def interval_propagate(self, *v):
        # TODO: Weight perturbation not support yet
        h_L, h_U = v[0]
        weight, bias = v[1][0], v[2][0]
        current_mean, current_var = self.current_mean.to(weight.device), self.current_var.to(weight.device)

        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0

        tmp_weight = weight / torch.sqrt(current_var + self.eps)
        tmp_weight_abs = tmp_weight.abs()
        tmp_bias = bias - current_mean * tmp_weight

        center = tmp_weight.view(1, -1, 1, 1) * mid + tmp_bias.view(1, -1, 1, 1)
        deviation = tmp_weight_abs.view(1, -1, 1, 1) * diff
        lower = center - deviation
        upper = center + deviation
        return lower, upper

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundConv(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']
        if len(inputs) == 3:
            self.has_bias = True
        else:
            self.has_bias = False
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.bounded = False
        self.IBP_rets = None
        self.to(device)
        self.mode = options.get("conv_mode", "matrix")

    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.has_bias
        bias = x[2] if self.has_bias else None
        output = F.conv2d(x[0], x[1], bias, self.stride, self.padding, self.dilation, self.groups)
        self.output_shape = output.size()[1:]
        self.input_shape = x[0].size()[1:]
        return output

    def bound_backward(self, last_lA, last_uA, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].fv

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) == torch.Tensor:
                shape = last_A.size()
            # when (W−F+2P)%S != 0, construct the output_padding
                output_padding0 = int(self.input_shape[1]) - (int(self.output_shape[1]) - 1) * self.stride[0] + 2 * \
                                self.padding[0] - int(weight.size()[2])
                output_padding1 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[1] + 2 * \
                                self.padding[1] - int(weight.size()[3])
                next_A = F.conv_transpose2d(last_A.reshape(shape[0] * shape[1], *shape[2:]), weight, None,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups, output_padding=(output_padding0, output_padding1))
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
                if self.has_bias:
                    sum_bias = (last_A.sum((3, 4)) * x[2].fv).sum(2)
                else:
                    sum_bias = 0
                return next_A, sum_bias
            elif type(last_A) == Patches:
                # Here we build and propagate a Patch object with (patches, stride, padding)
                # shape of the patches is [batch_size, L, out_c, in_c, K * K]

                # Now we only consider last_A that are Patches objects, which means that we only consider conv layers
                assert type(last_A) == Patches
                if last_A.identity == 0:
                    # [batch_size, L, out_c, in_c, K, K]
                    patches = last_A.patches
                    batch_size = last_A.patches.size(0)
                    L = patches.size(1)
                    out_c = patches.size(2)
                    patches = patches.reshape(-1, patches.size(-3), patches.size(-2), patches.size(-1))

                    pieces = F.conv_transpose2d(patches, weight, stride=self.stride)
                    pieces = pieces.view(batch_size, L, out_c, pieces.size(-3), pieces.size(-2), pieces.size(-1))

                    if self.has_bias:
                        patches = last_A.patches
                        patches_sum = patches.sum((-1, -2)) 

                        sum_bias = (patches_sum * x[2].fv).sum(-1).transpose(-2, -1)
                        sum_bias = sum_bias.view(batch_size, -1, int(math.sqrt(L)), int(math.sqrt(L))).transpose(0, 1)
                        # shape of the bias is [batch_size, L, out_channel]
                    else:
                        sum_bias = 0
                elif last_A.identity == 1:
                    pieces = weight.view(1, 1, weight.size(0), weight.size(1), weight.size(2), weight.size(3))
                    # Here we should transpose sum_bias to set the batch dim to 1, which is aimed to keep consistent with the matrix version
                    sum_bias = x[2].fv.unsqueeze(0).unsqueeze(2).unsqueeze(3).transpose(0, 1)
                else:
                    raise NotImplementedError()
                padding = last_A.padding if last_A is not None else 0
                stride = last_A.stride if last_A is not None else 1

                padding = padding * self.stride[0] + self.padding[0]
                stride *= self.stride[0]
                
                return Patches(pieces, stride, padding, pieces.shape), sum_bias
            else:
                raise NotImplementedError()

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def interval_propagate(self, *v, C=None):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        norm = Interval.get_perturbation(v[0])
        norm = norm[0]

        h_L, h_U = v[0]
        weight = v[1][0]
        bias = v[2][0] if self.has_bias else None

        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        elif norm > 0:
            norm, eps = Interval.get_perturbation(v[0])
            # L2 norm, h_U and h_L are the same.
            mid = h_U
            # TODO: padding
            deviation = torch.mul(weight, weight).sum((1, 2, 3)).sqrt() * eps
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else: # Here we calculate the L0 norm IBP bound using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = h_U
            k = int(eps)
            weight_sum = torch.sum(weight.abs(), 1)
            deviation = torch.sum(torch.topk(weight_sum.view(weight_sum.shape[0], -1), k)[0], dim=1) * ratio

            if self.has_bias:
                center = F.conv2d(mid, weight, v[2][0], self.stride, self.padding, self.dilation, self.groups)
            else:
                center = F.conv2d(mid, weight, None, self.stride, self.padding, self.dilation, self.groups)

            ss = center.shape
            deviation = deviation.repeat(ss[2] * ss[3]).view(-1, ss[1]).t().view(ss[1], ss[2], ss[3])
        
        center = F.conv2d(mid, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        upper = center + deviation
        lower = center - deviation
        return lower, upper

    def bound_forward(self, dim_in, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        weight = x[1].lb
        bias = x[2].lb if self.has_bias else None
        x = x[0]

        mid_w = (x.lw + x.uw) / 2
        mid_b = (x.lb + x.ub) / 2
        diff_w = (x.uw - x.lw) / 2
        diff_b = (x.ub - x.lb) / 2
        weight_abs = weight.abs()
        shape = mid_w.shape
        shape_wconv = [shape[0] * shape[1]] + list(shape[2:])
        deviation_w = F.conv2d(
            diff_w.reshape(shape_wconv), weight_abs, None, 
            self.stride, self.padding, self.dilation, self.groups)
        deviation_b = F.conv2d(
            diff_b, weight_abs, None, 
            self.stride, self.padding, self.dilation, self.groups)
        center_w = F.conv2d(
            mid_w.reshape(shape_wconv), weight, None, 
            self.stride, self.padding, self.dilation, self.groups)
        center_b =  F.conv2d(
            mid_b, weight, bias, 
            self.stride, self.padding, self.dilation, self.groups)
        deviation_w = deviation_w.reshape(shape[0], -1, *deviation_w.shape[1:])
        center_w = center_w.reshape(shape[0], -1, *center_w.shape[1:])

        return LinearBound(
            lw = center_w - deviation_w,
            lb = center_b - deviation_b,
            uw = center_w + deviation_w,
            ub = center_b + deviation_b)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return x[0]


class BoundMaxPool2d(MaxPool2d):
    def __init__(self, input_name, name, ori_name,
                 prev_layer,
                 kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        raise NotImplementedError


class BoundAveragePool(AvgPool2d):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        # assumptions: ceil_mode=False, count_include_pad=True
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])
        kernel_size = attr['kernel_shape']
        stride = attr['strides']
        padding = [attr['pads'][0], attr['pads'][1]]
        ceil_mode = False
        count_include_pad = True
        super().__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                         ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.fv = None
        self.bounded = False
        self.IBP_rets = None
        self.from_input = False

    def forward(self, x):
        self.input_shape = x.size()[1:]
        output = super().forward(x)
        return output

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            shape = last_A.size()
            # propagate A to the next layer, with batch concatenated together
            next_A = F.interpolate(last_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size) / (
                np.prod(self.kernel_size))
            next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            return next_A, 0

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        h_L = super().forward(h_L)
        h_U = super().forward(h_U)
        return h_L, h_U

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundGlobalAveragePool(AdaptiveAvgPool2d):
    def __init__(self, input_name, name, ori_name, prev_layer, output_size, output_index):
        raise NotImplementedError

        super().__init__(output_size=output_size)
        self.input_name = input_name
        self.output_name = []
        self.name = name
        self.ori_name = ori_name
        self.fv = None
        self.bounded = False
        self.IBP_rets = None

    @staticmethod
    def convert(act_layer, prev_layer, input_name, name):
        nl = BoundGlobalAveragePool(input_name, name, prev_layer, act_layer.output_size)
        return nl

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def interval_propagate(self, h_L, h_U):
        raise NotImplementedError


class BoundConcat(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.IBP_rets = None

    def forward(self, *x):  # x is a list of tensors
        x = [(item if isinstance(item, torch.Tensor) else torch.tensor(item)) for item in x]
        self.input_size = [item.shape[self.axis] for item in x]
        if self.axis < 0:
            self.axis = x[0].ndim + self.axis
        return torch.cat(x, dim=self.axis)

    def interval_propagate(self, *v):
        norms = []
        eps = []
        # Collect perturbation information for all inputs.
        for i, _v in enumerate(v):
            if self.is_input_perturbed(i):
                n, e = Interval.get_perturbation(_v)
                norms.append(n)
                eps.append(e)
            else:
                norms.append(None)
                eps.append(0.0)
        eps = np.array(eps)
        # Supporting two cases: all inputs are Linf norm, or all inputs are L2 norm perturbed.
        # Some inputs can be constants without perturbations.
        all_inf = all(map(lambda x: x is None or x == np.inf, norms))
        all_2 = all(map(lambda x: x is None or x == 2, norms))

        h_L = [_v[0] for _v in v]
        h_U = [_v[1] for _v in v]
        if all_inf:
            # Simply returns a tuple. Every subtensor has its own lower and upper bounds.
            return self.forward(*h_L), self.forward(*h_U)
        elif all_2:
            # Sum the L2 norm over all subtensors, and use that value as the new L2 norm.
            # This will be an over-approximation of the original perturbation (we can prove it).
            max_eps = np.sqrt(np.sum(eps * eps))
            # For L2 norm perturbed inputs, lb=ub and for constants lb=ub. Just propagate one object.
            r = self.forward(*h_L)
            ptb = PerturbationLpNorm(norm=2, eps=max_eps)
            return Interval(r, r, ptb)
        else:
            raise RuntimeError("BoundConcat does not support inputs with norm {}".format(norms))

    def bound_backward(self, last_lA, last_uA, *x):
        if self.axis < 0:
            self.axis = len(self.default_shape) + self.axis
        assert (self.axis > 0)

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return torch.split(last_A, self.input_size, dim=self.axis + 1)

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)
        if uA is None:
            return [(lA[i] if lA is not None else None, None) for i in range(len(lA))], 0, 0
        if lA is None:
            return [(None, uA[i] if uA is not None else None) for i in range(len(uA))], 0, 0
        return [(lA[i], uA[i]) for i in range(len(lA))], 0, 0

    def bound_forward(self, dim_in, *x):
        if self.axis < 0:
            self.axis = x[0].lb.ndim + self.axis
        assert (self.axis == 0 and not self.from_input or self.from_input)
        lw = torch.cat([item.lw for item in x], dim=self.axis + 1)
        lb = torch.cat([item.lb for item in x], dim=self.axis)
        uw = torch.cat([item.uw for item in x], dim=self.axis + 1)
        ub = torch.cat([item.ub for item in x], dim=self.axis)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert np.min(x) == np.max(x)
        assert x[0] != self.axis
        return x[0]


class BoundAdd(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.mode = options.get("conv_mode", "matrix")

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y

    def bound_backward(self, last_lA, last_uA, x, y):
        def _bound_oneside(last_A, w):
            if last_A is None:
                return None
            return self.broadcast_backward(last_A, w)

        uA_x = _bound_oneside(last_uA, x)
        uA_y = _bound_oneside(last_uA, y)
        lA_x = _bound_oneside(last_lA, x)
        lA_y = _bound_oneside(last_lA, y)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = Bound.broadcast_forward(dim_in, x, self.default_shape)
        y_lw, y_lb, y_uw, y_ub = Bound.broadcast_forward(dim_in, y, self.default_shape)
        lw, lb = x_lw + y_lw, x_lb + y_lb
        uw, ub = x_uw + y_uw, x_ub + y_ub
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        assert (not isinstance(y, torch.Tensor))
        return x[0] + y[0], x[1] + y[1]

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


class BoundSub(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x - y

    def bound_backward(self, last_lA, last_uA, x, y):
        def _bound_oneside(last_A, w, sign=-1):
            if last_A is None:
                return None
            return self.broadcast_backward(sign * last_A, w)

        uA_x = _bound_oneside(last_uA, x, sign=1)
        uA_y = _bound_oneside(last_uA, y, sign=-1)
        lA_x = _bound_oneside(last_lA, x, sign=1)
        lA_y = _bound_oneside(last_lA, y, sign=-1)
        return [(lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def bound_forward(self, dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = Bound.broadcast_forward(dim_in, x, self.default_shape)
        y_lw, y_lb, y_uw, y_ub = Bound.broadcast_forward(dim_in, y, self.default_shape)
        lw, lb = x_lw - y_uw, x_lb - y_ub
        uw, ub = x_uw - y_lw, x_ub - y_lb
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, x, y):
        return x[0] - y[1], x[1] - y[0]

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


class BoundPad(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        if len(attr) == 1:
            self.padding = [0, 0, 0, 0]
            self.value = 0.0
        else:
            self.padding = attr['pads'][2:4] + attr['pads'][6:8]
            self.value = attr['value']
        assert self.padding == [0, 0, 0, 0]
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, *x):
        return x[0]  # F.pad(x[0], self.padding, 'constant', self.value)

    def bound_backward(self, last_lA, last_uA, x, pad=None):
        # TODO
        if pad:
            return [(last_lA, last_uA), (None, None)], 0, 0
        else:
            return [(last_lA, last_uA)], 0, 0

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        return h_L, h_U

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == 0
        return 0


class BoundActivation(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True
        self.relaxed = False

    def _init_linear(self, x):
        self.mask_pos = torch.gt(x.lower, 0).to(torch.float)
        self.mask_neg = torch.lt(x.upper, 0).to(torch.float)
        self.mask_both = 1 - self.mask_pos - self.mask_neg

        self.lw = torch.zeros(x.lower.shape, device=self.device)
        self.lb = self.lw.clone()
        self.uw = self.lw.clone()
        self.ub = self.lw.clone()

    def _add_linear(self, mask, type, k, x0, y0):
        if mask is None:
            mask = 1
        if type == 'lower':
            w_out, b_out = self.lw, self.lb
        else:
            w_out, b_out = self.uw, self.ub

        w_out += mask * k
        b_out += mask * (-x0 * k + y0)

    def bound_relax(self, x):
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA, x):
        if not self.relaxed:
            self._init_linear(x)
            self.bound_relax(x)

        def _bound_oneside(last_A, sign=-1):
            if last_A is None:
                return None, 0

            if self.batch_dim == 0:
                if sign == -1:
                    _A = last_A.clamp(min=0) * self.lw.unsqueeze(0) + last_A.clamp(max=0) * self.uw.unsqueeze(0)
                    _bias = last_A.clamp(min=0) * self.lb.unsqueeze(0) + last_A.clamp(max=0) * self.ub.unsqueeze(0)
                elif sign == 1:
                    _A = last_A.clamp(min=0) * self.uw.unsqueeze(0) + last_A.clamp(max=0) * self.lw.unsqueeze(0)
                    _bias = last_A.clamp(min=0) * self.ub.unsqueeze(0) + last_A.clamp(max=0) * self.lb.unsqueeze(0)
                while _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=-1)
            elif self.batch_dim == -1:
                mask = torch.gt(last_A, 0.).to(torch.float)
                if sign == -1:
                    _A = last_A * (mask * self.lw.unsqueeze(0).unsqueeze(1) +
                                   (1 - mask) * self.uw.unsqueeze(0).unsqueeze(1))
                    _bias = last_A * (mask * self.lb.unsqueeze(0).unsqueeze(1) +
                                      (1 - mask) * self.ub.unsqueeze(0).unsqueeze(1))
                elif sign == 1:
                    _A = last_A * (mask * self.uw.unsqueeze(0).unsqueeze(1) +
                                   (1 - mask) * self.lw.unsqueeze(0).unsqueeze(1))
                    _bias = last_A * (mask * self.ub.unsqueeze(0).unsqueeze(1) +
                                      (1 - mask) * self.lb.unsqueeze(0).unsqueeze(1))
                while _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=-1)
            else:
                raise NotImplementedError

            return _A, _bias

        lA, lbias = _bound_oneside(last_lA, sign=-1)
        uA, ubias = _bound_oneside(last_uA, sign=+1)

        return [(lA, uA)], lbias, ubias

    def bound_forward(self, dim_in, x):
        if not self.relaxed:
            self._init_linear(x)
            self.bound_relax(x)

        if self.lw.ndim > 0:
            if x.lw is not None:
                lw = self.lw.unsqueeze(1).clamp(min=0) * x.lw + \
                     self.lw.unsqueeze(1).clamp(max=0) * x.uw
                uw = self.uw.unsqueeze(1).clamp(max=0) * x.lw + \
                     self.uw.unsqueeze(1).clamp(min=0) * x.uw
            else:
                lw = uw = None
        else:
            if x.lw is not None:
                lw = self.lw.unsqueeze(0).clamp(min=0) * x.lw + \
                     self.lw.unsqueeze(0).clamp(max=0) * x.uw
                uw = self.uw.unsqueeze(0).clamp(min=0) * x.lw + \
                     self.uw.unsqueeze(0).clamp(max=0) * x.uw
            else:
                lw = uw = None
        lb = self.lw.clamp(min=0) * x.lb + self.lw.clamp(max=0) * x.ub + self.lb
        ub = self.uw.clamp(max=0) * x.lb + self.uw.clamp(min=0) * x.ub + self.ub

        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundLeakyRelu(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True
        self.options = options.get('relu')
        self.alpha = attr['alpha']

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.alpha)

    def bound_backward(self, last_lA, last_uA, x=None):
        if x is not None:
            lb_r = x.lower.clamp(max=0)
            ub_r = x.upper.clamp(min=0)
        else:
            lb_r = self.lower.clamp(max=0)
            ub_r = self.upper.clamp(min=0)
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        upper_d = (ub_r - self.alpha * lb_r) / (ub_r - lb_r)
        upper_b = - lb_r * upper_d + self.alpha * lb_r

        if self.options == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.options == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float() + (upper_d < 1.0).float() * self.alpha
        elif self.options == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float() + (upper_d <= 0.0).float() * self.alpha
        else:
            lower_d = (upper_d > 0.5).float() + (upper_d <= 0.5).float() * self.alpha

        upper_d = upper_d.unsqueeze(0)
        lower_d = lower_d.unsqueeze(0)
        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None
        ubias = lbias = 0
        if last_uA is not None:
            neg_uA = last_uA.clamp(max=0)
            pos_uA = last_uA.clamp(min=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
            ubias = self.get_bias(pos_uA, upper_b)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            lbias = self.get_bias(neg_lA, upper_b)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        guard_eps = 1e-5
        self.upper = h_U
        self.lower = h_L
        return F.leaky_relu(h_L, self.alpha), F.leaky_relu(h_U, self.alpha)


class BoundRelu(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.options = options.get('relu')

        if self.options == "random_evaluation":
            self.slope = None

    def forward(self, x):
        if self.options == "random_evaluation":
            if self.slope is None or self.slope.shape != x.shape:
                self.slope = torch.ones_like(x, dtype=torch.float).to(x.device)
                self.slope.requires_grad_(True)
        return F.relu(x)

    # linear relaxation for nonlinear functions
    def bound_relax(self, x):
        m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)
        self._add_linear(mask=self.mask_neg, type='lower',
                         k=x.lower * 0, x0=0, y0=0)
        self._add_linear(mask=self.mask_neg, type='upper',
                         k=x.lower * 0, x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='lower',
                         k=torch.ones_like(x.lower), x0=0, y0=0)
        self._add_linear(mask=self.mask_pos, type='upper',
                         k=torch.ones_like(x.lower), x0=0, y0=0)

        upper_k = x.upper / (x.upper - x.lower).clamp(min=1e-12)
        self._add_linear(mask=self.mask_both, type='upper',
                         k=upper_k, x0=x.lower, y0=0)
        if self.options == "same-slope":
            lower_k = upper_k
        elif self.options == "zero-lb":
            lower_k = torch.zeros_like(upper_k)
        elif self.options == "one-lb":
            lower_k = torch.ones_like(upper_k)
        else:
            # adaptive
            lower_k = torch.gt(torch.abs(x.upper), torch.abs(x.lower)).to(torch.float)
        self._add_linear(mask=self.mask_both, type='lower',
                         k=lower_k, x0=0., y0=0.)

    def bound_backward(self, last_lA, last_uA, x=None):
        if x is not None:
            lb_r = x.lower.clamp(max=0)
            ub_r = x.upper.clamp(min=0)
        else:
            lb_r = self.lower.clamp(max=0)
            ub_r = self.upper.clamp(min=0)

        self.I = ((lb_r != 0) * (ub_r != 0)).detach()  # unstable neurons

        ub_r = torch.max(ub_r, lb_r + 1e-8)
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d

        use_lower_b = False

        if self.options == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.options == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.options == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        elif self.options == "reversed-adaptive":
            lower_d = (upper_d < 0.5).float()
        elif self.options == "random_evaluation":
            lower_d = self.slope.clone().clamp(min=0.0, max=1.0)
            if x is not None:
                lower = x.lower
                upper = x.upper
            else:
                lower = self.lower
                upper = self.upper
            lower_d[lower >= 0] = 1.0
            lower_d[upper < 0] = 0.0
        else:
            lower_d = (upper_d > 0.5).float()

        upper_d = upper_d.unsqueeze(0)
        lower_d = lower_d.unsqueeze(0)

        # Choose upper or lower bounds based on the sign of last_A
        uA = lA = None

        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0

            if type(last_A) == torch.Tensor:
                neg_A = last_A.clamp(max=0)
                pos_A = last_A.clamp(min=0)
                A = d_pos * pos_A + d_neg * neg_A
                bias = 0
                if b_pos is not None:
                    bias = bias + self.get_bias(pos_A, b_pos)
                if b_neg is not None:
                    bias = bias + self.get_bias(neg_A, b_neg)
                return A, bias
            elif type(last_A) == Patches:
                # if last_A is not an identity matrix
                assert last_A.identity == 0
                if last_A.identity == 0:
                    d_pos = d_pos.squeeze(0)
                    d_neg = d_neg.squeeze(0)

                    neg_A_patches = last_A.patches.clamp(max=0)
                    pos_A_patches = last_A.patches.clamp(min=0)

                    # unfold the slope matrix as patches
                    d_pos_unfolded = F.unfold(d_pos, kernel_size=pos_A_patches.size(-1), padding=last_A.padding, stride=last_A.stride)
                    d_neg_unfolded = F.unfold(d_neg, kernel_size=pos_A_patches.size(-1), padding=last_A.padding, stride=last_A.stride)

                    L = d_pos_unfolded.size(-1)

                    # reshape the unfolded patches as [batch_size, L, 1, in_c, H, W]
                    d_pos_unfolded = d_pos_unfolded.transpose(-2, -1)
                    d_pos_unfolded = d_pos_unfolded.view(d_pos_unfolded.size(0), d_pos_unfolded.size(1), pos_A_patches.size(-3), pos_A_patches.size(-2), pos_A_patches.size(-1)).unsqueeze(2)
                    d_neg_unfolded = d_neg_unfolded.transpose(-2, -1)
                    d_neg_unfolded = d_neg_unfolded.view(d_pos_unfolded.size(0), d_pos_unfolded.size(1), pos_A_patches.size(-3), pos_A_patches.size(-2), pos_A_patches.size(-1)).unsqueeze(2)

                    # multiply
                    prod = d_pos_unfolded * pos_A_patches + d_neg_unfolded * neg_A_patches

                    bias = 0
                    if b_pos is not None:
                        bias = bias + self.get_bias(Patches(pos_A_patches, last_A.stride, last_A.padding, last_A.shape), b_pos)
                    if b_neg is not None:
                        bias = bias + self.get_bias(Patches(neg_A_patches, last_A.stride, last_A.padding, last_A.shape), b_neg)

                    return Patches(prod, last_A.stride, last_A.padding, prod.shape, 0), bias.transpose(0, 1)

        uA, ubias = _bound_oneside(last_uA, upper_d, lower_d, upper_b, None)
        lA, lbias = _bound_oneside(last_lA, lower_d, upper_d, None, upper_b)

        self.d = upper_d.squeeze(0)
        return [(lA, uA)], lbias, ubias

    def interval_propagate(self, *v):
        h_L, h_U = v[0][0], v[0][1]
        guard_eps = 1e-5
        self.unstable = ((h_L < -guard_eps) & (h_U > guard_eps))
        self.upper = h_U
        self.lower = h_L

        if self.options == "random_evaluation":
            if self.slope is None or self.slope.shape != h_U.shape:
                self.slope = torch.zeros(h_U.shape, dtype=torch.float).cuda()
                self.slope.requires_grad_(True)

        return F.relu(h_L), F.relu(h_U)


class BoundTanh(Tanh, BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.precompute_relaxation('tanh', torch.tanh, self.dtanh)

    def dtanh(self, x):
        # to avoid bp error when cosh is too large
        x_limit = torch.tensor(20., device=x.device)
        mask = torch.lt(torch.abs(x), x_limit).float()
        cosh = torch.cosh(mask * x + 1 - mask)
        return mask * (1. / cosh.pow(2))

    """Precompute relaxation parameters for tanh and sigmoid"""

    @lockutils.synchronized('precompute_relaxation', external=True)
    def precompute_relaxation(self, name, func, dfunc):
        max_iter = 10
        self.step_pre = 0.01
        self.num_points_pre = 2000
        self.limit_pre = self.step_pre * self.num_points_pre
        path = os.path.join(user_data_dir, 'precomputed_{}'.format(name))
        if os.path.exists(path):
            self.d_lower, self.d_upper = torch.load(path, map_location=self.device)
            return

        with torch.no_grad():
            d_lower, d_upper = [], []

            lower = torch.tensor(-10., device=self.device)
            for _upper in range(0, self.num_points_pre + 5):
                upper = torch.tensor(_upper * self.step_pre, device=self.device)
                f_upper = func(upper)
                diff = lambda d: (f_upper - func(d)) / (upper - d + epsilon) - dfunc(d)
                d, _l, _u = lower / 2, lower, torch.zeros_like(lower)
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * mask_p + _l * (1 - mask_p)
                    _u = d * (1 - mask_p) + _u * mask_p
                    d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
                d_lower.append(d)

            upper = torch.tensor(10., device=self.device)
            for _lower in range(0, self.num_points_pre + 5):
                lower = torch.tensor(_lower * (-self.step_pre), device=self.device)
                f_lower = func(lower)
                diff = lambda d: (func(d) - f_lower) / (d - lower + epsilon) - dfunc(d)
                d, _l, _u = upper / 2, torch.zeros_like(upper), upper
                for t in range(max_iter):
                    v = diff(d)
                    mask_p = torch.gt(v, 0).to(torch.float)
                    _l = d * (1 - mask_p) + _l * mask_p
                    _u = d * mask_p + _u * (1 - mask_p)
                    d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
                d_upper.append(d)

            self.d_lower = torch.tensor(d_lower, device=self.device)
            self.d_upper = torch.tensor(d_upper, device=self.device)
            torch.save((self.d_lower, self.d_upper), path)

    def forward(self, x):
        return super().forward(x)

    def bound_relax_impl(self, x, func, dfunc):
        lower = x.lower.clamp(min=-self.limit_pre, max=self.limit_pre)
        upper = x.upper.clamp(min=-self.limit_pre, max=self.limit_pre)

        # lower bound for negative
        m = (lower + upper) / 2
        y_l, y_m, y_u = func(lower), func(m), func(upper)
        k = dfunc(m)
        self._add_linear(mask=self.mask_neg, type='lower', k=k, x0=m, y0=y_m)
        # upper bound for positive
        self._add_linear(mask=self.mask_pos, type='upper', k=k, x0=m, y0=y_m)

        # upper bound for negative
        k = (y_u - y_l) / (upper - lower + epsilon)
        self._add_linear(mask=self.mask_neg, type='upper', k=k, x0=lower, y0=y_l)
        # lower bound for positive
        self._add_linear(mask=self.mask_pos, type='lower', k=k, x0=lower, y0=y_l)

        d = torch.index_select(
            self.d_lower, 0,
            torch.max(
                torch.zeros_like(upper, dtype=torch.long).reshape(-1),
                (upper / self.step_pre).to(torch.long).reshape(-1)
            ) + 1
        ).reshape(upper.shape)
        y_d = func(d)
        k = (y_d - y_u) / (d - upper + epsilon)
        self._add_linear(mask=self.mask_both, type='lower', k=k, x0=d, y0=y_d)

        d = torch.index_select(
            self.d_upper, 0,
            torch.max(
                torch.zeros_like(lower, dtype=torch.long).reshape(-1),
                (lower / -self.step_pre).to(torch.long).reshape(-1)
            ) + 1
        ).reshape(upper.shape)
        k = (func(d) - y_l) / (d - lower + epsilon)
        self._add_linear(mask=self.mask_both, type='upper', k=k, x0=d, y0=func(d))

    def bound_relax(self, x):
        self.bound_relax_impl(x, torch.tanh, self.dtanh)


class BoundSigmoid(BoundTanh):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super(BoundTanh, self).__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.precompute_relaxation('sigmoid', torch.sigmoid, self.dsigmoid)

    def forward(self, x):
        return torch.sigmoid(x)

    def dsigmoid(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def bound_relax(self, x):
        self.bound_relax_impl(x, torch.sigmoid, self.dsigmoid)


class BoundExp(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.options = options.get('exp')
        self.max_input = 0

    def forward(self, x):
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(x, dim=-1, keepdim=True)[0].detach()
            return torch.exp(x - self.max_input)
        return torch.exp(x)

    def interval_propagate(self, *v):
        assert (len(v) == 1)
        # unary monotonous functions only
        h_L, h_U = v[0]
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(h_U, dim=-1, keepdim=True)[0]
            h_L, h_U = h_L - self.max_input, h_U - self.max_input
        else:
            self.max_input = 0
        return torch.exp(h_L), torch.exp(h_U)

    def bound_forward(self, dim_in, x):
        m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)

        exp_l, exp_m, exp_u = torch.exp(x.lower), torch.exp(m), torch.exp(x.upper)

        kl = exp_m
        lw = x.lw * kl.unsqueeze(1)
        lb = kl * (x.lb - m + 1)

        ku = (exp_u - exp_l) / (x.upper - x.lower + epsilon)
        uw = x.uw * ku.unsqueeze(1)
        ub = x.ub * ku - ku * x.lower + exp_l

        return LinearBound(lw, lb, uw, ub)

    def bound_backward(self, last_lA, last_uA, x):
        # Special case when computing log_softmax (FIXME: find a better solution, this trigger condition is not reliable).
        # FIXME disabled for fairseq
        if self.loss_fusion and last_lA is None and last_uA is not None and torch.min(
                last_uA) >= 0 and x.from_input:
            # Adding an extra bias term to the input. This is equivalent to adding a constant and subtract layer before exp.
            # Note that we also need to adjust the bias term at the end.
            if self.options == 'no-detach':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0]
            elif self.options != 'no-max-input':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0].detach()
            else:
                self.max_input = 0
            adjusted_lower = x.lower - self.max_input
            adjusted_upper = x.upper - self.max_input
            # relaxation for upper bound only (used in loss fusion)
            exp_l, exp_u = torch.exp(adjusted_lower), torch.exp(adjusted_upper)
            k = (exp_u - exp_l) / (adjusted_upper - adjusted_lower + epsilon)
            if k.requires_grad:
                k = k.clamp(min=1e-6)
            uA = last_uA * k.unsqueeze(0)
            ubias = last_uA * (-adjusted_lower * k + exp_l).unsqueeze(0)

            if ubias.ndim > 2:
                ubias = torch.sum(ubias, dim=tuple(range(2, ubias.ndim)))
            # Also adjust the missing ubias term.
            if uA.ndim > self.max_input.ndim:
                A = torch.sum(uA, dim=tuple(range(self.max_input.ndim, uA.ndim)))
            else:
                A = uA

            # These should hold true in loss fusion
            assert self.batch_dim == 0
            assert A.shape[0] == 1
            
            batch_size = A.shape[1]
            ubias -= (A.reshape(batch_size, -1) * self.max_input.reshape(batch_size, -1)).sum(dim=-1).unsqueeze(0)
            return [(None, uA)], 0, ubias
        else:
            return super().bound_backward(last_lA, last_uA, x)

    def bound_relax(self, x):
        m = torch.min((x.lower + x.upper) / 2, x.lower + 0.99)
        exp_l, exp_m, exp_u = torch.exp(x.lower), torch.exp(m), torch.exp(x.upper)
        k = exp_m
        self._add_linear(mask=None, type='lower', k=k, x0=m, y0=exp_m)
        min_val = -1e9  # to avoid (-inf)-(-inf) when both input.lower and input.upper are -inf
        exp_l, exp_u = exp_l.clamp(min=min_val), exp_u.clamp(min=min_val)
        k = (exp_u - exp_l) / (x.upper.clamp(min=min_val) - x.lower.clamp(min=min_val) + epsilon)
        k = k.clamp(min=1e-6)  # TODO double-check this (for gradient purpose)
        self._add_linear(mask=None, type='upper', k=k, x0=x.lower, y0=exp_l)


class BoundLog(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    def forward(self, x):
        # FIXME adhoc implementation for loss fusion
        if self.loss_fusion:
            return torch.logsumexp(self.inputs[0].inputs[0].inputs[0].fv, dim=-1) 
        return torch.log(x.clamp(min=epsilon))

    def bound_relax(self, x):
        rl, ru = self.forward(x.lower), self.forward(x.upper)
        ku = (ru - rl) / (x.upper - x.lower + epsilon)
        self._add_linear(mask=None, type='lower', k=ku, x0=x.lower, y0=rl)
        m = (x.lower + x.upper) / 2
        k = torch.reciprocal(m)
        rm = self.forward(m)
        self._add_linear(mask=None, type='upper', k=k, x0=m, y0=rm)

    def interval_propagate(self, *v):
        # FIXME adhoc implementation now
        if self.loss_fusion:
            lower = torch.logsumexp(self.inputs[0].inputs[0].inputs[0].lower, dim=-1) 
            upper = torch.logsumexp(self.inputs[0].inputs[0].inputs[0].upper, dim=-1) 
            return lower, upper
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x):
        A, lbias, ubias = super().bound_backward(last_lA, last_uA, x)
        # FIXME adhoc implementation now
        if self.loss_fusion:
            assert A[0][0] is None
            exp_module = self.inputs[0].inputs[0]
            ubias = ubias + self.get_bias(A[0][1], exp_module.max_input.squeeze(-1))
        return A, lbias, ubias


class BoundReciprocal(BoundActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    def forward(self, x):
        return torch.reciprocal(x)

    def bound_relax(self, x):
        m = (x.lower + x.upper) / 2
        kl = -1 / m.pow(2)
        self._add_linear(mask=None, type='lower', k=kl, x0=m, y0=1. / m)
        ku = -1. / (x.lower * x.upper)
        self._add_linear(mask=None, type='upper', k=ku, x0=x.lower, y0=1. / x.lower)

    def interval_propagate(self, *v):
        # support when h_L > 0
        h_L, h_U = v[0]
        return torch.reciprocal(h_U.float()), torch.reciprocal(h_L.float())


class BoundUnsqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]

    def forward(self, x):
        self.input_shape = x.shape
        if self.axes < 0:
            self.axes = len(self.input_shape) + self.axes + 1
        return x.unsqueeze(self.axes)

    def bound_backward(self, last_lA, last_uA, x):
        if self.axes == 0:
            return last_lA, 0, last_uA, 0
        else:
            return [(last_lA.squeeze(self.axes + 1) if last_lA is not None else None,
                     last_uA.squeeze(self.axes + 1) if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
        if len(self.input_shape) == 0:
            lw, lb = x.lw.unsqueeze(1), x.lb.unsqueeze(0)
            uw, ub = x.uw.unsqueeze(1), x.ub.unsqueeze(0)
        else:
            lw, lb = x.lw.unsqueeze(self.axes + 1), x.lb.unsqueeze(self.axes)
            uw, ub = x.uw.unsqueeze(self.axes + 1), x.ub.unsqueeze(self.axes)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif self.axes > x[0]:
            return x[0]
        raise NotImplementedError


class BoundSqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]

    def forward(self, x):
        return x.squeeze(self.axes)

    def bound_backward(self, last_lA, last_uA, x):
        assert (self.axes != 0)
        return [(last_lA.unsqueeze(self.axes + 1) if last_lA is not None else None,
                 last_uA.unsqueeze(self.axes + 1) if last_uA is not None else None)], 0, 0

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif x[0] < self.axes:
            return x[0]
        elif x[0] > self.axes:
            return x[0] - 1
        else:
            assert 0


class BoundConstantOfShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.device = device
        self.value = attr['value'].to(self.device)

    def forward(self, x):
        self.x = x
        self.from_input = True
        return self.value.expand(*list(x))

    def bound_backward(self, last_lA, last_uA, x):
        if last_lA is not None:
            lower_sum_b = last_lA * self.value
            while lower_sum_b.ndim > 2:
                lower_sum_b = torch.sum(lower_sum_b, dim=-1)
        else:
            lower_sum_b = 0

        if last_uA is not None:
            upper_sum_b = last_uA * self.value
            while upper_sum_b.ndim > 2:
                upper_sum_b = torch.sum(upper_sum_b, dim=-1)
        else:
            upper_sum_b = 0

        return [(None, None)], lower_sum_b, upper_sum_b

    def bound_forward(self, dim_in, x):
        assert (len(self.x) >= 1)
        lb = ub = torch.ones(self.default_shape, device=self.device) * self.value
        lw = uw = torch.zeros(self.x[0], dim_in, *self.x[1:], device=self.device)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        self.x = v[0][0]
        value = torch.ones(list(v[0][0]), device=self.device) * self.value
        return value, value

    def infer_batch_dim(self, batch_size, *x):
        # FIXME Should avoid referring to batch_size; Treat `torch.Size` results differently
        if self.x[0] == batch_size:
            return 0
        else:
            return -1


class BoundConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.value = attr['value'].to(self.device)

    def forward(self):
        return self.value.to(self.device)

    def infer_batch_dim(self, batch_size, *x):
        return -1

    def bound_backward(self, last_lA, last_uA):
        def _bound_oneside(A):
            if A is None:
                return 0.0

            if type(A) == torch.Tensor:
                while A.ndim > 2:
                    A = torch.sum(A, dim=-1)
            elif type(A) == Patches:
                patches = A.patches
                patches_reshape = torch.sum(patches, dim=(-1, -2, -3)) * self.value.to(self.device)
                patches_reshape = patches_reshape.transpose(-1, -2)
                return patches_reshape.view(patches_reshape.size(0), patches_reshape.size(1), int(math.sqrt(patches_reshape.size(2))), -1).transpose(0, 1)

            return A * self.value.to(self.device)

        lbias = _bound_oneside(last_lA)
        ubias = _bound_oneside(last_uA)
        return [], lbias, ubias

    def bound_forward(self, dim_in):
        lw = uw = torch.zeros(dim_in, device=self.device)
        lb = ub = self.value
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        return self.value.to(self.device), self.value.to(self.device)


class BoundShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @staticmethod
    def shape(x):
        return x.shape if isinstance(x, torch.Tensor) else torch.tensor(x).shape

    def forward(self, x):
        self.from_input = False
        return BoundShape.shape(x)

    def bound_backward(self, last_lA, last_uA, x):
        raise NotImplementedError

    def bound_forward(self, dim_in, x):
        return self.fv

    def infer_batch_dim(self, batch_size, *x):
        return -1


class BoundGather(Bound):
    def __init__(self, input_name, name, ori_name, attr, x, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, x, output_index, options, device)
        self.axis = attr['axis'] if 'axis' in attr else 0
        self.nonlinear = False  # input shape required

    def forward(self, x, indices):
        self.input_shape = x.shape
        self.indices = indices
        x = x.to(self.indices.device)  # Boundshape.shape() will return value on cpu only
        if indices.ndim == 0:
            # `index_select` requires `indices` to be a 1-D tensor
            return torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
        elif self.axis == 0:
            return torch.index_select(x, dim=self.axis, index=indices.reshape(-1)) \
                .reshape(*indices.shape, x.shape[-1])
        raise ValueError(
            'Unsupported shapes in Gather: data {}, indices {}, axis {}'.format(x.shape, indices.shape, self.axis))

    def bound_backward(self, last_lA, last_uA, x, indices):
        assert (self.from_input)

        assert self.indices.ndim == 0  # TODO

        def _bound_oneside(A):
            if A is None:
                return None
            assert (self.indices.ndim == 0)

            A = A.unsqueeze(self.axis + 1)
            idx = int(self.indices)
            tensors = []
            if idx > 0:
                shape_pre = list(A.shape)
                shape_pre[self.axis + 1] *= idx
                tensors.append(torch.zeros(shape_pre, device=self.device))
            tensors.append(A)
            if self.input_shape[self.axis] - idx - 1 > 0:
                shape_next = list(A.shape)
                shape_next[self.axis + 1] *= self.input_shape[self.axis] - idx - 1
                tensors.append(torch.zeros(shape_next, device=self.device))
            return torch.cat(tensors, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, indices):
        assert self.indices.ndim == 0  # TODO

        if isinstance(x, torch.Size):
            lw = uw = torch.zeros(dim_in, device=self.device)
            lb = ub = torch.index_select(
                torch.tensor(x, device=self.device),
                dim=self.axis, index=self.indices).squeeze(self.axis)
        else:
            axis = self.axis + 1
            lw = torch.index_select(x.lw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            uw = torch.index_select(x.uw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            lb = torch.index_select(x.lb, dim=self.axis, index=self.indices).squeeze(self.axis)
            ub = torch.index_select(x.ub, dim=self.axis, index=self.indices).squeeze(self.axis)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        return self.forward(v[0][0], v[1][0]), self.forward(v[0][1], v[1][0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] != -1:
            assert self.axis != x[0]
            return x[0]
        else:
            return x[1]

    def infer_batch_dim(self, batch_size, *x):
        if x[0] != -1:
            assert self.axis != x[0]
            return x[0]
        else:
            return x[1]


class BoundGatherElements(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, input, output_index, options, device)
        self.axis = attr['axis']

    def forward(self, x, index):
        self.index = index
        return torch.gather(x, dim=self.axis, index=index)

    def bound_backward(self, last_lA, last_uA, x, index):
        assert self.from_input
        
        dim = self._get_dim()

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(
                last_A.shape[0], last_A.shape[1], *x.default_shape[1:], device=last_A.device)
            A.scatter_(
                dim=dim + 1,
                index=self.index.unsqueeze(0).repeat(A.shape[0], *([1] * (A.ndim - 1))),
                src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def interval_propagate(self, *v):
        return self.forward(v[0][0], v[1][0]), \
               self.forward(v[0][1], v[1][1])

    def bound_forward(self, dim_in, x, index):
        assert self.axis != 0
        dim = self._get_dim()
        return LinearBound(
            torch.gather(x.lw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.lb, dim=dim, index=self.index),
            torch.gather(x.uw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.ub, dim=dim, index=self.index))

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]
    
    def _get_dim(self):
        dim = self.axis
        if dim < 0:
            dim = len(self.default_shape) + dim
        return dim


class BoundPrimConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, input, output_index, options, device)
        self.value = attr['value']

    def forward(self):
        return torch.tensor([], device=self.device)

    def bound_backward(self, last_lA, last_uA):
        raise NotImplementedError

    def interval_propagate(self, h_U, h_L, C=None):
        raise NotImplementedError


class BoundRNN(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.complex = True
        self.output_index = output_index

    def forward(self, x, weight_input, weight_recurrent, bias, sequence_length, initial_h):
        assert (torch.sum(torch.abs(initial_h)) == 0)

        self.input_size = x.shape[-1]
        self.hidden_size = weight_input.shape[-2]

        class BoundRNNImpl(nn.Module):
            def __init__(self, input_size, hidden_size,
                         weight_input, weight_recurrent, bias, output_index, options, device):
                super().__init__()

                self.input_size = input_size
                self.hidden_size = hidden_size

                self.cell = torch.nn.RNNCell(
                    input_size=input_size,
                    hidden_size=hidden_size
                )

                self.cell.weight_ih.data.copy_(weight_input.squeeze(0).data)
                self.cell.weight_hh.data.copy_(weight_recurrent.squeeze(0).data)
                self.cell.bias_ih.data.copy_((bias.squeeze(0))[:hidden_size].data)
                self.cell.bias_hh.data.copy_((bias.squeeze(0))[hidden_size:].data)

                self.output_index = output_index

            def forward(self, x):
                length = x.shape[0]
                outputs = []
                hidden = torch.zeros(x.shape[1], self.hidden_size, device=self.device)
                for i in range(length):
                    hidden = self.cell(x[i, :], hidden)
                    outputs.append(hidden.unsqueeze(0))
                outputs = torch.cat(outputs, dim=0)

                if self.output_index == 0:
                    return outputs
                else:
                    return hidden

        self.model = BoundRNNImpl(
            self.input_size, self.hidden_size,
            weight_input, weight_recurrent, bias,
            self.output_index, self.device)
        self.input = (x,)

        return self.model(self.input)


class BoundTranspose(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.perm = attr['perm']
        self.perm_inv_inc_one = [-1] * (len(self.perm) + 1)
        self.perm_inv_inc_one[0] = 0
        for i in range(len(self.perm)):
            self.perm_inv_inc_one[self.perm[i] + 1] = i + 1

    def forward(self, x):
        self.input_shape = x.shape
        return x.permute(*self.perm)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return last_A.permute(self.perm_inv_inc_one)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        if self.input_shape[0] != 1:
            perm = [0] + [(p + 1) for p in self.perm]
        else:
            assert (self.perm[0] == 0)
            perm = [0, 1] + [(p + 1) for p in self.perm[1:]]
        lw, lb = x.lw.permute(*perm), x.lb.permute(self.perm)
        uw, ub = x.uw.permute(*perm), x.ub.permute(self.perm)

        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        else:
            return self.perm.index(x[0])


class BoundMul(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    @staticmethod
    def get_bound_mul(x_l, x_u, y_l, y_u):
        alpha_l = y_l
        beta_l = x_l
        gamma_l = -alpha_l * beta_l

        alpha_u = y_u
        beta_u = x_l
        gamma_u = -alpha_u * beta_u

        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    # Special case when input is x * x.
    @staticmethod
    def get_bound_square(x_l, x_u):
        # Lower bound is a z=0 line if x_l and x_u have different signs.
        # Otherwise, the lower bound is a tangent line at x_l.
        # The lower bound should always be better than IBP.

        # If both x_l and x_u < 0, select x_u. If both > 0, select x_l.
        # If x_l < 0 and x_u > 0, we use the z=0 line as the lower bound.
        x_m = F.relu(x_l) - F.relu(-x_u)
        alpha_l = 2 * x_m
        gamma_l = - x_m * x_m

        # Upper bound: connect the two points (x_l, x_l^2) and (x_u, x_u^2).
        # The upper bound should always be better than IBP.
        alpha_u = x_l + x_u
        gamma_u = - x_l * x_u

        # Parameters before the second variable are all zeros, not used.
        beta_l = torch.zeros_like(x_l)
        beta_u = beta_l
        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    @staticmethod
    def _relax(x, y):
        if x is y:
            # A shortcut for x * x.
            return BoundMul.get_bound_square(x.lower, x.upper)

        x_l, x_u = x.lower, x.upper
        y_l, y_u = y.lower, y.upper

        # broadcast
        for k in [1, -1]:
            x_l = x_l + k * y_l
            x_u = x_u + k * y_u
        for k in [1, -1]:
            y_l = y_l + k * x_l
            y_u = y_u + k * x_u

        return BoundMul.get_bound_mul(x_l, x_u, y_l, y_u)

    def bound_backward(self, last_lA, last_uA, x, y):
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul._relax(x, y)

        alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
        beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)

        def _bound_oneside(last_A,
                           alpha_pos, beta_pos, gamma_pos,
                           alpha_neg, beta_neg, gamma_neg):
            if last_A is None:
                return None, None, 0
            last_A_pos, last_A_neg = last_A.clamp(min=0), last_A.clamp(max=0)
            A_x = last_A_pos * alpha_pos + last_A_neg * alpha_neg
            A_y = last_A_pos * beta_pos + last_A_neg * beta_neg
            last_A = last_A.reshape(last_A.shape[0], last_A.shape[1], -1)
            A_x = self.broadcast_backward(A_x, x)
            A_y = self.broadcast_backward(A_y, y)
            bias = self.get_bias(last_A_pos, gamma_pos) + \
                   self.get_bias(last_A_neg, gamma_neg)
            return A_x, A_y, bias

        lA_x, lA_y, lbias = _bound_oneside(
            last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
        uA_x, uA_y, ubias = _bound_oneside(
            last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def bound_forward(dim_in, x, y):
        x_lw, x_lb, x_uw, x_ub = x.lw, x.lb, x.uw, x.ub
        y_lw, y_lb, y_uw, y_ub = y.lw, y.lb, y.uw, y.ub

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul._relax(x, y)

        if x_lw is None: x_lw = 0
        if y_lw is None: y_lw = 0
        if x_uw is None: x_uw = 0
        if y_uw is None: y_uw = 0

        lw = alpha_l.unsqueeze(1).clamp(min=0) * x_lw + alpha_l.unsqueeze(1).clamp(max=0) * x_uw
        lw = lw + beta_l.unsqueeze(1).clamp(min=0) * y_lw + beta_l.unsqueeze(1).clamp(max=0) * y_uw
        lb = alpha_l.clamp(min=0) * x_lb + alpha_l.clamp(max=0) * x_ub + \
             beta_l.clamp(min=0) * y_lb + beta_l.clamp(max=0) * y_ub + gamma_l
        uw = alpha_u.unsqueeze(1).clamp(max=0) * x_lw + alpha_u.unsqueeze(1).clamp(min=0) * x_uw
        uw = uw + beta_u.unsqueeze(1).clamp(max=0) * y_lw + beta_u.unsqueeze(1).clamp(min=0) * y_uw
        ub = alpha_u.clamp(max=0) * x_lb + alpha_u.clamp(min=0) * x_ub + \
             beta_u.clamp(max=0) * y_lb + beta_u.clamp(min=0) * y_ub + gamma_u

        return LinearBound(lw, lb, uw, ub)

    @staticmethod
    def interval_propagate(*v):
        x, y = v[0], v[1]
        if x is y:
            # A shortcut for x * x.
            h_L, h_U = v[0]
            r0 = h_L * h_L
            r1 = h_U * h_U
            # When h_L < 0, h_U > 0, lower bound is 0.
            # When h_L < 0, h_U < 0, lower bound is h_U * h_U.
            # When h_L > 0, h_U > 0, lower bound is h_L * h_L.
            l = F.relu(h_L) - F.relu(-h_U)
            return l * l, torch.max(r0, r1)

        r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
        lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
        upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
        return lower, upper

    @staticmethod
    def infer_batch_dim(batch_size, *x):
        if x[0] == -1:
            return x[1]
        elif x[1] == -1:
            return x[0]
        else:
            assert x[0] == x[1]
            return x[0]


class BoundDiv(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    def forward(self, x, y):
        self.x, self.y = x, y
        return x / y

    def bound_backward(self, last_lA, last_uA, x, y):
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        A, lower_b, upper_b = mul.bound_backward(last_lA, last_uA, x, y_r)

        A_y, lower_b_y, upper_b_y = reciprocal.bound_backward(A[1][0], A[1][1], y)
        upper_b = upper_b + upper_b_y
        lower_b = lower_b + lower_b_y

        return [A[0], A_y[0]], lower_b, upper_b

    def bound_forward(self, dim_in, x, y):
        reciprocal, mul, y_r = self._convert_to_mul(x, y)
        y_r_linear = reciprocal.bound_forward(dim_in, y)
        y_r_linear = y_r_linear._replace(lower=y_r.lower, upper=y_r.upper)
        return mul.bound_forward(dim_in, x, y_r_linear)

    def interval_propagate(self, *v):
        x, y = v[0], v[1]
        y_r = BoundReciprocal.interval_propagate(None, y)
        return BoundMul.interval_propagate(x, y_r)

    def _convert_to_mul(self, x, y):
        try:
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', self.ori_name, {}, [], 0, None,
                                         self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', self.ori_name, {}, [], 0, None, self.device)
        except:
            # to make it compatible with previous code
            reciprocal = BoundReciprocal(self.input_name, self.name + '/reciprocal', None, {}, [], 0, None, self.device)
            mul = BoundMul(self.input_name, self.name + '/mul', None, {}, [], 0, None, self.device)
        reciprocal.default_shape = mul.default_shape = self.default_shape
        reciprocal.batch_dim = mul.batch_dim = self.batch_dim

        y_r = copy.copy(y)
        if isinstance(y_r, LinearBound):
            y_r = y_r._replace(lower=1. / y.upper, upper=1. / y.lower)
        else:
            y_r.lower = 1. / y.upper
            y_r.upper = 1. / y.lower
        return reciprocal, mul, y_r

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


class BoundNeg(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x):
        return -x

    def bound_backward(self, last_lA, last_uA, x):
        return [(-last_lA if last_lA is not None else None,
                 -last_uA if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(-x.lw, -x.lb, -x.uw, -x.ub)

    def interval_propagate(self, *v):
        return -v[0][1], -v[0][0]

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundMatMul(BoundLinear):
    # Reuse most functions from BoundLinear.
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x = x
        self.y = y
        return x.matmul(y)

    def interval_propagate(self, *v):
        w_l = v[1][0].transpose(-1, -2)
        w_u = v[1][1].transpose(-1, -2)
        lower, upper = super().interval_propagate(v[0], (w_l, w_u))
        return lower, upper   

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2
        # BoundLinear has W transposed.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        results = super().bound_backward(last_lA, last_uA, *x)
        # Transpose input back.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
        uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
        # Transpose result on A.
        return [results[0][0], (lA_y, uA_y), results[0][2]], results[1], results[2]

    def bound_forward(self, dim_in, x, y):
        return super().bound_forward(dim_in, x, LinearBound(
            y.lw.transpose(-1, -2) if y.lw is not None else None,
            y.lb.transpose(-1, -2) if y.lb is not None else None,
            y.uw.transpose(-1, -2) if y.uw is not None else None,
            y.ub.transpose(-1, -2) if y.ub is not None else None,
            y.lower.transpose(-1, -2) if y.lower is not None else None,
            y.upper.transpose(-1, -2) if y.upper is not None else None
        ))

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


class BoundCast(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.to = attr['to']
        self.data_types = [
            None,  # invalid
            torch.float,
            torch.uint8,
            torch.int8,
            None,  # uint16
            torch.int16,
            torch.int32,
            torch.int64,
            None,  # string
            torch.bool,
            torch.float16,
            torch.float32,
            None,  # uint32
            None,  # uint64
        ]
        self.type = self.data_types[self.to]
        assert self.type is not None

    def forward(self, x):
        self.type_in = x.dtype
        return x.to(self.type)

    def bound_backward(self, last_lA, last_uA, x):
        lA = last_lA.to(self.type_in) if last_lA is not None else None
        uA = last_uA.to(self.type_in) if last_uA is not None else None
        return [(lA, uA)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(
            x.lw.to(self.type), x.lb.to(self.type),
            x.uw.to(self.type), x.ub.to(self.type))

    def interval_propagate(self, *v):
        return v[0][0].to(self.type), v[0][1].to(self.type)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundSoftmaxImpl(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        max_x = torch.max(x, dim=self.axis).values
        assert self.axis == int(self.axis)
        # FIXME double-check whether this can break previous experiments
        # This does not seem to be a good implementation for computing bounds.
        x = torch.exp(x - max_x.unsqueeze(self.axis))
        s = torch.sum(x, dim=self.axis, keepdim=True)
        return x / s


# The `option != 'complex'` case is not used in the auto_LiRPA main paper.
class BoundSoftmax(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.option = options.get('softmax', 'complex')
        if self.option == 'complex':
            self.complex = True
        else:
            self.max_input = 30
            self.sigmoid = BoundSigmoid(self.name, None, None, {}, [], 0, {}, self.device)

    def forward(self, x):
        assert self.axis == int(self.axis)
        if self.option == 'complex':
            self.input = (x,)
            self.model = BoundSoftmaxImpl(self.axis)
            self.model.device = self.device
            return self.model(x)
        else:
            return F.softmax(x, dim=self.axis)

    def bound_backward(self, last_lA, last_uA, x):
        raise NotImplementedError

    def interval_propagate(self, *v):
        assert self.option != 'complex'
        h_L, h_U = v[0]
        shift = h_U.max(dim=self.axis, keepdim=True).values
        exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
        lower = exp_L / (torch.sum(exp_U, dim=self.axis, keepdim=True) - exp_U + exp_L + epsilon)
        upper = exp_U / (torch.sum(exp_L, dim=self.axis, keepdim=True) - exp_L + exp_U + epsilon)
        return lower, upper  

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]


class BoundReduceMax(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes']
        # for torch.max, `dim` must be an int
        if isinstance(self.axis, list):
            assert len(self.axis) == 1
            self.axis = self.axis[0]
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True

    def forward(self, x):
        self.input_shape = x.shape
        if self.axis < 0:
            self.axis += len(self.input_shape)
        assert self.axis > 0
        res = torch.max(x, dim=self.axis, keepdim=self.keepdim)
        self.indices = res.indices
        return res.values

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            indices = self.indices.unsqueeze(0)
            if not self.keepdim:
                assert (self.from_input)
                last_A = last_A.unsqueeze(self.axis + 1)
                indices = indices.unsqueeze(self.axis + 1)
            shape = list(last_A.shape)
            shape[self.axis + 1] *= self.input_shape[self.axis]
            A = torch.zeros(shape, device=last_A.device)
            A.scatter_(dim=self.axis + 1, index=indices, src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] != self.axis
        return x[0]


class BoundReduceMean(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes']
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True

    def forward(self, x):
        self.input_shape = x.shape
        return torch.mean(x, dim=self.axis, keepdim=self.keepdim)

    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]
                assert self.axis[i] > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                repeat = [1] * last_A.ndim
                size = self.input_shape[axis]
                if axis > 0:
                    repeat[axis + 1] *= size
                last_A = last_A.repeat(*repeat) / size
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.keepdim)
        assert (len(self.axis) == 1)
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        size = self.input_shape[axis]
        lw = x.lw.sum(dim=axis + 1, keepdim=True) / size
        lb = x.lb.sum(dim=axis, keepdim=True) / size
        uw = x.uw.sum(dim=axis + 1, keepdim=True) / size
        ub = x.ub.sum(dim=axis, keepdim=True) / size
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] in self.axis:
            assert not self.perturbed
            return -1
        return x[0]


class BoundReduceSum(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes'] if 'axes' in attr else None
        self.keepdim = bool(attr['keepdims'])

    def forward(self, x):
        self.input_shape = x.shape
        if self.axis is not None:
            return torch.sum(x, dim=self.axis, keepdim=self.keepdim)
        else:
            return torch.sum(x)

    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]
                assert self.axis[i] > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                repeat = [1] * last_A.ndim
                size = self.input_shape[axis]
                if axis > 0:
                    repeat[axis + 1] *= size
                last_A = last_A.repeat(*repeat)
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert self.keepdim
        assert len(self.axis) == 1
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        lw, lb = x.lw.sum(dim=axis + 1, keepdim=True), x.lb.sum(dim=axis, keepdim=True)
        uw, ub = x.uw.sum(dim=axis + 1, keepdim=True), x.ub.sum(dim=axis, keepdim=True)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert not x[0] in self.axis
        return x[0]


class BoundDropout(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.dropout = nn.Dropout(p=attr['ratio'])
        self.scale = 1 / (1 - attr['ratio'])

    def forward(self, x):
        res = self.dropout(x)
        self.mask = res == 0
        return res

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return torch.where(self.mask.unsqueeze(0), torch.tensor(0).to(last_A), last_A * self.scale)
        lA = _bound_oneside(last_lA)
        uA = _bound_oneside(last_uA)
        return [(lA, uA)], 0, 0

    def bound_forward(self, dim_in, x):
        assert (torch.min(self.mask) >= 0)
        lw = x.lw * self.mask.unsqueeze(1)
        lb = x.lb * self.mask
        uw = x.uw * self.mask.unsqueeze(1)
        ub = x.ub * self.mask
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        if not self.training:
            return h_L, h_U        
        else:
            lower = torch.where(self.mask, torch.tensor(0).to(h_L), h_L * self.scale)
            upper = torch.where(self.mask, torch.tensor(0).to(h_U), h_U * self.scale)
            return lower, upper

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundSplit(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.split = attr['split']

    def forward(self, x):
        self.input_shape = x.shape
        return torch.split(x, self.split, dim=self.axis)[self.output_index]

    def bound_backward(self, last_lA, last_uA, x):
        assert (self.axis > 0)
        pre = sum(self.split[:self.output_index])
        suc = sum(self.split[(self.output_index + 1):])

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = []
            if pre > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], pre, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            A.append(last_A)
            if suc > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], suc, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            return torch.cat(A, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.axis > 0 and self.from_input)
        lw = torch.split(x.lw, self.split, dim=self.axis + 1)[self.output_index]
        uw = torch.split(x.uw, self.split, dim=self.axis + 1)[self.output_index]
        lb = torch.split(x.lb, self.split, dim=self.axis)[self.output_index]
        ub = torch.split(x.ub, self.split, dim=self.axis)[self.output_index]
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] != self.axis
        return x[0]


class BoundEqual(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x, y):
        return x == y

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)


class BoundPow(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    def forward(self, x, y):
        return torch.pow(x, y)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        exp = v[1][0]
        assert exp == int(exp)
        exp = int(exp)
        pl, pu = torch.pow(v[0][0], exp), torch.pow(v[0][1], exp)
        if exp % 2 == 1:
            return pl, pu
        else:
            pl, pu = torch.min(pl, pu), torch.max(pl, pu)
            mask = 1 - ((v[0][0] < 0) * (v[0][1] > 0)).float()
            return pl * mask, pu

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundSqrt(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True

    def forward(self, x):
        return torch.sqrt(x)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundExpand(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x, y):
        y = y.clone()
        assert y.ndim == 1
        n, m = x.ndim, y.shape[0]
        assert n <= m
        for i in range(n):
            if y[m - n + i] == 1:
                y[m - n + i] = x.shape[i]
            else:
                assert x.shape[i] == 1 or x.shape[i] == y[m - n + i]
        return x.expand(*list(y))

    def infer_batch_dim(self, batch_size, *x):
        # FIXME should avoid referring to batch_size
        if self.fv.shape[0] != batch_size:
            return -1
        else:
            raise NotImplementedError('fv shape {}'.format(self.fv.shape))


class BoundWhere(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, condition, x, y):
        return torch.where(condition.to(torch.bool), x, y)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(0)
        condition = v[0][0]
        return tuple([torch.where(condition, v[1][j], v[2][j]) for j in range(2)])

    def bound_backward(self, last_lA, last_uA, condition, x, y):
        assert torch.allclose(condition.lower.float(), condition.upper.float())
        assert self.from_input
        mask = condition.lower.float()

        def _bound_oneside(last_A):
            if last_A is None:
                return None, None
            assert last_A.ndim > 1
            A_x = self.broadcast_backward(mask.unsqueeze(0) * last_A, x)
            A_y = self.broadcast_backward((1 - mask).unsqueeze(0) * last_A, y)
            return A_x, A_y

        lA_x, lA_y = _bound_oneside(last_lA)
        uA_x, uA_y = _bound_oneside(last_uA)

        return [(None, None), (lA_x, uA_x), (lA_y, uA_y)], 0, 0

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x[1:])


class BoundNot(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x):
        return x.logical_not()

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundCumSum(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x, axis):
        self.axis = axis
        return torch.cumsum(x, axis)

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]


class BoundSlice(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x, start, end, axes, steps=1):
        assert steps == 1 and axes == int(axes) and start == int(start) and end == int(end)
        start, end, axes, steps = int(start), int(end), int(axes), int(steps)
        shape = x.shape if isinstance(x, torch.Tensor) else [len(x)]
        if start < 0:
            start += shape[axes]
        if end < 0:
            end += shape[axes]
        end = min(end, shape[axes])
        return torch.narrow(x, dim=axes, start=start, length=(end - start))

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == -1
        return -1


class BoundSin(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x):
        return torch.sin(x)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundCos(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x):
        return torch.cos(x)

    def infer_batch_dim(self, batch_size, *x):
        return x[0]


class BoundRange(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, start, end, step):
        if start.dtype == end.dtype == step.dtype == torch.int64:
            return torch.arange(start, end, step, dtype=torch.int64, device=self.device)
        else:
            return torch.arange(start, end, step, device=self.device)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == x[1] == x[2] == -1
        return -1


class BoundScatterND(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    def forward(self, x, indices, updates):
        assert updates.ndim == x.ndim + indices.ndim - indices.shape[-1] - 1
        # It looks like there is no directly available implementation of ScatterND in torch.
        # Supporting a special case only here.
        assert x.ndim == 2 and indices.ndim == 3
        assert (updates == 0).all()
        assert (indices[:, :, 0] == 1).all()
        assert indices.shape[0] == 1 and indices.shape[1] == x.shape[1]
        assert (indices[:, :, 1] == torch.arange(0, x.shape[1], device=self.device)).all()
        res = x.clone()
        res[1, :] = 0
        return res

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] == x[1] == x[2] == -1
        return -1
