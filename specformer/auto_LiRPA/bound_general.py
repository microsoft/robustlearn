import time
from collections import OrderedDict, deque

import torch.optim as optim
from torch.nn import DataParallel, Parameter

from auto_LiRPA.bound_op_map import bound_op_map
from auto_LiRPA.bound_ops import *
from auto_LiRPA.bounded_tensor import BoundedTensor, BoundedParameter
from auto_LiRPA.parse_graph import parse_module
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import LinearBound, logger, eyeC, unpack_inputs, Patches, BoundList


class BoundedModule(nn.Module):
    def __init__(self, model, global_input, bound_opts={}, auto_batch_dim=True, device='auto',
            verbose=False):
        super(BoundedModule, self).__init__()
        if isinstance(model, BoundedModule):
            for key in model.__dict__.keys():
                setattr(self, key, getattr(model, key))
            return
        self.verbose = verbose
        self.bound_opts = bound_opts
        self.auto_batch_dim = auto_batch_dim
        self.device = device if device != 'auto' else next(model.parameters()).device
        self.global_input = global_input
        if auto_batch_dim:
            # logger.warning('Using automatic batch dimension inferring, which may not be correct')
            self.init_batch_size = -1

        state_dict_copy = copy.deepcopy(model.state_dict())
        object.__setattr__(self, 'ori_state_dict', state_dict_copy)
        self._convert(model, global_input)
        self._mark_perturbed_nodes()

    def __call__(self, *input, **kwargs):

        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            opt = "forward" 
        for kwarg in [
                'disable_multi_gpu', 'no_replicas', 'get_property', 
                'node_class', 'att_name']:
            if kwarg in kwargs:
                kwargs.pop(kwarg)
        if opt == "compute_bounds":
            return self.compute_bounds(**kwargs)
        else:
            return self.forward(*input, **kwargs)

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def load_state_dict(self, state_dict, strict=False):
        new_dict = OrderedDict()
        # translate name to ori_name
        for k, v in state_dict.items():
            if k in self.node_name_map:
                new_dict[self.node_name_map[k]] = v
        return super(BoundedModule, self).load_state_dict(new_dict, strict=strict)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                # translate name to ori_name
                if name in self.node_name_map:
                    name = self.node_name_map[name]
                yield name, v

    def train(self, mode=True):
        super().train(mode)
        for node in self._modules.values():
            node.train(mode=mode)

    def eval(self):
        super().eval()
        for node in self._modules.values():
            node.eval()

    def forward(self, *x, final_node_name=None):
        self._set_input(*x)

        degree_in = {}
        queue = deque()
        for key in self._modules.keys():
            l = self._modules[key]
            degree_in[l.name] = len(l.input_name)
            if degree_in[l.name] == 0:
                queue.append(l)
        forward_values = {}

        final_output = None
        while len(queue) > 0:
            l = queue.popleft()

            inp = [forward_values[l_pre] for l_pre in l.input_name]
            for l_pre in l.input_name:
                l.from_input = l.from_input or self._modules[l_pre].from_input
            fv = l.forward(*inp)
            if isinstance(fv, torch.Size) or isinstance(fv, tuple):
                fv = torch.tensor(fv, device=self.device)
            object.__setattr__(l, 'forward_value', fv)
            object.__setattr__(l, 'fv', fv)
            # infer batch dimension
            if not hasattr(l, 'batch_dim'):
                inp_batch_dim = [self._modules[l_pre].batch_dim for l_pre in l.input_name]
                try:
                    l.batch_dim = l.infer_batch_dim(self.init_batch_size, *inp_batch_dim)
                    try: logger.debug('Batch dimension of ({})[{}]: fv shape {}, infered {}, input batch dimensions {}'.format(
                        l, l.name, l.forward_value.shape, l.batch_dim, inp_batch_dim
                    ))
                    except: pass
                except:
                    raise Exception('Fail to infer the batch dimension of ({})[{}]: fv shape {}, input batch dimensions {}'.format(
                        l, l.name, l.forward_value.shape, inp_batch_dim
                    ))

            if isinstance(l.forward_value, torch.Tensor):
                l.default_shape = l.forward_value.shape
            forward_values[l.name] = l.forward_value
            logger.debug('Forward at {}[{}], fv shape {}'.format(l, l.name, fv.shape))

            for l_next in l.output_name:
                degree_in[l_next] -= 1
                if degree_in[l_next] == 0:  # all inputs of this node have already set
                    queue.append(self._modules[l_next])

        if final_node_name:
            return forward_values[final_node_name]
        else:
            out = deque([forward_values[n] for n in self.output_name])
            def _fill_template(template):
                if template is None:
                    return out.popleft()
                elif isinstance(template, list) or isinstance(template, tuple):
                    res = []
                    for t in template:
                        res.append(_fill_template(t))
                    return tuple(res) if isinstance(template, tuple) else res
                elif isinstance(template, dict):
                    res = {}
                    for key in template:
                        res[key] = _fill_template(template[key])
                    return res
                else:
                    raise NotImplementedError
            return _fill_template(self.output_template)

    """Mark the graph nodes and determine which nodes need perturbation."""
    def _mark_perturbed_nodes(self):
        degree_in = {}
        queue = deque()
        # Initially the queue contains all "root" nodes.
        for key in self._modules.keys():
            l = self._modules[key]
            degree_in[l.name] = len(l.input_name)
            if degree_in[l.name] == 0:
                queue.append(l) #in_degree ==0 -> root node

        while len(queue) > 0:
            l = queue.popleft()
            # Obtain all output node, and add the output nodes to the queue if all its input nodes have been visited.
            # the initial "perturbed" property is set in BoundInput or BoundParams object, depending on ptb.
            for name_next in l.output_name:
                node_next = self._modules[name_next]
                if isinstance(l, BoundShape):
                    # Some nodes like Shape, even connected, do not really propagate bounds.
                    # TODO: make this a property of node?
                    pass
                else:
                    # The next node is perturbed if it is already perturbed, or this node is perturbed.
                    node_next.perturbed = node_next.perturbed or l.perturbed
                degree_in[name_next] -= 1
                if degree_in[name_next] == 0:  # all inputs of this node have been visited, now put it in queue.
                    queue.append(node_next)
        return

    def _clear_and_set_new(self, new_interval):
        for l in self._modules.values():
            if hasattr(l, 'linear'):
                if isinstance(l.linear, tuple):
                    for item in l.linear:
                        del(item)
                delattr(l, 'linear')
            for attr in ['forward_value', 'lower', 'upper', 'interval']:
                if hasattr(l, attr):
                    delattr(l, attr)
            # Given an interval here to make IBP/CROWN start from this node
            if new_interval is not None and l.name in new_interval.keys():
                l.interval = tuple(new_interval[l.name][:2])
                l.lower = new_interval[l.name][0]
                l.upper = new_interval[l.name][1]
            # Mark all nodes as non-perturbed except for weights.
            if not hasattr(l, 'perturbation') or l.perturbation is None:
                l.perturbed = False

    def _set_input(self, *x, new_interval=None):
        self._clear_and_set_new(new_interval=new_interval)
        inputs_unpacked = unpack_inputs(x)
        for name, index in zip(self.input_name, self.input_index):
            node = self._modules[name]
            node.value = inputs_unpacked[index]
            if isinstance(node.value, (BoundedTensor, BoundedParameter)):
                node.perturbation = node.value.ptb
            else:
                node.perturbation = None
        # Mark all perturbed nodes.
        self._mark_perturbed_nodes()
        if self.init_batch_size == -1:
            # Automatic batch dimension inferring: get the batch size from 
            # the first dimension of the first input tensor.
            self.init_batch_size = inputs_unpacked[0].shape[0]

    def _get_node_input(self, nodesOP, nodesIn, node):
        ret = []
        ori_names = []
        for i in range(len(node.inputs)):
            found = False
            for op in nodesOP:
                if op.name == node.inputs[i]:
                    ret.append(op.bound_node)
                    break
            if len(ret) == i + 1:
                continue
            for io in nodesIn:
                if io.name == node.inputs[i]:
                    ret.append(io.bound_node)
                    ori_names.append(io.ori_name)
                    break
            if len(ret) <= i:
                raise ValueError('cannot find inputs of node: {}'.format(node.name))
        return ret, ori_names

    # move all tensors in the object to a specified device
    def _to(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, tuple):
            return tuple([self._to(item, device) for item in obj])
        elif isinstance(obj, list):
            return list([self._to(item, device) for item in obj])
        elif isinstance(obj, dict):
            res = {}
            for key in obj:
                res[key] = self._to(obj[key], device)
            return res
        else:
            raise NotImplementedError(type(obj))

    def _convert_nodes(self, model, global_input):
        global_input_cpu = self._to(global_input, 'cpu')
        model.train()
        model.to('cpu')
        nodesOP, nodesIn, nodesOut, template = parse_module(model, global_input_cpu)
        model.to(self.device)
        for i in range(0, len(nodesIn)):
            if nodesIn[i].param is not None:
                nodesIn[i] = nodesIn[i]._replace(param=nodesIn[i].param.to(self.device))

        # FIXME: better way to handle buffers, do not hard-code it for BN!
        # Other nodes can also have buffers.
        bn_nodes = []
        for n in range(len(nodesOP)):
            if nodesOP[n].op == 'onnx::BatchNormalization':
                bn_nodes.extend(nodesOP[n].inputs[3:])  # collect names of  running_mean and running_var

        global_input_unpacked = unpack_inputs(global_input)

        # Convert input nodes and parameters.
        for i, n in enumerate(nodesIn):
            if n.input_index is not None:
                nodesIn[i] = nodesIn[i]._replace(bound_node=BoundInput(
                    nodesIn[i].inputs, nodesIn[i].name, nodesIn[i].ori_name,
                    value=global_input_unpacked[nodesIn[i].input_index], 
                    perturbation=nodesIn[i].perturbation))
            else:
                bound_class = BoundBuffers if n.name in bn_nodes else BoundParams                    
                nodesIn[i] = nodesIn[i]._replace(bound_node=bound_class(
                    nodesIn[i].inputs, nodesIn[i].name, nodesIn[i].ori_name,
                    value=nodesIn[i].param, perturbation=nodesIn[i].perturbation))

        unsupported_ops = []

        # Convert other operation nodes.
        for n in range(len(nodesOP)):
            attr = nodesOP[n].attr
            inputs, ori_names = self._get_node_input(nodesOP, nodesIn, nodesOP[n])

            try:
                if nodesOP[n].op in bound_op_map:
                    op = bound_op_map[nodesOP[n].op]
                elif nodesOP[n].op.startswith('onnx::'):
                    op = eval('Bound{}'.format(nodesOP[n].op[6:]))
                else:
                    raise KeyError
            except (NameError, KeyError):
                unsupported_ops.append(nodesOP[n].op)
                logger.error('The node has an unsupported operation: {}'.format(nodesOP[n]))
                continue

            if nodesOP[n].op == 'onnx::BatchNormalization':
                # BatchNormalization node needs model.training flag to set running mean and vars
                # set training=False to avoid wrongly updating running mean/vars during bound wrapper
                nodesOP[n] = nodesOP[n]._replace(
                    bound_node=op(
                        nodesOP[n].inputs, nodesOP[n].name, None, attr,
                        inputs, nodesOP[n].output_index, self.bound_opts, self.device, False))
            else:
                nodesOP[n] = nodesOP[n]._replace(
                    bound_node=op(
                        nodesOP[n].inputs, nodesOP[n].name, None, attr,
                        inputs, nodesOP[n].output_index, self.bound_opts, self.device))

        if unsupported_ops:
            logger.error('Unsupported operations:')
            for op in set(unsupported_ops):
                logger.error(op)
            raise NotImplementedError('There are unsupported operations')

        return nodesOP, nodesIn, nodesOut, template

    def _build_graph(self, nodesOP, nodesIn, nodesOut, template):
        nodes = []
        for node in nodesOP + nodesIn:
            assert (node.bound_node is not None)
            nodes.append(node.bound_node)
        # We were assuming that the original model had only one output node.
        # When there are multiple output nodes, this seems to be the first output element.
        # In this case, we are assuming that we always aim to compute the bounds for the first
        # output element.
        self.final_name = nodesOP[-1].name 
        assert self.final_name == nodesOut[0]
        self.input_name, self.input_index, self.root_name = [], [], []
        for node in nodesIn:
            self.root_name.append(node.name)
            if node.input_index is not None:
                self.input_name.append(node.name)
                self.input_index.append(node.input_index)
        self.output_name = nodesOut
        self.output_template = template
        for l in nodes:
            self._modules[l.name] = l
            l.output_name = []
            if isinstance(l.input_name, str):
                l.input_name = [l.input_name]
        for l in nodes:
            for l_pre in l.input_name:
                self._modules[l_pre].output_name.append(l.name)

    def _split_complex(self, nodesOP, nodesIn):
        found_complex = False
        for n in range(len(nodesOP)):
            if hasattr(nodesOP[n].bound_node, 'complex') and \
                    nodesOP[n].bound_node.complex:
                found_complex = True
                _nodesOP, _nodesIn, _, _ = self._convert_nodes(
                    nodesOP[n].bound_node.model, nodesOP[n].bound_node.input)
                name_base = nodesOP[n].name + '/split'
                rename_dict = {}
                for node in _nodesOP + _nodesIn:
                    rename_dict[node.name] = name_base + node.name

                num_inputs = len(nodesOP[n].bound_node.input)

                # assuming each supported complex operation only has one output
                for i in range(num_inputs):
                    rename_dict[_nodesIn[i].name] = nodesOP[n].inputs[i]
                rename_dict[_nodesOP[-1].name] = nodesOP[n].name

                def rename(node):
                    node = node._replace(name=rename_dict[node.name])
                    node = node._replace(inputs=[rename_dict[name] for name in node.inputs])
                    node.bound_node.name = rename_dict[node.bound_node.name]
                    node.bound_node.input_name = [
                        rename_dict[name] for name in node.bound_node.input_name]
                    return node

                for i in range(len(_nodesOP)):
                    _nodesOP[i] = rename(_nodesOP[i])
                for i in range(len(_nodesIn)):
                    _nodesIn[i] = rename(_nodesIn[i])

                nodesOP = nodesOP[:n] + _nodesOP + nodesOP[(n + 1):]
                nodesIn = nodesIn + _nodesIn[num_inputs:]
 
                break

        return nodesOP, nodesIn, found_complex

    """build a dict with {ori_name: name, name: ori_name}"""
    def _get_node_name_map(self, ):
        self.node_name_map = {}
        for node in self._modules.values():
            if isinstance(node, BoundInput) or isinstance(node, BoundParams):
                for p in list(node.named_parameters()):
                    if node.ori_name not in self.node_name_map:
                        self.node_name_map[node.ori_name] = node.name + '.' + p[0]
                        self.node_name_map[node.name + '.' + p[0]] = node.ori_name
                for p in list(node.named_buffers()):
                    if node.ori_name not in self.node_name_map:
                        self.node_name_map[node.ori_name] = node.name + '.' + p[0]
                        self.node_name_map[node.name + '.' + p[0]] = node.ori_name

    # convert a Pytorch model to a model with bounds
    def _convert(self, model, global_input):
        if self.verbose:
            logger.info('Converting the model...')

        if not isinstance(global_input, tuple):
            global_input = (global_input,)
        self.num_global_inputs = len(global_input)

        nodesOP, nodesIn, nodesOut, template = self._convert_nodes(model, global_input)
        global_input = self._to(global_input, self.device)

        while True:
            self._build_graph(nodesOP, nodesIn, nodesOut, template)
            self.forward(*global_input)  # running means/vars changed
            nodesOP, nodesIn, found_complex = self._split_complex(nodesOP, nodesIn)
            if not found_complex: 
                break

        self._get_node_name_map()

        # load self.ori_state_dict again to avoid the running means/vars changed during forward()
        self.load_state_dict(self.ori_state_dict)
        model.load_state_dict(self.ori_state_dict)
        delattr(self, 'ori_state_dict')

        # The final node used in the last time calling `compute_bounds`
        self.last_final_node = None

        logger.debug('NodesOP:')
        for node in nodesOP:
            logger.debug('{}'.format(node._replace(param=None)))
        logger.debug('NodesIn')
        for node in nodesIn:
            logger.debug('{}'.format(node._replace(param=None)))

        if self.verbose:
            logger.info('Model converted to support bounds')

    def get_optimized_bounds(self, x=None, aux=None, C=None, IBP=False, forward=False, method='backward', bound_lower=True,
                       bound_upper=True, reuse_ibp=False, return_A=False, final_node_name=None, average_A=False, new_interval=None, iteration=10):
        if x is None:
            output = self.forward(self.global_input)
        else:
            output = self.forward(x)

        parameters = []
        for model in self._modules.values():
            if isinstance(model, BoundRelu):
                parameters.append(model.slope)

        lr = 10
        opt = optim.SGD(parameters, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=0.1)
        last_l = 1e8

        for i in range(iteration):
            l, u = self.compute_bounds(x, aux, C, IBP, forward, method, bound_lower, bound_upper, reuse_ibp,
                                       return_A=False, final_node_name=final_node_name, average_A=average_A, new_interval=new_interval)
            assert l.shape[1] == 1
            if i == 0 or i == iteration - 1:
                print('optimal slope:', l.flatten(), scheduler.get_last_lr())
                if (l > 1e-4).all():  # all lower bounds > 0, no need to optimize
                    break

            l = l.sum()
            opt.zero_grad()
            l = (-1 * l) * (l < 1e-4)  # only optimize the lower bounds < 0
            # early stop
            if last_l <= l and iteration < 100:
                break
            l.backward(retain_graph=True)
            opt.step()
            # record.append(l.detach().item())
            if i > 5 or iteration < 100: scheduler.step()

            last_l = l.detach().clone()
        # np.save('solve_slope.npy', np.array(record))
        return self.compute_bounds(x, aux, C, IBP, forward, method, bound_lower, bound_upper, reuse_ibp, return_A, final_node_name, average_A, new_interval)

    def compute_bounds(self, x=None, aux=None, C=None, IBP=False, forward=False, method='backward', bound_lower=True,
                       bound_upper=True, reuse_ibp=False, 
                       return_A=False, final_node_name=None, average_A=False, new_interval=None,
                       return_b=False, b_dict=None):
        if not bound_lower and not bound_upper:
            raise ValueError('At least one of bound_lower and bound_upper in compute_bounds should be True')
        A_dict = {} if return_A else None
        if x is not None:
            self._set_input(*x, new_interval=new_interval)

        # Several shortcuts.
        method = method.lower() if method is not None else method
        if method == 'ibp':
            # Pure IBP bounds.
            method = None
            IBP = True
        elif method == 'ibp+backward' or method == 'ibp+crown' or method == 'crown-ibp':
            method = 'backward'
            IBP = True
        elif method == 'crown':
            method = 'backward'
        elif method == 'forward':
            forward = True
        elif method == 'forward+backward':
            method = 'backward'
            forward = True

        if IBP and method is None and reuse_ibp:
            # directly return the previously saved ibp bounds
            return self.ibp_lower, self.ibp_upper
        root = [self._modules[name] for name in self.root_name]
        batch_size = root[0].fv.shape[0]
        dim_in = 0
        for i in range(len(root)):
            value = root[i].forward()
            if root[i].perturbation is not None:
                root[i].linear, root[i].center, root[i].aux = \
                    root[i].perturbation.init(value, aux=aux, forward=forward)
                # This input/parameter has perturbation. Create an interval object.
                root[i].lower, root[i].upper = root[i].interval = \
                    Interval(root[i].linear.lower, root[i].linear.upper, root[i].perturbation)
                if forward:
                    root[i].dim = root[i].linear.lw.shape[1]
                    dim_in += root[i].dim
            else:
                # This inpute/parameter does not has perturbation. Use plain tuple defaulting to Linf perturbation.
                root[i].interval = (value, value)                
                root[i].lower = root[i].upper = value

        if forward:
            self._init_forward(root, dim_in)

        final = self._modules[self.final_name] if final_node_name is None else self._modules[final_node_name]
        logger.debug('Final node {}[{}]'.format(final, final.name))

        if C is None:
            # C is an identity matrix by default 
            if final.default_shape is None:
                raise ValueError('C is not provided while node {} has no default shape'.format(final.shape))
            dim_output = int(np.prod(final.default_shape[1:]))
            C = torch.eye(dim_output).to(self.device).unsqueeze(0).repeat(batch_size, 1, 1)

        if IBP:
            lower, upper = self._IBP_general(node=final, C=C)
            self.ibp_lower, self.ibp_upper = lower, upper

        if method is None:
            return self.ibp_lower, self.ibp_upper

        # check whether weights are perturbed and set nonlinear for the BoundMatMul operation
        for n in self._modules.values():
            if isinstance(n, (BoundLinear, BoundConv, BoundBatchNormalization)):
                n.nonlinear = False
                for l_name in n.input_name[1:]:
                    node = self._modules[l_name]
                    if hasattr(node, 'perturbation'):
                        if node.perturbation is not None:
                            n.nonlinear = True

        # BFS to find out whether each node is used given the current final node
        if final != self.last_final_node:
            self.last_final_node = final
            for i in self._modules.values():
                i.used = False
            final.used = True
            queue = deque([final])
            while len(queue) > 0:
                n = queue.popleft()
                for n_pre_name in n.input_name:
                    n_pre = self._modules[n_pre_name]
                    if not n_pre.used:
                        n_pre.used = True
                        queue.append(n_pre)

        for i in self._modules.values(): # for all nodes
            if not i.used:
                continue
            if hasattr(i, 'nonlinear') and i.nonlinear: 
                for l_name in i.input_name:
                    node = self._modules[l_name]
                    if not hasattr(node, 'lower'):
                        if not node.perturbed and hasattr(node, 'forward_value'):
                            node.interval = node.lower, node.upper = \
                                node.forward_value, node.forward_value
                            continue
                        # FIXME check that weight perturbation is not affected
                        #      (from_input=True should be set for weights)
                        if not node.from_input and hasattr(node, 'forward_value'):
                            node.lower = node.upper = node.forward_value
                            continue
                        if forward:
                            l, u = self._forward_general(
                                node=node, root=root, dim_in=dim_in, concretize=True)
                        else:
                            # assign concretized bound for ReLU layer to save computational cost
                            if (isinstance(node, BoundActivation) or isinstance(node, BoundTranspose)) and hasattr(self._modules[node.input_name[0]], 'lower'):
                                node.lower = node.forward(self._modules[node.input_name[0]].lower)
                                node.upper = node.forward(self._modules[node.input_name[0]].upper)
                            elif isinstance(node, BoundReshape) and \
                                    hasattr(self._modules[node.input_name[0]], 'lower') and \
                                    hasattr(self._modules[node.input_name[1]], 'value'):
                                # Node for input value.
                                val_input = self._modules[node.input_name[0]]
                                # Node for input parameter (e.g., shape, permute)
                                arg_input = self._modules[node.input_name[1]]
                                node.lower = node.forward(val_input.lower, arg_input.value)
                                node.upper = node.forward(val_input.upper, arg_input.value)
                            else:
                                # Here we avoid creating a big C matrix in the first linear layer
                                flag = False
                                if type(node) == BoundLinear or type(node) == BoundConv:
                                    for l_pre in node.input_name:
                                        if type(self._modules[l_pre]) == BoundInput:
                                            self._IBP_general(node)
                                            flag = True
                                            break
                                if not flag:
                                    dim = int(np.prod(node.default_shape[1:]))
                                    # FIXME: C matrix shape incorrect for BoundParams.
                                    if (isinstance(node, BoundLinear) or isinstance(node, BoundMatMul)) and int(os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0:
                                        newC = eyeC([batch_size, dim, *node.default_shape[1:]], self.device)
                                    elif (isinstance(node, BoundConv) or isinstance(node, BoundBatchNormalization)) and node.mode == "patches":
                                        # Here we create an Identity Patches object 
                                        newC = Patches(None, 1, 0, [batch_size, node.default_shape[-2] * node.default_shape[-1], node.default_shape[-3], node.default_shape[-3], 1, 1], 1)
                                    elif isinstance(node, BoundAdd) and node.mode == "patches":
                                        num_channel = node.default_shape[-3]
                                        patches = (torch.eye(num_channel, device=self.device)).unsqueeze(0).unsqueeze(0).unsqueeze(4).unsqueeze(5) # now [1 * 1 * in_C * in_C * 1 * 1]
                                        newC = Patches(patches, 1, 0, [batch_size] + list(patches.shape[1:]))
                                    else:
                                        newC = torch.eye(dim, device=self.device)\
                                            .unsqueeze(0).repeat(batch_size, 1, 1)\
                                            .view(batch_size, dim, *node.default_shape[1:])
                                    if return_A:
                                        _, _, A_dict = self._backward_general(C=newC, node=node, root=root,
                                                                            return_A=return_A, A_dict=A_dict)
                                    else:
                                        self._backward_general(C=newC, node=node, root=root, return_A=return_A)

        if method == 'backward':
            return self._backward_general(C=C, node=final, root=root, bound_lower=bound_lower, bound_upper=bound_upper,
                                          return_A=return_A, average_A=average_A, A_dict=A_dict,
                                          return_b=return_b, b_dict=b_dict)
        elif method == 'forward':
            return self._forward_general(C=C, node=final, root=root, dim_in=dim_in, concretize=True)
        else:
            raise NotImplementedError

    """ improvement on merging BoundLinear, BoundGatherElements and BoundSub
    when loss fusion is used in training"""
    def _IBP_loss_fusion(self, node, C):
        # not using loss fusion
        if not (isinstance(self.bound_opts, dict) and \
                'loss_fusion' in self.bound_opts and self.bound_opts['loss_fusion']):
            return None

        # Currently this function has issues in more complicated networks.
        if 'no_ibp_loss_fusion' in self.bound_opts and self.bound_opts['no_ibp_loss_fusion']:
            return None

        if C is None and isinstance(node, BoundSub):
            node_gather = self._modules[node.input_name[1]]
            if isinstance(node_gather, BoundGatherElements) or isinstance(node_gather, BoundGatherAten):
                node_linear = self._modules[node.input_name[0]]
                node_start = self._modules[node_linear.input_name[0]]
                if isinstance(node_linear, BoundLinear):
                    w = self._modules[node_linear.input_name[1]].fv
                    b = self._modules[node_linear.input_name[2]].fv
                    labels = self._modules[node_gather.input_name[1]]
                    if not hasattr(node_start, 'interval'):
                        self._IBP_general(node_start)
                    for inp in node_gather.input_name:
                        n = self._modules[inp]
                        if not hasattr(n, 'interval'):
                            self._IBP_general(n)
                    if torch.isclose(labels.lower, labels.upper, 1e-8).all():
                        labels = labels.lower
                        batch_size = labels.shape[0]
                        w = w.unsqueeze(0).repeat(batch_size, 1, 1)
                        w = w - torch.gather(w, dim=1, 
                            index=labels.unsqueeze(-1).repeat(1, w.shape[1], w.shape[2]))
                        b = b.unsqueeze(0).repeat(batch_size, 1)
                        b = b - torch.gather(b, dim=1,
                            index=labels.repeat(1, b.shape[1]))
                        lower, upper = node_start.interval
                        lower, upper = lower.unsqueeze(1), upper.unsqueeze(1)
                        node.lower, node.upper = node_linear.interval_propagate(
                            (lower, upper), (w, w), (b.unsqueeze(1), b.unsqueeze(1)))
                        node.interval = node.lower, node.upper = node.lower.squeeze(1), node.upper.squeeze(1)
                        return node.interval
        return None

    def _IBP_general(self, node=None, C=None):
        if hasattr(node, 'interval'):
            return node.interval
            
        if not node.perturbed and hasattr(node, 'forward_value'):
            node.interval = node.lower, node.upper = node.forward_value, node.forward_value
            return node.interval
            
        logger.debug('IBP at {}[{}]'.format(node, node.name))

        interval = self._IBP_loss_fusion(node, C)
        if interval is not None:
            return interval

        for n_pre in node.input_name:
            n = self._modules[n_pre]
            if not hasattr(n, 'interval'):
                self._IBP_general(n)

        inp = [self._modules[n_pre].interval for n_pre in node.input_name]
        if C is not None:
            if isinstance(node, BoundLinear) and not node.is_input_perturbed(1):
                # merge the output node with the specification, available when weights of this layer are not perturbed
                node.interval = node.interval_propagate(*inp, C=C)
            else:
                interval_before_C = [node.interval_propagate(*inp)]
                node.interval = BoundLinear.interval_propagate(None, *interval_before_C, C=C)
        else:
            node.interval = node.interval_propagate(*inp)
        node.lower, node.upper = node.interval

        if isinstance(node.lower, torch.Size):
            node.lower = torch.tensor(node.lower)
            node.interval = (node.lower, node.upper)
        if isinstance(node.upper, torch.Size):
            node.upper = torch.tensor(node.upper)
            node.interval = (node.lower, node.upper)

        return node.interval

    def _backward_general(self, C=None, node=None, root=None, bound_lower=True, bound_upper=True, 
            return_A=False, average_A=False, A_dict=None, return_b=False, b_dict=None):
        logger.debug('Backward from ({})[{}]'.format(node, node.name))

        _print_time = False

        degree_out = {}
        for l in self._modules.values():
            l.bounded = True
            l.lA = l.uA = None
            degree_out[l.name] = 0
        queue = deque([node])
        while len(queue) > 0:
            l = queue.popleft()
            for l_pre in l.input_name:
                degree_out[l_pre] += 1 # calculate the out degree
                if self._modules[l_pre].bounded:
                    self._modules[l_pre].bounded = False
                    queue.append(self._modules[l_pre])
        node.bounded = True
        batch_size, output_dim = C.shape[:2]

        if not isinstance(C, eyeC) and not isinstance(C, Patches):
            C = C.transpose(0, 1)
        elif isinstance(C, eyeC):
            C = C._replace(shape=(C.shape[1], C.shape[0], C.shape[2]))

        node.lA = C if bound_lower else None
        node.uA = C if bound_upper else None
        lb = ub = torch.tensor(0.).to(self.device)

        def _get_A_shape(node):
            shape_A = ''
            if bound_lower:
                try: shape_A += 'lA shape {} '.format(node.lA.shape) 
                except: pass
            if bound_upper:
                try: shape_A += 'uA shape {} '.format(node.uA.shape) 
                except: pass
            return shape_A

        queue = deque([node])
        A_record = {}
        while len(queue) > 0:
            l = queue.popleft()  # backward from l
            l.bounded = True

            if return_b:
                b_dict[l.name] = {
                    'lower_b': lb,
                    'upper_b': ub
                }            

            if l.name in self.root_name or l == root: continue

            for l_pre in l.input_name: # if all the succeeds are done, then we can turn to this node in the next iteration.
                _l = self._modules[l_pre]
                degree_out[l_pre] -= 1
                if degree_out[l_pre] == 0:
                    queue.append(_l)

            if l.lA is not None or l.uA is not None:
                def bound_add(A, B):
                    if type(A) == torch.Tensor and type(A) == torch.Tensor:
                        return A + B
                    elif type(A) == Patches and type(B) == Patches:
                        # Here we have to merge two patches, and if A.stride != B.stride, the patches will become a matrix, 
                        # in this case, we will avoid using this mode
                        assert A.stride == B.stride, "A.stride should be the same as B.stride, otherwise, please use the matrix mode"

                        # change paddings to merge the two patches
                        if A.padding != B.padding:
                            if A.padding > B.padding:
                                B = B._replace(patches = F.pad(B.patches, (A.padding - B.padding, A.padding - B.padding, A.padding - B.padding, A.padding - B.padding)))
                            else:
                                A = A._replace(patches = F.pad(A.patches, (B.padding - A.padding, B.padding - A.padding, B.padding - A.padding, B.padding - A.padding)))
                        return Patches(A.patches + B.patches, B.stride, max(A.padding, B.padding), (A.patches + B.patches).shape)
                    elif type(A) == BoundList:
                        A.bound_list.append(B)
                        return A

                def add_bound(node, lA, uA):
                    if lA is not None:
                        node.lA = lA if node.lA is None else bound_add(node.lA, lA)
                    if uA is not None:
                        node.uA = uA if node.uA is None else bound_add(node.uA, uA)

                # TODO can we just use l.inputs?
                input_nodes = [self._modules[l_name] for l_name in l.input_name]
                if _print_time:
                    start_time = time.time()

                # FIXME make fixed nodes have fixed `forward_value` that is never cleaned out
                if not l.perturbed and hasattr(l, 'forward_value'):
                    lb = lb + l.get_bias(l.lA, l.forward_value)
                    ub = ub + l.get_bias(l.uA, l.forward_value)
                    continue

                small_A = 0
                if l.lA is not None and not isinstance(l.lA, eyeC) and not isinstance(l.lA, Patches) and torch.norm(l.lA, p=1) < epsilon:
                    small_A += 1
                if l.uA is not None and not isinstance(l.uA, eyeC) and not isinstance(l.uA, Patches) and torch.norm(l.uA, p=1) < epsilon:
                    small_A += 1
                if small_A == 2:
                    continue

                small_A = 0
                if isinstance(l.lA, Patches) and l.lA.identity == 0 and torch.norm(l.lA.patches, p=1) < epsilon:
                    small_A += 1
                if isinstance(l.lA, Patches) and l.uA.identity == 0 and torch.norm(l.uA.patches, p=1) < epsilon:
                    small_A += 1
                if small_A == 2:
                    continue
                
                try:
                    try:
                        # TODO automatically check A shape
                        logger.debug('Backward at {}[{}], fv shape {}, {}'.format(
                            l, l.name, l.forward_value.shape, _get_A_shape(l)))
                    except: pass            
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *input_nodes)
                    if return_A:
                        A_record.update({l.name: A})
                except:
                    raise Exception('Error at bound_backward of {}, {}'.format(l, l.name))
  
                if _print_time:
                    time_elapsed = time.time() - start_time
                    if time_elapsed > 1e-3:
                        print(l, time_elapsed)
                lb = lb + lower_b
                ub = ub + upper_b                  

                for i, l_pre in enumerate(l.input_name):
                    try: logger.debug('  {} -> {}, uA shape {}'.format(l.name, l_pre, A[i][1].shape))
                    except: pass
                    _l = self._modules[l_pre]
                    add_bound(_l, lA=A[i][0], uA=A[i][1])

        if lb.ndim >= 2:
            lb = lb.transpose(0, 1)
        if ub.ndim >= 2:
            ub = ub.transpose(0, 1)
        output_shape = node.default_shape[1:]
        if np.prod(node.default_shape[1:]) != output_dim and type(C) != Patches:
            output_shape = [-1]

        if return_A:
            # return A matrix as a dict: {node.name: [A_lower, A_upper]}
            this_A_dict = {'bias': [lb, ub]}
            for i in range(len(root)):
                if root[i].lA is None and root[i].uA is None: continue
                this_A_dict.update({root[i].name: [root[i].lA, root[i].uA]})
            this_A_dict.update(A_record)
            A_dict.update({node.name: this_A_dict})

        for i in range(len(root)): 
            if root[i].lA is None and root[i].uA is None: continue
            # FIXME maybe this one is broken after moving the output dimension to the first
            if average_A and isinstance(root[i], BoundParams):
                A_shape = root[i].lA.shape if bound_lower else root[i].uA.shape
                lA = root[i].lA.mean(0, keepdim=True).repeat(A_shape[0], *[1]*len(A_shape[1:])) if bound_lower else None
                uA = root[i].uA.mean(0, keepdim=True).repeat(A_shape[0], *[1]*len(A_shape[1:])) if bound_upper else None
            else:
                lA = root[i].lA
                uA = root[i].uA
            
            if not isinstance(root[i].lA, eyeC) and not isinstance(root[i].lA, Patches):
                lA = root[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None
            if not isinstance(root[i].uA, eyeC) and not isinstance(root[i].lA, Patches):
                uA = root[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None
            if root[i].perturbation is not None:
                if isinstance(root[i], BoundParams):
                    # add batch_size dim for weights node
                    lb = lb + root[i].perturbation.concretize(
                        root[i].center.unsqueeze(0), lA,
                        sign=-1, aux=root[i].aux) if bound_lower else None
                    ub = ub + root[i].perturbation.concretize(
                        root[i].center.unsqueeze(0), uA,
                        sign=+1, aux=root[i].aux) if bound_upper else None
                else:
                    lb = lb + root[i].perturbation.concretize(root[i].center, lA, sign=-1, aux=root[i].aux) if bound_lower else None
                    ub = ub + root[i].perturbation.concretize(root[i].center, uA, sign=+1, aux=root[i].aux) if bound_upper else None
            # FIXME to simplify
            elif i < self.num_global_inputs:
                if not isinstance(lA, eyeC):
                    lb = lb + lA.bmm(root[i].fv.view(batch_size, -1, 1)).squeeze(-1) if bound_lower else None
                else:
                    lb = lb + root[i].fv.view(batch_size, -1) if bound_lower else None
                if not isinstance(uA, eyeC):
                    # FIXME looks questionable
                    ub = ub + uA.bmm(root[i].fv.view(batch_size, -1, 1)).squeeze(-1) if bound_upper else None
                else:
                    ub = ub + root[i].fv.view(batch_size, -1) if bound_upper else None
            else:
                if not isinstance(lA, eyeC):
                    lb = lb + lA.matmul(root[i].fv.view(-1, 1)).squeeze(-1) if bound_lower else None
                else:
                    lb = lb + root[i].fv.view(1, -1) if bound_lower else None
                if not isinstance(uA, eyeC):
                    # FIXME looks questionable
                    ub = ub + uA.matmul(root[i].fv.view(-1, 1)).squeeze(-1) if bound_upper else None
                else:
                    ub = ub + root[i].fv.view(1, -1) if bound_upper else None

        node.lower = lb.view(batch_size, *output_shape) if bound_lower else None
        node.upper = ub.view(batch_size, *output_shape) if bound_upper else None

        if return_A: return node.lower, node.upper, A_dict
        return node.lower, node.upper

    def _forward_general(self, C=None, node=None, root=None, dim_in=None, concretize=False):
        if hasattr(node, 'lower'):
            return node.lower, node.upper

        if not node.from_input:
            w = None
            b = node.forward_value
            node.linear = LinearBound(w, b, w, b, b, b)
            node.lower = node.upper = b
            node.interval = (node.lower, node.upper) 
            return node.interval

        if not hasattr(node, 'linear'):
            for l_pre in node.input_name:
                l = self._modules[l_pre]
                if not hasattr(l, 'linear'):
                    self._forward_general(node=l, root=root, dim_in=dim_in)

            inp = [self._modules[l_pre].linear for l_pre in node.input_name]

            if C is not None and isinstance(node, BoundLinear) and not node.is_input_perturbed(1):
                node.linear = node.bound_forward(dim_in, *inp, C=C)
                C_merged = True
            else:
                node.linear = node.bound_forward(dim_in, *inp)
                C_merged = False
            
            lw, uw = node.linear.lw, node.linear.uw
            lower, upper = node.linear.lb, node.linear.ub

            if C is not None and not C_merged:
                # FIXME use bound_forward of BoundLinear
                C_pos, C_neg = C.clamp(min=0), C.clamp(max=0)
                _lw = torch.matmul(lw, C_pos.transpose(-1, -2)) + torch.matmul(uw, C_neg.transpose(-1, -2))
                _uw = torch.matmul(uw, C_pos.transpose(-1, -2)) + torch.matmul(lw, C_neg.transpose(-1, -2))
                lw, uw = _lw, _uw
                _lower = torch.matmul(lower.unsqueeze(1), C_pos.transpose(-1, -2)) + \
                    torch.matmul(upper.unsqueeze(1), C_neg.transpose(-1, -2))
                _upper = torch.matmul(upper.unsqueeze(1), C_pos.transpose(-1, -2)) + \
                    torch.matmul(lower.unsqueeze(1), C_neg.transpose(-1, -2))        
                lower, upper = _lower.squeeze(1), _upper.squeeze(1)  
            else:
                lower, upper = lower.squeeze(1), upper.squeeze(1)  
        else:
            lw, uw = node.linear.lw, node.linear.uw
            lower, upper = node.linear.lb, node.linear.ub          

        if concretize:
            if node.linear.lw is not None:
                prev_dim_in = 0
                batch_size = lw.shape[0]
                assert (lw.ndim > 1)
                lA = lw.reshape(batch_size, dim_in, -1).transpose(1, 2)
                uA = uw.reshape(batch_size, dim_in, -1).transpose(1, 2)
                for i in range(len(root)):
                    if root[i].perturbation is not None:
                        _lA = lA[:, :, prev_dim_in : (prev_dim_in + root[i].dim)]
                        _uA = uA[:, :, prev_dim_in : (prev_dim_in + root[i].dim)]
                        lower = lower + root[i].perturbation.concretize(
                            root[i].center, _lA, sign=-1, aux=root[i].aux).view(lower.shape)
                        upper = upper + root[i].perturbation.concretize(
                            root[i].center, _uA, sign=+1, aux=root[i].aux).view(upper.shape)
                        prev_dim_in += root[i].dim
                if C is None:
                    node.linear = node.linear._replace(lower=lower, upper=upper)
            if C is None:
                node.lower, node.upper = lower, upper
            return lower, upper

    def _init_forward(self, root, dim_in):
        if dim_in == 0:
            raise ValueError("At least one node should have a specified perturbation")
        prev_dim_in = 0
        batch_size = root[0].fv.shape[0]
        for i in range(len(root)):
            if root[i].perturbation is not None:
                shape = root[i].linear.lw.shape
                device = root[i].linear.lw.device
                root[i].linear = root[i].linear._replace(
                    lw=torch.cat([
                        torch.zeros(shape[0], prev_dim_in, *shape[2:], device=device),
                        root[i].linear.lw,
                        torch.zeros(shape[0], dim_in - shape[1], *shape[2:], device=device)
                    ], dim=1),
                    uw=torch.cat([
                        torch.zeros(shape[0], prev_dim_in, *shape[2:], device=device),
                        root[i].linear.uw,
                        torch.zeros(shape[0], dim_in - shape[1] - prev_dim_in, *shape[2:], device=device)
                    ], dim=1)
                )
                if i >= self.num_global_inputs:
                    root[i].forward_value = root[i].forward_value.unsqueeze(0).repeat(
                        *([batch_size] + [1] * self.forward_value.ndim))
                prev_dim_in += shape[1]
            else:
                fv = root[i].forward_value
                shape = fv.shape
                if root[i].from_input:
                    w = torch.zeros(shape[0], dim_in, *shape[1:], device=self.device)
                else:
                    w = None
                b = fv
                root[i].linear = LinearBound(w, b, w, b, b, b)
                root[i].lower = root[i].upper = b
                root[i].interval = (root[i].lower, root[i].upper)

    """Add perturbation to an intermediate node and it is treated as an independent 
    node in bound computation."""
    def add_intermediate_perturbation(self, node, perturbation):
        node.perturbation = perturbation
        node.perturbed = True
        # NOTE This change is currently inreversible
        if not node.name in self.root_name:
            self.root_name.append(node.name)


class BoundDataParallel(DataParallel):
    # https://github.com/huanzhang12/CROWN-IBP/blob/master/bound_layers.py
    # This is a customized DataParallel class for our project
    def __init__(self, *inputs, **kwargs):
        super(BoundDataParallel, self).__init__(*inputs, **kwargs)
        self._replicas = None

    # Overide the forward method
    def forward(self, *inputs, **kwargs):
        disable_multi_gpu = False  # forward by single GPU
        no_replicas = False  # forward by multi GPUs but without replicate
        if "disable_multi_gpu" in kwargs:
            disable_multi_gpu = kwargs["disable_multi_gpu"]
            kwargs.pop("disable_multi_gpu")

        if "no_replicas" in kwargs:
            no_replicas = kwargs["no_replicas"]
            kwargs.pop("no_replicas")

        if not self.device_ids or disable_multi_gpu:
            if kwargs.pop("get_property", False):
                return self.get_property(self, *inputs, **kwargs)
            return self.module(*inputs, **kwargs)            

        if kwargs.pop("get_property", False):
            if self._replicas is None:
                assert 0, 'please call IBP/CROWN before get_property'
            if len(self.device_ids) == 1:
                return self.get_property(self.module, **kwargs)
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            kwargs = list(kwargs)
            for i in range(len(kwargs)):
                kwargs[i]['model'] = self._replicas[i]
            outputs = self.parallel_apply([self.get_property]*len(kwargs), inputs, kwargs)
            return self.gather(outputs, self.output_device)

        # Only replicate during forward/IBP propagation. Not during interval bounds
        # and CROWN-IBP bounds, since weights have not been updated. This saves 2/3
        # of communication cost.
        if not no_replicas:
            if self._replicas is None:  # first time
                self._replicas = self.replicate(self.module, self.device_ids)
            elif kwargs.get("method_opt", "forward") == "forward":
                self._replicas = self.replicate(self.module, self.device_ids)
            elif kwargs.get("x") is not None and kwargs.get("IBP") is True:  #
                self._replicas = self.replicate(self.module, self.device_ids)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        # TODO: can be done in parallel, only support same ptb for all inputs per forward/IBP propagation
        if len(inputs) > 0 and hasattr(inputs[0], 'ptb') and inputs[0].ptb is not None:
            # compute bounds without x
            # inputs_scatter is a normal tensor, we need to assign ptb to it if inputs is a BoundedTensor
            inputs_scatter, kwargs = self.scatter((inputs, inputs[0].ptb.x_L, inputs[0].ptb.x_U), kwargs, self.device_ids)
            # inputs_scatter = inputs_scatter[0]
            bounded_inputs = []
            for input_s in inputs_scatter:  # GPU numbers
                ptb = PerturbationLpNorm(norm=inputs[0].ptb.norm, eps=inputs[0].ptb.eps, x_L=input_s[1], x_U=input_s[2])
                # bounded_inputs.append(tuple([(BoundedTensor(input_s[0][0], ptb))]))
                input_s = list(input_s[0])
                input_s[0] = BoundedTensor(input_s[0], ptb)
                input_s = tuple(input_s)
                bounded_inputs.append(input_s)

            # bounded_inputs = tuple(bounded_inputs)
        elif kwargs.get("x") is not None and hasattr(kwargs.get("x")[0], 'ptb') and kwargs.get("x")[0].ptb is not None:
            # compute bounds with x
            # kwargs['x'] is a normal tensor, we need to assign ptb to it
            x = kwargs.get("x")[0]
            bounded_inputs = []
            inputs_scatter, kwargs = self.scatter((inputs, x.ptb.x_L, x.ptb.x_U), kwargs, self.device_ids)
            for input_s, kw_s in zip(inputs_scatter, kwargs):  # GPU numbers
                ptb = PerturbationLpNorm(norm=x.ptb.norm, eps=x.ptb.eps, x_L=input_s[1], x_U=input_s[2])
                kw_s['x'] = list(kw_s['x'])
                kw_s['x'][0] = BoundedTensor(kw_s['x'][0], ptb)
                kw_s['x'] = (kw_s['x'])
                bounded_inputs.append(tuple(input_s[0], ))
        else:
            # normal forward
            inputs_scatter, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            bounded_inputs = inputs_scatter

        if len(self.device_ids) == 1:
            return self.module(*bounded_inputs[0], **kwargs[0])
        outputs = self.parallel_apply(self._replicas[:len(bounded_inputs)], bounded_inputs, kwargs)
        return self.gather(outputs, self.output_device)

    @staticmethod
    def get_property(model, node_class, att_name):
        for _, node in model.named_modules():
            # Find the Exp neuron in computational graph
            if isinstance(node, node_class):
                return getattr(node, att_name)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # add 'module.' here before each keys in self.module.state_dict() if needed
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        return self.module._named_members(get_members_fn, prefix, recurse)
