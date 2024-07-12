import os
import torch
from collections import OrderedDict
import re
from collections import namedtuple
from torch.onnx import OperatorExportTypes
from packaging import version
from auto_LiRPA.bounded_tensor import BoundedTensor, BoundedParameter
from auto_LiRPA.utils import logger, unpack_inputs

Node = namedtuple('Node', (
    'name', 'ori_name', 'inputs', 'attr', 'op', 'param', 'input_index', 
    'bound_node', 'output_index', 'perturbation'))

def replace(name, scope):
    return '/'.join([scope[name], name])

def get_node_name(node):
    return node.debugName()

def parse_graph(graph, inputs, params):
    # in what scope is each node used as an input
    scope = {}

    for n in graph.nodes():
        n_inputs = [get_node_name(i) for i in n.inputs()]

        for inp in n_inputs:
            if not inp in scope:
                scope[inp] = n.scopeName()

        for out in n.outputs():
            name = out.debugName()
            scope[name] = n.scopeName()

    nodesOP = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}

        n_inputs = [replace(get_node_name(i), scope) for i in n.inputs()]

        for i, out in enumerate(list(n.outputs())):
            name = get_node_name(out)
            nodesOP.append(Node(**{'name': replace(name, scope),
                                'ori_name': '',
                                'op': n.kind(),
                                'inputs': n_inputs,
                                'attr': attrs,
                                'param': None,  # will assign parameters later
                                'input_index': None, # for input nodes only
                                'bound_node': None, 
                                'output_index': i, 
                                'perturbation': None, }))
            if n.kind() == 'onnx::BatchNormalization': 
                break  # bn layer has some redundant outputs
    nodesOP_dict = {}
    for n in nodesOP:
        nodesOP_dict[n.name] = n

    # filter out input nodes in `graph.inputs()` that are actually used
    nodesIn = []
    used_index = []
    for i, n in enumerate(graph.inputs()):
        name = get_node_name(n)
        used = name in scope.keys()
        used_index.append(used)
        if used:
            nodesIn.append(n)

    # filter out input nodes in `inputs` that are actually used
    inputs_unpacked = unpack_inputs(inputs)
    assert len(list(graph.inputs())) == len(inputs_unpacked) + len(params)
    inputs = [inputs_unpacked[i] for i in range(len(inputs_unpacked)) if used_index[i]]  
    # index of the used inputs among all the inputs
    input_index = [i for i in range(len(inputs_unpacked)) if used_index[i]]
    # Add a name to all inputs
    inputs = list(zip(["input_{}".format(input_index[i]) for i in range(len(inputs))], inputs))
    # filter out params that are actually used
    params = [params[i] for i in range(len(params)) if used_index[i + len(inputs_unpacked)]]
    inputs_and_params = inputs + params
    assert len(nodesIn) == len(inputs_and_params) 

    # output nodes of the module
    nodesOut = []
    for n in graph.outputs():
        # we only record names
        nodesOut.append(replace(get_node_name(n), scope))

    for i, n in enumerate(nodesIn):
        name = get_node_name(n)
        if isinstance(inputs_and_params[i][1], BoundedTensor) or \
                isinstance(inputs_and_params[i][1], BoundedParameter):
            perturbation = inputs_and_params[i][1].ptb
        else:
            perturbation = None
        if n.type().sizes() != list(inputs_and_params[i][1].size()):
            raise RuntimeError("Input tensor shapes do not much: {} != {}".format(
                n.type().sizes(), list(inputs_and_params[i][1].size())))
        nodesIn[i] = Node(**{'name': replace(name, scope),
                             'ori_name': inputs_and_params[i][0],
                             'op': 'Parameter',
                             'inputs': [], 
                             'attr': str(n.type()),
                             'param': inputs_and_params[i][1] if i >= len(inputs) else None,
                             # index among all the inputs including unused ones 
                             'input_index': input_index[i] if i < len(inputs) else None,
                             'bound_node': None,
                             'output_index': None,
                             # Input nodes may have perturbation, if they are wrapped in BoundedTensor or BoundedParameters
                             'perturbation': perturbation, })

    return nodesOP, nodesIn, nodesOut

def _get_jit_params(module, param_exclude, param_include):
    state_dict = torch.jit._unique_state_dict(module, keep_vars=True)

    if param_exclude is not None:
        param_exclude = re.compile(param_exclude)
    if param_include is not None:
        param_include = re.compile(param_include)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if param_exclude is not None and param_exclude.match(k) is not None:
            print('\nremove input element {} from nodesIn\n'.format(k))
            continue
        if param_include is not None and param_include.match(k) is None:
            continue
        new_state_dict[k] = v

    params = zip(new_state_dict.keys(), new_state_dict.values())

    return params

"""Construct a template for the module output with `None` representing places 
to be filled with tensor results"""
def get_output_template(out):
    if isinstance(out, torch.Tensor):
        return None
    elif isinstance(out, list):
        return list([get_output_template(o) for o in out])
    elif isinstance(out, tuple):
        return tuple([get_output_template(o) for o in out])
    elif isinstance(out, dict):
        template = {}
        for key in out:
            template[key] = get_output_template(out[key])
        return template
    else:
        raise NotImplementedError

def parse_module(module, inputs, param_exclude=".*AuxLogits.*", param_include=None):
    params = _get_jit_params(module, param_exclude=param_exclude, param_include=param_include)
    if version.parse(torch.__version__) < version.parse("1.4.0"):
        trace, out = torch.jit.get_trace_graph(module, inputs)
        torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        trace_graph = trace.graph()
    else:
        # _get_trace_graph becomes an internal function in version >= 1.4.0
        trace, out = torch.jit._get_trace_graph(module, inputs)
        # this is not present in older torch
        from torch.onnx.symbolic_helper import _set_opset_version
        if version.parse(torch.__version__) < version.parse("1.5.0"):
            _set_opset_version(11)
        else:
            _set_opset_version(12)
        trace_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

    logger.debug('trace_graph: {}'.format(trace_graph))

    if int(os.environ.get('AUTOLIRPA_DEBUG_GRAPH', 0)) > 0:
        print("Graph before ONNX convertion:")
        print(trace)
        print("ONNX graph:")
        print(trace_graph)

    if not isinstance(inputs, tuple):
        inputs = (inputs, )
    
    nodesOP, nodesIn, nodesOut = parse_graph(trace_graph, tuple(inputs), tuple(params))

    for i in range(len(nodesOP)):
        param_in = OrderedDict()
        for inp in nodesOP[i].inputs:
            for n in nodesIn:
                if inp == n.name:
                    param_in.update({inp:n.param})
        nodesOP[i] = nodesOP[i]._replace(param=param_in)

    template = get_output_template(out)

    return nodesOP, nodesIn, nodesOut, template
