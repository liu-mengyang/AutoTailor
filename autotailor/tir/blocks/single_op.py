import math

import numpy as np
import onnx_graphsurgeon as gs

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

from .block import Block
import autotailor.tir.globvar as globvar
from autotailor.tir.build_utils import (generate_torch_block,
                                        sub_filter_start_end, get_padding)

class ReshapeModule(nn.Module):
    def __init__(self, shape):
        super(ReshapeModule, self).__init__()
        self.shape = shape

    def forward(self, x):
        # print(f"Reshape: {x.shape} -> {self.shape}")
        return torch.reshape(x, self.shape)


class SingleOp(Block):
    def __init__(self, gs_node: gs.Node):
        super().__init__()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.device_str = f"cuda:{dist.get_rank()}"
        elif torch.cuda.is_available():
            self.device_str = "cuda:0"
        else:
            self.device_str = "cpu"

        self.type = gs_node.op
        self.name = gs_node.name
        self.features = {
            'activated': True,
            'trans_type': 'passive',
            'in_shape': (),
            'out_shape': (),
            'in_channel': 0,
            'out_channel': 0,
            "max_in_channel": 0,
            "max_out_channel": 0,
            "has_weights": False,
        }
        # point to the instantiated module
        self.pointer = None

        if globvar.tir_prebuild:
            self._pre_build()

    @property
    def info(self):
        # in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        return info_tuple

    def bind_weight(self):
        pass

    def count_flops_params(self):
        return 0, 0

    def _pre_build(self):
        self.super_module = generate_torch_block([self.name])

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
        else:
            op_module = self.super_module
        self.pointer = op_module
        return op_module

    def update_weights(self, weight_name, param):
        raise NotImplementedError

    def get_weights(self):
        return None

    def update(self, kv_features: dict):
        super().update(kv_features)
        self.features['out_shape'] = self.features['in_shape']
        self.features['out_channel'] = self.features['in_channel']
        if self.features["max_out_channel"] == 0:
            # init
            self.features["max_out_channel"] = self.features["in_channel"]

        # if self.name == "/feature_exactor.2/layers.0/Mul":
            # print(self.features)

    def update_grad(self):
        pass

    @property
    def info_dict(self):
        ret_dict = super().info_dict

        return ret_dict


class ConvOp(SingleOp):
    def __init__(self, gs_node: gs.Node):
        """Build conv single operator block.

        Args:
            gs_node: node in graphsurgeon.
        """
        super(ConvOp, self).__init__(gs_node)
        self.features['trans_type'] = 'active'
        self.features['max_in_channel'] = gs_node.inputs[1].shape[1]
        self.features['max_out_channel'] = gs_node.inputs[1].shape[0]
        self.features['max_kernel_size'] = gs_node.attrs['kernel_shape'][0]
        self.features['in_channel'] =  gs_node.inputs[1].shape[1]
        self.features['out_channel'] = gs_node.inputs[1].shape[0]
        self.features['kernel_size'] = gs_node.attrs['kernel_shape'][0]
        self.features['stride'] = gs_node.attrs['strides'][0]
        self.features['group'] = gs_node.attrs['group']
        self.features['has_bias'] = True if len(gs_node.inputs)==3 else False
        self.features["has_weights"] = True

        if globvar.sharing:
            globvar.super_weights[self.name] = {}
            globvar.super_weights[self.name]["weight"] = nn.Parameter(
                torch.tensor(gs_node.inputs[1].values,
                             device=self.device_str,
                             requires_grad=True))
            self.super_weights = globvar.super_weights[self.name]["weight"]
            if self.features["has_bias"]:
                globvar.super_weights[self.name]["bias"] = nn.Parameter(
                    torch.tensor(gs_node.inputs[2].values,
                                 device=self.device_str,
                                 requires_grad=True))
                self.super_bias = globvar.super_weights[self.name]["bias"]
        else:
            self.super_weights = torch.tensor(gs_node.inputs[1].values, requires_grad=True)
            if self.features["has_bias"]:
                self.super_bias = torch.tensor(gs_node.inputs[2].values, requires_grad=True)

        self.transform_matrix_dict = {}

    def bind_weight(self):
        self.super_weights = globvar.super_weights[self.name]["weight"]
        if self.features["has_bias"]:
            self.super_bias = globvar.super_weights[self.name]["bias"]

    @property
    def info(self):
        # kernel_size, stride, group, has_bias, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["kernel_size"],
            self.features["stride"],
            self.features["group"],
            self.features["has_bias"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        # assert self.features["in_shape"][1] == self.features["in_channel"], self.name
        # assert self.features["out_shape"][1] == self.features["out_channel"], self.name
        return info_tuple

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError(f"{k} is invalid")

        inp_shape = self.features['in_shape']
        if len(inp_shape)==4:
            b, c, h, w = inp_shape
        elif len(inp_shape)==3:
            c, h, w = inp_shape
        else:
            raise NotImplementedError

        stride = self.features['stride']

        if stride == 1:
            h = w = h
        else:
            h = w = math.ceil(h / stride)

        if len(inp_shape)==4:
            out_shape = (b, self.features['out_channel'], h, w)
        elif len(inp_shape)==3:
            out_shape = (self.features['out_channel'], h, w)

        self.features['out_shape'] = out_shape # update out shape

    def get_weights(self):
        weights = [self.super_weights]
        if self.features["has_bias"]:
            weights.append(self.super_bias)
        return weights

    def update_weights(self, weight_name, param):
        if weight_name == "weight":
            self.super_weights = param
        elif weight_name == "bias":
            self.super_bias = param
        else:
            raise NotImplementedError

    def update_grad(self):
        # weight
        grad = self.pointer.weight.grad
        cout, cin, ks_h, ks_w = grad.shape
        self.super_weights.grad[:cout, :cin, :ks_h, :ks_w] += grad
        if self.features["has_bias"]:
            # bias
            grad = self.pointer.bias.grad
            cout = grad.shape[0]
            self.super_bias.grad[:cout] += grad

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module

        in_channel_size = self.features["in_channel"]
        out_channel_size = self.features["out_channel"]
        kernel_size = self.features["kernel_size"]
        stride = self.features["stride"]
        group = self.features["group"]
        has_bias = self.features["has_bias"]

        hw = self.features["in_shape"][-1]
        padding = get_padding(kernel_size, stride, hw)

        # load params
        if len(self.transform_matrix_dict) > 0:
            # kernel transform mode
            # start from full weights
            filters = self.super_weights[:out_channel_size,
                                         :in_channel_size,
                                         :,
                                         :]
            src_k = self.features["max_kernel_size"]
            while src_k > kernel_size:
                find_flag = False
                for matrix_k, matrix_v in self.transform_matrix_dict.items():
                    split_ks = matrix_k.split("to")
                    if int(split_ks[0]) == src_k:
                        tar_k = int(split_ks[1])
                        start, end = sub_filter_start_end(src_k, tar_k)
                        _input_filter = filters[:, :, start:end, start:end]
                        _input_filter = _input_filter.contiguous()
                        _input_filter = _input_filter.view(
                            _input_filter.size(0), _input_filter.size(1), -1
                        )
                        _input_filter = _input_filter.view(-1, _input_filter.size(2))
                        _input_filter = F.linear(_input_filter, matrix_v)
                        _input_filter = _input_filter.view(
                            filters.size(0), filters.size(1), tar_k ** 2
                        )
                        _input_filter = _input_filter.view(
                            filters.size(0), filters.size(1), tar_k, tar_k
                        )
                        filters = _input_filter
                        src_k = tar_k
                        find_flag = True
                        break
                if not find_flag:
                    raise NotImplementedError
        else:
            start, end = sub_filter_start_end(self.features["max_kernel_size"],
                                          kernel_size)
            filters = self.super_weights[:out_channel_size,
                                :in_channel_size,
                                start:end,
                                start:end]

        self.module = nn.Conv2d(in_channel_size,
                           out_channel_size,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=group,
                           bias=has_bias)
        with torch.no_grad():
            self.module.weight = nn.Parameter(filters)

            if has_bias:
                self.module.bias = nn.Parameter(self.super_bias[:out_channel_size])
        self.pointer = self.module
        return self.module

    def count_flops_params(self):
        ks = self.features['kernel_size']
        s = self.features['stride']
        g = self.features['group']
        hw = self.features['in_shape'][-1]
        cin = self.features['in_channel']
        cout = self.features['out_channel']

        params = cout * ks * ks * cin
        flops = 2 * hw / s * hw / s * params / g
        return flops, params


class DepthConvOp(ConvOp):
    def __init__(self, gs_node: gs.Node):
        """Build depth conv single operator block

        Args:
            gs_node: node in graphsurgeon
        """
        super(DepthConvOp, self).__init__(gs_node)
        self.type = 'DepthConv'

        self.features['in_channel'] = self.features['out_channel']
        self.features['max_in_channel'] = self.features['max_out_channel']

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        super().update(kv_features)
        self.features["out_channel"] = self.features["in_channel"]
        self.features['group'] = self.features['out_channel'] # update group

    def count_flops_params(self):
        ks = self.features['kernel_size']
        s = self.features['stride']
        g = self.features['group']
        hw = self.features['in_shape'][-1]
        cout = self.features['out_channel']

        params = cout * ks * ks
        flops = 2 * hw / s * hw / s * params * (cout // g)
        return flops, params


class LinearOp(SingleOp):
    def __init__(self, gs_node, feature_map=None):
        """Build linear single operator block

        Args:
            gs_node (graphsurgeon.Node): node in graphsurgeon
            feature_map (dict, optional): the map of the inference features. Defaults to None.
        """
        super().__init__(gs_node)
        self.features['trans_type'] = 'active'
        self.features['source_type'] = gs_node.op
        self.features['has_bias'] = True if len(gs_node.inputs)==3 else False
        self.features["has_weights"] = True
        self.type = 'Linear'
        if gs_node.op == 'Gemm':
            self.features['max_in_channel'] = gs_node.inputs[1].shape[1]
            self.features['max_out_channel'] = gs_node.inputs[1].shape[0]
            self.features['in_channel'] = gs_node.inputs[1].shape[1]
            self.features['out_channel'] = gs_node.inputs[1].shape[0]
        elif gs_node.op == 'MatMul':
            self.features['max_in_channel'] = gs_node.inputs[1].shape[0]
            self.features['in_channel'] = gs_node.inputs[1].shape[0]
            self.features['max_out_channel'] = gs_node.inputs[1].shape[1]
            self.features['out_channel'] = gs_node.inputs[1].shape[1]

        if globvar.sharing:
            globvar.super_weights[self.name] = {}
            globvar.super_weights[self.name]["weight"] = nn.Parameter(
                torch.tensor(gs_node.inputs[1].values,
                             device=self.device_str,
                             requires_grad=True))
            self.super_weights = globvar.super_weights[self.name]["weight"]
            if self.features["has_bias"]:
                globvar.super_weights[self.name]["bias"] = nn.Parameter(
                    torch.tensor(gs_node.inputs[2].values,
                                 device=self.device_str,
                                 requires_grad=True))
                self.super_bias = globvar.super_weights[self.name]["bias"]
        else:
            self.super_weights = torch.tensor(gs_node.inputs[1].values, requires_grad=True)
            if self.features["has_bias"]:
                self.super_bias = torch.tensor(gs_node.inputs[2].values, requires_grad=True)

    def bind_weight(self):
        self.super_weights = globvar.super_weights[self.name]["weight"]
        if self.features["has_bias"]:
            self.super_bias = globvar.super_weights[self.name]["bias"]

    @property
    def info(self):
        # has_bias, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["has_bias"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"],
        )
        return info_tuple

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        if len(inp_shape)==2:
            b, cin = inp_shape
            out_shape = (b, self.features['out_channel'])

            self.features['out_shape'] = out_shape # update out shape
            self.features['in_channel'] = cin # update in width
            return out_shape
        elif len(inp_shape)==3:
            b, m, n = inp_shape
            out_shape = (b, m, self.features['out_channel'])
            self.features['out_shape'] = out_shape # update out shape
            self.features['in_channel'] = n # update in width

            return out_shape
        elif len(inp_shape)==4:
            b, hw, m, n = inp_shape
            out_shape = (b, hw, m, self.features['out_channel'])
            self.features['out_shape'] = out_shape # update out shape
            self.features['in_channel'] = n # update in width

            return out_shape
        else:
            raise NotImplementedError

    def get_weights(self):
        weights = [self.super_weights]
        if self.features["has_bias"]:
            weights.append(self.super_bias)
        return weights

    def update_weights(self, weight_name, param):
        if weight_name == "weight":
            self.super_weights = param
        elif weight_name == "bias":
            self.super_bias = param
        else:
            raise NotImplementedError

    def update_grad(self):
        # weight
        grad = self.pointer.weight.grad
        cout, cin = grad.shape
        self.super_weights.grad[:cout, :cin] += grad
        if self.features["has_bias"]:
            # bias
            grad = self.pointer.bias.grad
            cout = grad.shape[0]
            self.super_bias.grad[:cout] += grad

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module
        in_channel_size = self.features["in_channel"]
        out_channel_size = self.features["out_channel"]
        has_bias = self.features["has_bias"]

        self.module = nn.Linear(in_channel_size,
                                out_channel_size,
                                bias=has_bias)

        # load params
        with torch.no_grad():
            self.module.weight = nn.Parameter(self.super_weights[:out_channel_size,:in_channel_size])
            if has_bias:
                self.module.bias = nn.Parameter(self.super_bias[:out_channel_size])
        self.pointer = self.module
        return self.module

    def count_flops_params(self):
        cin = self.features['in_channel']
        cout = self.features['out_channel']
        in_shape = self.features['in_shape']

        if len(in_shape)==2:
            batch = in_shape[0]
        elif len(in_shape)==3:
            batch = in_shape[0]*in_shape[1]
        elif len(in_shape)==4:
            batch = in_shape[0]*in_shape[1]*in_shape[2]

        params = cin * cout
        flops = 2 * params * batch
        return flops, params

# The following ops has weights

class BNOp(SingleOp):
    def __init__(self, gs_node: gs.Node):
        """Build batchnorm single operator block

        Args:
            gs_node: node in graphsurgeon
        """
        super().__init__(gs_node)
        self.type = "BatchNormalization"
        self.features['in_channel'] = gs_node.inputs[1].shape[0]
        self.features['out_channel'] = gs_node.inputs[1].shape[0]
        self.features['max_in_channel'] = gs_node.inputs[1].shape[0]
        self.features['max_out_channel'] = gs_node.inputs[1].shape[0]

        self.features["momentum"] = gs_node.attrs["momentum"]
        self.features["epsilon"] = gs_node.attrs["epsilon"]
        self.features["has_weights"] = True
        self.features['has_bias'] = True if len(gs_node.inputs)>=3 else False

        if globvar.sharing:
            globvar.super_weights[self.name] = {}
            globvar.super_weights[self.name]["weight"] = nn.Parameter(
                torch.tensor(gs_node.inputs[1].values,
                             device=self.device_str,
                             requires_grad=True))
            globvar.super_weights[self.name]["bias"] = nn.Parameter(
                torch.tensor(gs_node.inputs[2].values,
                             device=self.device_str,
                             requires_grad=True))
            globvar.super_weights[self.name]["mean"] = torch.tensor(gs_node.inputs[3].values,
                                                                    device=self.device_str,)
            globvar.super_weights[self.name]["var"] = torch.tensor(gs_node.inputs[4].values,
                                                                    device=self.device_str,)
            self.super_weights = globvar.super_weights[self.name]["weight"]
            self.super_bias = globvar.super_weights[self.name]["bias"]
            self.super_mean = globvar.super_weights[self.name]["mean"]
            self.super_var = globvar.super_weights[self.name]["var"]
        else:
            self.super_weights = torch.tensor(gs_node.inputs[1].values, requires_grad=True)
            self.super_bias = torch.tensor(gs_node.inputs[2].values, requires_grad=True)
            self.super_mean = torch.tensor(gs_node.inputs[3].values)
            self.super_var = torch.tensor(gs_node.inputs[4].values)

    def bind_weight(self):
        self.super_weights = globvar.super_weights[self.name]["weight"]
        self.super_bias = globvar.super_weights[self.name]["bias"]
        self.super_mean = globvar.super_weights[self.name]["mean"]
        self.super_var = globvar.super_weights[self.name]["var"]

    def get_weights(self):
        weights = [self.super_weights]
        if self.features["has_bias"]:
            weights.append(self.super_bias)
        return weights

    def update_weights(self, weight_name, param):
        if weight_name == "weight":
            self.super_weights = param
        elif weight_name == "bias":
            self.super_bias = param
        else:
            raise NotImplementedError

    def update_grad(self):
        # weight
        grad = self.pointer.weight.grad
        c = grad.shape[0]

        self.super_weights.grad[:c] += grad
        if self.features["has_bias"]:
            # bias
            grad = self.pointer.bias.grad
            c = grad.shape[0]
            self.super_bias.grad[:c] += grad

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module
        in_shape = self.features["in_shape"]
        in_channel_size = self.features["in_channel"]

        if len(in_shape) == 4:
            self.module = nn.BatchNorm2d(in_channel_size,
                                        momentum=self.features["momentum"],
                                        eps=self.features["epsilon"])
        elif len(in_shape) == 2 or len(in_shape) == 3:
            self.module = nn.BatchNorm1d(in_channel_size,
                                        momentum=self.features["momentum"],
                                        eps=self.features["epsilon"])

        # load params
        with torch.no_grad():
            self.module.weight = nn.Parameter(self.super_weights[:in_channel_size])
            self.module.bias = nn.Parameter(self.super_bias[:in_channel_size])
            self.module.running_mean = (self.super_mean[:in_channel_size])
            self.module.running_var = (self.super_var[:in_channel_size])
        self.pointer = self.module
        return self.module


class LNOp(SingleOp):
    def __init__(self, gs_node: gs.Node):
        """Build layernorm single operator block

        Args:
            gs_node: node in graphsurgeon
        """
        super().__init__(gs_node)
        self.type = "LayerNormalization"
        self.features['in_channel'] = gs_node.inputs[1].shape[-1]
        self.features['out_channel'] = gs_node.inputs[1].shape[-1]
        self.features['max_in_channel'] = gs_node.inputs[1].shape[-1]
        self.features['max_out_channel'] = gs_node.inputs[1].shape[-1]

        # list() so these are plain Python lists, not protobuf RepeatedScalarContainer
        # (the latter is unpicklable -> breaks spawn-based multiprocessing for ViT)
        self.features["dims"] = list(gs_node.inputs[1].shape)
        self.features["max_dims"] = list(gs_node.inputs[1].shape)

        self.features["axis"] = gs_node.attrs["axis"]
        self.features["epsilon"] = gs_node.attrs["epsilon"]
        self.features["has_weights"] = True
        self.features['has_bias'] = True if len(gs_node.inputs)>=3 else False
        if globvar.sharing:
            globvar.super_weights[self.name] = {}
            globvar.super_weights[self.name]["weight"] = nn.Parameter(
                torch.tensor(gs_node.inputs[1].values,
                             device=self.device_str,
                             requires_grad=True))
            globvar.super_weights[self.name]["bias"] = nn.Parameter(
                torch.tensor(gs_node.inputs[2].values,
                             device=self.device_str,
                             requires_grad=True))
            self.super_weights = globvar.super_weights[self.name]["weight"]
            self.super_bias = globvar.super_weights[self.name]["bias"]
        else:
            self.super_weights = torch.tensor(gs_node.inputs[1].values, requires_grad=True)
            self.super_bias = torch.tensor(gs_node.inputs[2].values, requires_grad=True)

    def bind_weight(self):
        self.super_weights = globvar.super_weights[self.name]["weight"]
        self.super_bias = globvar.super_weights[self.name]["bias"]

    def update(self, kv_features):
        super().update(kv_features)
        inp_shape = self.features["in_shape"]
        axis = self.features["axis"]
        if axis < 0:
            axis = len(inp_shape) + axis

        dims = list(self.features["dims"])
        dim_cnt = 0
        # print(axis)
        # print(inp_shape)
        for i in range(axis, len(inp_shape)):
            dims[dim_cnt] = inp_shape[i]
            dim_cnt += 1
        self.features["dims"] = tuple(dims)

    def get_weights(self):
        weights = [self.super_weights]
        if self.features["has_bias"]:
            weights.append(self.super_bias)
        return weights

    def update_weights(self, weight_name, param):
        if weight_name == "weight":
            self.super_weights = param
        elif weight_name == "bias":
            self.super_bias = param
        else:
            raise NotImplementedError

    def update_grad(self):
        # weight
        grad = self.pointer.weight.grad
        c = grad.shape[-1]
        self.super_weights.grad[:c] += grad
        if self.features["has_bias"]:
            # bias
            grad = self.pointer.bias.grad
            c = grad.shape[-1]
            self.super_bias.grad[:c] += grad

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module
        in_channel_size = self.features["in_channel"]
        dims = self.features["dims"]
        self.module = nn.LayerNorm(dims,
                                   eps=self.features["epsilon"])
        # load params
        with torch.no_grad():
            if len(dims) == 1:
                self.module.weight = nn.Parameter(self.super_weights[:dims[0]])
                if self.features["has_bias"]:
                    self.module.bias = nn.Parameter(self.super_bias[:dims[0]])
            elif len(dims) == 2:
                self.module.weight = nn.Parameter(self.super_weights[:dims[0], :dims[1]])
                if self.features["has_bias"]:
                    self.module.bias = nn.Parameter(self.super_bias[:dims[0], :dims[1]])
        self.pointer = self.module
        return self.module

    @property
    def info(self):
        # dims, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"],
        )
        return info_tuple


class BiasAddOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.type = "BiasAdd"
        for inp in gs_node.inputs:
            if len(inp.inputs) == 0:
                # find the bias inp
                self.features['in_channel'] = inp.shape[-1]
                self.features['out_channel'] = inp.shape[-1]
                self.features['max_in_channel'] = inp.shape[-1]
                self.features['max_out_channel'] = inp.shape[-1]
                self.features["shape"] = inp.shape
                self.features["out_shape"] = inp.shape
                if globvar.sharing:
                    globvar.super_weights[self.name] = {}
                    globvar.super_weights[self.name]["bias"] = nn.Parameter(
                        torch.tensor(inp.values,
                                     device=self.device_str,
                                     requires_grad=True))
                    self.super_bias = globvar.super_weights[self.name]["bias"]
                else:
                    self.super_bias = torch.tensor(inp.values, requires_grad=True)

    def bind_weight(self):
        self.super_bias = globvar.super_weights[self.name]["bias"]

    def update(self, kv_features):
        for k, v in kv_features.items():
            if k in self.features and k in self.update_dimensions:
                self.features[k] = v
            else:
                raise KeyError
        shape = list(self.features["shape"])
        if len(shape) == 1:
            shape[0] = self.features["in_channel"]
            self.features["shape"] = tuple(shape)
            in_shape = list(self.features["in_shape"])
            in_shape[-1] = shape[0]
            self.features["out_shape"] = tuple(in_shape)
        elif len(shape) == 4 or len(shape) == 3:
            shape = self.features["in_shape"]
            self.features["shape"] = tuple(shape)
            self.features["out_shape"] = self.features["shape"]
        else:
            raise NotImplementedError(self.name)
        self.features["out_channel"] = self.features["in_channel"]

        # print(f"BiasAdd inp_shape: {self.features['in_shape']}; out_shape: {self.features['out_shape']}")

    def get_weights(self):
        weights = [self.super_bias]
        return weights

    def update_weights(self, weight_name, param):
        self.super_bias = param

    def update_grad(self):
        grad = self.pointer.bias_param.grad
        super_grad = torch.zeros(self.super_bias.shape)
        slices = tuple(slice(0, dim) for dim in grad.shape)
        self.super_bias.grad[slices] += grad

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module
        bias_shape = self.features["shape"]

        class BiasAddModule(nn.Module):
            def __init__(self, bias_shape):
                super(BiasAddModule, self).__init__()
                self.bias_param = nn.Parameter(torch.randn(bias_shape))

            def forward(self, x):
                # print(x.shape)
                # print(self.bias_param.shape)
                return x + self.bias_param

        self.module = BiasAddModule(bias_shape)

        # load params
        with torch.no_grad():
            if len(bias_shape) == 1:
                c = bias_shape[0]
                self.module.bias_param = nn.Parameter(self.super_bias[:c])
            elif len(bias_shape) == 3:
                c1 = bias_shape[0]
                c2 = bias_shape[1]
                c3 = bias_shape[2]
                self.module.bias_param = nn.Parameter(self.super_bias[:c1, :c2, :c3])
            elif len(bias_shape) == 4:
                c1 = bias_shape[0]
                c2 = bias_shape[1]
                c3 = bias_shape[2]
                c4 = bias_shape[3]
                self.module.bias_param = nn.Parameter(self.super_bias[:c1, :c2, :c3, :c4])
            else:
                raise NotImplementedError
        # print(f"biasadd: {self.module.bias_param.shape}")
        self.pointer = self.module
        return self.module


class LinearMatMulOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.type = "LinearMatMul"
        # print("Parsing LinearMatMul")
        for i, inp in enumerate(gs_node.inputs):
            if len(inp.inputs) == 0:
                # find the weight inp
                assert i == 1
                assert len(inp.shape) == 2
                self.features['in_channel'] = inp.shape[0]
                self.features['out_channel'] = inp.shape[1]
                self.features['max_in_channel'] = inp.shape[0]
                self.features['max_out_channel'] = inp.shape[1]
                if globvar.sharing:
                    globvar.super_weights[self.name] = {}
                    globvar.super_weights[self.name]["weight"] = nn.Parameter(
                        torch.tensor(np.transpose(inp.values, (1, 0)),
                                     device=self.device_str,
                                     requires_grad=True))
                    self.super_weights = globvar.super_weights[self.name]["weight"]
                else:
                    self.super_weights = torch.tensor(np.transpose(inp.values, (1, 0)), requires_grad=True)
        self.features["has_bias"] = False
        self.super_bias = None
        # print(self.features)

    def bind_weight(self):
        self.super_weights = globvar.super_weights[self.name]["weight"]
        # print(self.name)
        if self.features["has_bias"]:
            self.super_bias = globvar.super_weights[self.name]["bias"]

    def get_weights(self):
        weights = [self.super_weights]
        if self.features["has_bias"]:
            weights.append(self.super_bias)
        return weights

    def update_weights(self, weight_name, param):
        if weight_name == "weight":
            self.super_weights = param
        elif weight_name == "bias":
            self.super_bias = param
        else:
            raise NotImplementedError

    def update_grad(self):
        # weight
        grad = self.pointer.weight.grad
        cout, cin = grad.shape
        self.super_weights.grad[:cout, :cin] += grad
        if self.features["has_bias"]:
            # bias
            grad = self.pointer.bias.grad
            cout = grad.shape[0]
            self.super_bias.grad[:cout] += grad

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        # print(f"LinearMatMul: {self.features}")
        if len(inp_shape)==2:
            b, cin = inp_shape
            out_shape = (b, self.features['out_channel'])

            self.features['out_shape'] = out_shape # update out shape
            self.features['in_channel'] = cin # update in width
        elif len(inp_shape)==3:
            b, m, n = inp_shape
            out_shape = (b, m, self.features['out_channel'])
            self.features['out_shape'] = out_shape # update out shape
            self.features['in_channel'] = n # update in width
        elif len(inp_shape)==4:
            b, hw, m, n = inp_shape
            out_shape = (b, hw, m, self.features['out_channel'])
            self.features['out_shape'] = out_shape # update out shape
            self.features['in_channel'] = n # update in width
        else:
            raise NotImplementedError
        # print(f"updated linearmatmul: {self.features}")

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module
        in_channel_size = self.features["in_channel"]
        out_channel_size = self.features["out_channel"]
        has_bias = self.features["has_bias"]

        self.module = nn.Linear(in_channel_size,
                                out_channel_size,
                                bias=has_bias)

        # load params
        # print(f"Update gemm to {out_channel_size}x{in_channel_size}")
        with torch.no_grad():
            self.module.weight = nn.Parameter(self.super_weights[:out_channel_size,:in_channel_size])
            if has_bias:
                self.module.bias = nn.Parameter(self.super_bias[:out_channel_size])
        self.pointer = self.module
        return self.module

    def count_flops_params(self):
        cin = self.features['in_channel']
        cout = self.features['out_channel']
        in_shape = self.features['in_shape']

        if len(in_shape)==2:
            batch = in_shape[0]
        elif len(in_shape)==3:
            batch = in_shape[0]*in_shape[1]
        elif len(in_shape)==4:
            batch = in_shape[0]*in_shape[1]*in_shape[2]

        params = cin * cout
        flops = 2 * params * batch
        return flops, params

    @property
    def info(self):
        # has_bias, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["has_bias"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"],
        )
        return info_tuple


class MatMulOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.type = "MatMul"

    def update(self, kv_features: dict) -> None:
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        assert len(self.features["in_shape"]) == 2
        # print(self.features["in_shape"])
        inp1 = self.features["in_shape"][0]
        inp2 = self.features["in_shape"][1]

        num_dims = len(inp1)

        assert inp1[-1] == inp2[-2]
        self.features["in_channel"] = inp1[-2]
        self.features["out_channel"] = inp2[-1]

        out_shape = []

        if num_dims > 2:
            # print(inp1)
            # print(inp2)
            for dim in range(num_dims-2):
                assert inp1[dim] == inp2[dim]
                out_shape.append(inp1[dim])

        out_shape.append(self.features["in_channel"])
        out_shape.append(self.features["out_channel"])
        self.features["out_shape"] = tuple(out_shape)

    def count_flops_params(self):
        inp1 = self.features["in_shape"][0]
        inp2 = self.features["in_shape"][1]
        num_dims = len(inp1)

        batch = 1
        if num_dims > 2:
            for dim in range(num_dims-2):
                batch *= inp1[dim]

        params = 0
        # flops = 2 * inp1[-2] * inp2[-2] * inp2[-1] * batch
        flops = 0
        return flops, params

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module

        class MatMulModule(nn.Module):
            def __init__(self):
                super(MatMulModule, self).__init__()

            def forward(self, x1, x2):
                y = x1 @ x2
                return y
        self.module = MatMulModule()
        self.pointer = self.module
        return self.module


class ScaleMulOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.type = "ScaleMul"
        for inp in gs_node.inputs:
            if len(inp.inputs) == 0 and inp.shape[0] > 1:
                # find the scale inp
                self.features['in_channel'] = inp.shape[0]
                self.features['out_channel'] = inp.shape[0]
                self.features['max_in_channel'] = inp.shape[0]
                self.features['max_out_channel'] = inp.shape[0]
                self.features["scale_shape"] = inp.shape
                if globvar.sharing:
                    globvar.super_weights[self.name] = {}
                    globvar.super_weights[self.name]["scale"] = nn.Parameter(
                        torch.tensor(inp.values,
                                     device=self.device_str,
                                     requires_grad=True))
                    self.super_scale = globvar.super_weights[self.name]["scale"]
                else:
                    self.super_scale = torch.tensor(inp.values, requires_grad=True)
        assert "scale_shape" in self.features

    def bind_weight(self):
        self.super_scale = globvar.super_weights[self.name]["scale"]

    def get_weights(self):
        weights = [self.super_scale]
        return weights

    def update_weights(self, weight_name, param):
        self.super_scale = param

    def update_grad(self, weight_name, grad):
        c = grad.shape[0]
        self.super_scale.grad[:c] += grad
        # super_grad = torch.zeros(self.super_scale.shape)
        # super_grad[:c] = grad
        # if self.super_scale.grad:
        #     self.super_scale.grad += super_grad
        # else:
        #     self.super_scale.grad = super_grad

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        out_shape = list(self.features["in_shape"])
        out_channel = self.features["out_channel"]
        scale_shape = list(self.features["scale_shape"])
        scale_shape[0] = out_channel
        out_shape[1] = out_channel
        self.features["out_shape"] = tuple(out_shape)
        self.features["scale_shape"] = tuple(scale_shape)

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module
        scale_shape = self.features["scale_shape"]
        in_channel_size = self.features["in_channel"]
        class ScaleMulModule(nn.Module):
            def __init__(self, scale_shape):
                super(ScaleMulModule, self).__init__()
                self.scale_param = nn.Parameter(torch.randn(scale_shape))

            def forward(self, x):
                return x * self.scale_param

        self.module = ScaleMulModule(scale_shape)

        # load params
        with torch.no_grad():
            self.module.scale_param = nn.Parameter(self.super_scale[:in_channel_size, :, :])
        self.pointer = self.module
        return self.module


class MulOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.type = "Mul"
        for inp in gs_node.inputs:
            if len(inp.inputs) == 0:
                # list(inp.shape): plain list, not protobuf RepeatedScalarContainer (unpicklable)
                self.features["in_shape"] = tuple([self.features["in_shape"], list(inp.shape)])

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                if k is not "in_shape":
                    self.features[k] = v
                else:
                    # print(self.features["in_shape"])
                    # print(v)
                    if len(v) == self.features["in_shape"] or len(self.features["in_shape"]) == 0:
                        self.features[k] = v
                    else:
                        if isinstance(self.features[k], tuple):
                            self.features[k] = list(self.features[k])
                            self.features[k][0] = v
                            self.features[k] = tuple(self.features[k])
                        else:
                            self.features[k][0] = v
            else:
                raise KeyError

        assert len(self.features["in_shape"]) == 2 or len(self.features["in_shape"]) == 1
        # print(self.features)
        inp1 = self.features["in_shape"][0]
        inp2 = self.features["in_shape"][1]
        len_inp1 = len(inp1)
        len_inp2 = len(inp2)
        out_shape = []
        if len_inp1 > len_inp2:
            out_shape = inp1
        else:
            out_shape = inp2
        # print(f"mulop: {out_shape}")

        self.features["out_shape"] = tuple(out_shape)


class MaxPoolOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['kernel_size'] = gs_node.attrs['kernel_shape'][0]
        self.features['stride'] = gs_node.attrs['strides'][0]

    @property
    def info(self):
        # kernel_size, stride, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["kernel_size"],
            self.features["stride"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        return info_tuple

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        b, c, h, w = inp_shape

        stride = self.features['stride']

        if stride == 1:
            h = w = h
        elif stride == 2:
            h = w = math.ceil(h / 2)
        elif stride == 4:
            h = w = math.ceil(h / 4)

        out_shape = (b, c, h, w)

        self.features['out_shape'] = out_shape # update out shape
        self.features['in_channel'] = c # update in width
        self.features['out_channel'] = c # update out width


class AveragePoolOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['kernel_size'] = gs_node.attrs['kernel_shape'][0]
        self.features['stride'] = gs_node.attrs['strides'][0]

    @property
    def info(self):
        # kernel_size, stride, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["kernel_size"],
            self.features["stride"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        return info_tuple

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        b, c, h, w = inp_shape

        stride = self.features['stride']

        if stride == 1:
            h = w = h
        elif stride == 2:
            h = w = math.ceil(h / 2)
        elif stride == 4:
            h = w = math.ceil(h / 4)

        out_shape = (b, c, h, w)

        self.features['out_shape'] = out_shape # update out shape
        self.features['in_channel'] = c # update in width
        self.features['out_channel'] = c # update out width


class GlobalAveragePoolOp(SingleOp):

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        b, c, h, w = inp_shape

        out_shape = (b, c, 1, 1)

        self.features['out_shape'] = out_shape # update out shape
        self.features['in_channel'] = c # update in width
        self.features['out_channel'] = c # update out width


class ReduceMeanOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['axes'] = gs_node.attrs['axes']
        self.features['keepdims'] = True if gs_node.attrs['keepdims']==1 else False

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        new_shape = []
        for i, ax in enumerate(inp_shape):
            if i in self.features['axes']:
                if self.features['keepdims']:
                    new_shape.append(1)
            else:
                new_shape.append(ax)

        out_shape = tuple(new_shape)
        self.features['out_shape'] = out_shape # update out shape
        self.features['in_channel'] = inp_shape[1] # update in width
        self.features['out_channel'] = inp_shape[1] # update out width


class FlattenOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['axis'] = gs_node.attrs['axis']

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        dim_left = 1
        dim_right = 1
        for i, ax in enumerate(inp_shape):
            if i < self.features["axis"]:
                dim_left *= ax
            else:
                dim_right *= ax

        out_shape = (dim_left, dim_right)
        self.features['out_shape'] = out_shape # update out shape
        self.features['in_channel'] = inp_shape[1] # update in width
        self.features['out_channel'] = inp_shape[1] # update out width


class ReshapeOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['shape'] = tuple(gs_node.inputs[1].values)
        self.features['in_channel'] = 0
        self.features['out_channel'] = 0
        self.update_dimensions.append("shape")

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError(f"{k} not in features")

        inp_shape = self.features['in_shape']

        num_inp_shape = len(inp_shape)
        num_shape = len(self.features["shape"])
        out_shape = list(self.features["shape"])
        # print(f"Out shape:{out_shape}")
        # find any dim
        any_dim = None
        for i in range(len(list(self.features["shape"]))):
            if out_shape[i] == -1:
                any_dim = i
                continue
        if num_inp_shape == 4 and num_shape == 2:
            out_shape[0] = inp_shape[0]
            out_shape[1] = inp_shape[1]
        else:
            assert num_inp_shape >= 3 and num_shape >= 3
            if num_inp_shape > num_shape:
                # 4 -> 3
                # TODO: support any dim
                # print(any_dim)

                if any_dim == 2:
                    # compress
                    out_shape[0] = inp_shape[0]
                    out_shape[1] = inp_shape[1]
                    out_shape[2] = int(inp_shape[2] * inp_shape[3])
                elif any_dim == 0:
                    # print(inp_shape)
                    # print(out_shape)
                    # print(inp_shape[2] * inp_shape[3])
                    if inp_shape[2] * inp_shape[3] == out_shape[2]:
                        # bs, squeeze the last two into one
                        # print(f"bs: {inp_shape[0]}")
                        out_shape[0] = inp_shape[0]
                        out_shape[1] = inp_shape[1]
                    elif inp_shape[1] == out_shape[1] and inp_shape[2] == inp_shape[3]:
                        out_shape[0] = inp_shape[0]
                        out_shape[2] = inp_shape[2] * inp_shape[3]
                    else:
                        out_shape[0] = int(inp_shape[0] * inp_shape[1])
                        out_shape[1] = inp_shape[2]
                        out_shape[2] = inp_shape[3]
                elif any_dim == -1:
                    out_shape[0] = inp_shape[0]
                    out_shape[1] = inp_shape[1]
                    out_shape[2] = inp_shape[2] * inp_shape[3]
                elif any_dim is None:
                    out_shape[0] = inp_shape[0]
                    out_shape[1] = inp_shape[1]
                    out_shape[2] = inp_shape[2] * inp_shape[3]
                else:
                    # print(any_dim)
                    # print(self.name)
                    raise NotImplementedError
            elif num_inp_shape < num_shape:
                # 3 -> 4
                if any_dim == 2:
                    # expand
                    if out_shape[2] == out_shape[3]:
                        out_shape[0] = inp_shape[0]
                        out_shape[1] = inp_shape[1]
                        out_shape[2] = out_shape[3] = int(inp_shape[2] ** 0.5)
                    else:
                        # keep hidden dim
                        out_shape[0] = inp_shape[0]
                        out_shape[1] = inp_shape[1]
                        out_shape[3] = int(inp_shape[2] // out_shape[2])
                elif any_dim == 0:
                    # expand
                    if out_shape[2] * out_shape[3] == inp_shape[2]:
                    # bs, extend the last dim into two
                        out_shape[0] = inp_shape[0]
                        out_shape[1] = inp_shape[1]
                    elif out_shape[0] == out_shape[1]:
                        out_shape[0] = out_shape[1] = int(inp_shape[0] ** 0.5)
                        out_shape[2] = inp_shape[1]
                        out_shape[3] = inp_shape[2]
                    elif out_shape[2] == out_shape[3] and inp_shape[1] == out_shape[1]:
                        out_shape[0] = inp_shape[0]
                        out_shape[2] = out_shape[3] = int(inp_shape[2] ** 0.5)
                    else:
                        # keep hidden dim
                        out_shape[0] = max(1, int(inp_shape[0] // out_shape[1]))
                        out_shape[2] = out_shape[1]
                        out_shape[3] = out_shape[2]
                elif any_dim == -1:
                    out_shape[0] = inp_shape[0]
                    out_shape[1] = inp_shape[1]
                    out_shape[3] = int(inp_shape[2] // out_shape[2])
                    # pass
                else:
                    if any_dim == (len(out_shape)-1):
                        out_shape[0] = inp_shape[0]
                        out_shape[1] = inp_shape[1]
                        out_shape[3] = int(inp_shape[2] // out_shape[2])
                        # pass
                    else:
                        raise NotImplementedError
            else:
                # bs update
                out_shape[0] = inp_shape[0]
        
        # print(f"Reshape: {inp_shape} -> {out_shape}")
        # print(inp_shape)
        # print(out_shape)
        # print(any_dim)
        self.features["shape"] = tuple(out_shape)
        # process any dim into real value
        # if any_dim:
        #     dim_sum = 1
        #     for dim in inp_shape:
        #         dim_sum *= dim
        #     dim_sum_tmp = 1
        #     for dim in out_shape:
        #         if dim != -1:
        #             dim_sum_tmp *= dim
        #     print(dim_sum)
        #     print(dim_sum_tmp)
        #     out_shape[any_dim] = dim_sum // dim_sum_tmp
        # print(f"any_dim: {any_dim}")
        # print(f"In: {inp_shape}")
        # print(f"Shape: {self.features['shape']}")
        # print(f"Reshape: {out_shape}")
            
        self.features['out_shape'] = tuple(out_shape) # update out shape
        self.features['out_channel'] = self.features["in_channel"]
        if any_dim != -1 and any_dim is not None:
            out_shape[any_dim] = -1
            self.features["shape"] = tuple(out_shape)

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module

        self.module = ReshapeModule(self.features["shape"])
        self.pointer = self.module
        return self.module

    @property
    def info(self):
        # shape, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["shape"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        return info_tuple


class TransposeOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['perm'] = tuple(gs_node.attrs['perm'])
        self.features['in_channel'] = 0
        self.features['out_channel'] = 0

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError

        inp_shape = self.features['in_shape']
        perm = self.features['perm']

        assert len(inp_shape)==len(perm)
        out_shape = []

        for dim_id in range(len(inp_shape)):
            out_shape.append(inp_shape[perm[dim_id]])

        self.features["out_channel"] = self.features["in_channel"]

        self.features['out_shape'] = tuple(out_shape) # update out shape
        # print(f"Transpose output: {out_shape}")

    @property
    def info(self):
        # perm, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["perm"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        return info_tuple


class SliceOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['starts'] = int(gs_node.inputs[1].values)
        self.features['ends'] = int(gs_node.inputs[2].values)
        self.features['axes'] = int(gs_node.inputs[3].values)


class SplitModule(nn.Module):
    def __init__(self, split_size, axis):
        super(SplitModule, self).__init__()
        self.split_size = split_size
        self.axis = axis

    def forward(self, x):
        # print(f"Split input: {x.shape}")
        return torch.split(x, self.split_size, self.axis)


class SplitOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.type = "Split"
        self.features["axis"] = gs_node.attrs["axis"]
        self.features['split'] = tuple(gs_node.inputs[1].values)
        self.features["super_split"] = tuple(gs_node.inputs[1].values)
        self.update_dimensions.append("split")

    # def _pre_build(self):
    #     pass

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            self.pointer = op_module
            return op_module
        # print("Build split module")
        self.module = SplitModule(self.features["split"],
                                  self.features["axis"])
        self.pointer = self.module
        return self.module

    @property
    def info(self):
        # split, axis, in_shape
        info_tuple = (
            self.features["split"],
            self.features["axis"],
            self.features["in_shape"],
        )
        return info_tuple


class ConcatOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features['axis'] = gs_node.attrs['axis']
        extra_dims = 0
        for inp in gs_node.inputs:
            if not isinstance(inp, gs.ir.tensor.Variable):
                extra_dims += torch.tensor(inp.values).shape[self.features['axis']]
        # print(extra_dims)
        self.features["extra_dims"] = extra_dims
    
    def update(self, kv_features: dict) -> None:
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError
        self.features["out_shape"] = self.features["in_shape"]
        out_shape = list(self.features["out_shape"])
        out_shape[self.features["axis"]] += self.features["extra_dims"]
        self.features["out_shape"] = tuple(out_shape)
        # print(f"Concat {self.features['out_shape']}")
    
    @property
    def info(self):
        # axis, extra_dims, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["axis"],
            self.features["extra_dims"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        return info_tuple


class ShapeOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
    
    def update(self, kv_features: dict) -> None:
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError
        self.features["out_shape"] = len(self.features["in_shape"])


class GatherOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features["axis"] = gs_node.attrs["axis"]
        self.features["indices"] = int(gs_node.inputs[1].values)
    
    def update(self, kv_features: dict) -> None:
        for k, v in kv_features.items():
            if k in self.features:
                self.features[k] = v
            else:
                raise KeyError
        
        if self.features["indices"] == 0 and self.features["axis"] == 1:
            in_shape = self.features["in_shape"]
            new_shape = (in_shape[0], in_shape[2])
            self.features["out_shape"] = new_shape
        else:
            raise NotImplementedError


class SoftmaxOp(SingleOp):
    def __init__(self, gs_node):
        super().__init__(gs_node)
        self.features["axis"] = gs_node.attrs["axis"]

    def build(self, cache=None, block_id=None):
        if cache:
            feature = f"{block_id}-{str(self.info)}"
            op_module = cache[self.type][feature]
            return op_module
        return nn.Softmax(dim=self.features["axis"])

    @property
    def info(self):
        # axis, in_shape, out_shape, in_channel, out_channel
        info_tuple = (
            self.features["axis"],
            self.features["in_shape"],
            self.features["out_shape"],
            self.features["in_channel"],
            self.features["out_channel"]
        )
        return info_tuple
