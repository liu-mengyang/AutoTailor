from .block import Block
from .single_op import *


class MBBlock(Block):
    def __init__(self, flow):
        """Build mobile net block

        Args:
            flow (TIR Flow): the list of nodes
        """
        super().__init__()
        self.type = 'MBBlock'
        self.flow = flow

        # extract information
        dw_conv = None
        point_conv = None
        for i, op in enumerate(self.flow):
            if isinstance(op, DepthConvOp):
                dw_conv = op
            elif isinstance(op, ConvOp) and op.features["kernel_size"] == 1:
                point_conv = op
            op.id = i
        assert dw_conv and point_conv

        self.features = {
            'activated': True,
            'trans_type': 'active',
            'in_shape': (),
            'out_shape': (),
            'max_in_channel': dw_conv.features['in_channel'],
            'max_out_channel': point_conv.features['out_channel'],
            'max_kernel_size': dw_conv.features['kernel_size'],
            'in_channel': dw_conv.features['in_channel'],
            'out_channel': point_conv.features['out_channel'],
            'kernel_size': dw_conv.features['kernel_size'],
        }

        self.dynamic_dimensions.append('kernel_size')

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Only in shape and width can be updated.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        super().update(kv_features)
        point_conv = None
        if "out_channel" in kv_features:
            for op in self.flow:
                if isinstance(op, ConvOp):
                    point_conv = op
                    op.update({"out_channel": self.features["out_channel"]})

        inp_shape = self.features['in_shape']

        tensor_shape = inp_shape
        cur_width = self.features['in_channel']
        for node in self.flow:
            # update each node in flow
            node.update({'in_channel': cur_width})
            cur_width = node.features['out_channel']
            node.update({'in_shape': tensor_shape})
            tensor_shape = node.features['out_shape']

        out_shape = self.flow[-1].features['out_shape']

        self.features['in_shape'] = inp_shape
        self.features['out_shape'] = out_shape

    def build(self, cache=None, block_id=None) -> nn.Module:
        """Build torch module for bottleneck block.

        Build operator one by one in each path.

        Retruns:
          Torch module of bottleneck.
        """
        op_modules = []
        for op in self.flow:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        self.module = nn.Sequential(*op_modules)
        return self.module

    def bind_weight(self):
        for op in self.flow:
            op.bind_weight()

    def transform(self, kv_features: dict) -> None:
        """Transform features of the block.

        MB block can transform kernel size

        Args:
          kv_features: the dictionary of to transform features.

        Raises:
          KeyError: the error occured in updating non exist feature or not
                    not support to transform this dimension.
        """
        super().transform(kv_features)

        dw_conv = None
        for op in self.flow:
            # update conv op in flow
            if isinstance(op, DepthConvOp):
                dw_conv = op
                op.update({"kernel_size": self.features["kernel_size"]})
        assert dw_conv

    def count_flops_params(self):
        params = 0
        flops = 0
        for op in self.flow:
            op_flops, op_params = op.count_flops_params()
            op_flops += op_flops
            op_params += op_params
        return flops, params

    def get_ops(self):
        op_lst = []
        for op in self.flow:
            if not isinstance(op, SingleOp):
                # nested block
                ops = op.get_ops()
                for subop in ops:
                    subop.id = str(op.id)+'-'+str(subop.id)
                    op_lst.append(subop)
            else:
                op_lst.append(op)
        return op_lst

    def update_grad(self):
        for op in self.flow:
            op.update_grad()

    @property
    def info_dict(self):
        ret_dict = super().info_dict

        ret_dict['flow'] = []

        for op in self.flow:
            ret_dict['flow'].append(op.info_dict)
        return ret_dict

    @property
    def info(self):
        info_list = []
        for op in self.flow:
            info_list.append(op.info)
        info_tuple = tuple(info_list)
        return info_tuple
