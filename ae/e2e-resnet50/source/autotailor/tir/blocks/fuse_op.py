from .block import Block
from .single_op import *


class FuseOp(SingleOp):
    def __init__(self, flow):
        self.flow = flow
        self.type = 'FuseOp'

    def count_flops_params(self):
        params = 0
        flops = 0
        for op in self.flow:
            op_flops, op_params = op.count_flops_params()
            op_flops += op_flops
            op_params += op_params
        return flops, params
    
    def build(self, cache=None, block_id=None) -> nn.Module:
        op_modules = []
        for op in self.flow:
            print(op.info)
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        self.module = nn.Sequential(*op_modules)
        return self.module

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
            info_list.append(f"{op.type}-{op.info}")
        info_tuple = tuple(info_list)
        return info_tuple
