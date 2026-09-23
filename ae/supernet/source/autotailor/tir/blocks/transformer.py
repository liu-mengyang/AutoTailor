import torch.nn as nn

from .residual import ResidualBlock
from .ffn import FFNBlock
from .attention import AttentionBlock
from .single_op import *


class TransformerBlock(Block):
    def __init__(self, attn_block, ffn_block):
        super().__init__()
        self.type = "TransformerBlock"
        self.flow = [attn_block, ffn_block]

        # extract information
        self.features = {
            "activated": True,
            "trans_type": "active",
            "in_shape": attn_block.features["in_shape"],
            "out_shape": ffn_block.features["out_shape"],
            "max_in_channel": attn_block.features["in_channel"],
            "max_out_channel": ffn_block.features["out_channel"],
            "in_channel": attn_block.features["in_channel"],
            "out_channel": ffn_block.features["out_channel"],
            "v_scale": attn_block.features["v_scale"],
            "max_v_scale": attn_block.features["max_v_scale"],
            "expand_ratio": ffn_block.features["expand_ratio"],
            "max_expand_ratio": ffn_block.features["max_expand_ratio"]
        }

        self.dynamic_dimensions.append("v_scale")
        self.dynamic_dimensions.append("expand_ratio")

        cnt = 0
        for op in self.flow:
            op.id = cnt
            cnt += 1

    def update(self, kv_features: dict) -> None:
        for k, v in kv_features.items():
            if k in self.features and k in self.update_dimensions:
                self.features[k] = v
            else:
                raise KeyError
        cur_width = self.features["in_channel"]
        cur_shape = self.features["in_shape"]

        for block in self.flow:
            block.update({"in_channel": cur_width,
                          "in_shape": cur_shape})
            cur_width = block.features["out_channel"]
            cur_shape = block.features["out_shape"]

        self.features["out_channel"] = cur_width
        self.features["out_shape"] = cur_shape

    def transform(self, kv_features: dict) -> None:
        super().transform(kv_features)

        if "expand_ratio" in kv_features.keys():
            cur_width = self.features["in_channel"]
            cur_shape = self.features["in_shape"]
            for block in self.flow:
                if isinstance(block, FFNBlock):
                    block.transform({"expand_ratio": kv_features["expand_ratio"]})
                cur_width = block.features["out_channel"]
                cur_shape = block.features["out_shape"]
            self.features["out_channel"] = cur_width
            self.features["out_shape"] = cur_shape
        elif "v_scale" in kv_features.keys():
            cur_width = self.features["in_channel"]
            cur_shape = self.features["in_shape"]
            for block in self.flow:
                if isinstance(block, AttentionBlock):
                    block.transform({"v_scale": kv_features["v_scale"]})
                cur_width = block.features["out_channel"]
                cur_shape = block.features["out_shape"]
            self.features["out_channel"] = cur_width
            self.features["out_shape"] = cur_shape

    def build(self, cache=None, block_id=None):
        op_modules = []
        for op in self.flow:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        self.module = nn.Sequential(*op_modules)
        return self.module

    def bind_weight(self):
        for op in self.flow:
            op.bind_weight()

    def get_ops(self):
        op_lst = []
        for op in self.flow:
            if not isinstance(op, SingleOp):
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

    def count_flops_params(self):
        params = 0
        flops = 0
        for op in self.flow:
            op_flops, op_params = op.count_flops_params()
            flops += op_flops
            params += op_params
        return flops, params
