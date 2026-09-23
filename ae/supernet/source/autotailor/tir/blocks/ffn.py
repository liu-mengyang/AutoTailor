from .residual import ResidualBlock
from .single_op import *
from .tailor_utils import make_divisible, closest


class FFNBlock(ResidualBlock):
    def __init__(self, block, supernet_cfg_dict, divisor=8):
        super().__init__(block.start_node,
                         block.last_node,
                         block.main_path,
                         block.residual_path)
        self.type = "FFNBlock"
        
        if "TransformerBlock" in supernet_cfg_dict["var"]["block_vars"]:
            all_expand_ratios = supernet_cfg_dict["var"]["block_vars"]["TransformerBlock"]["expand_ratio"]
        else:
            all_expand_ratios = None
        
        expand_ratio_list = []
        if isinstance(all_expand_ratios[0], list):
            for exp_lst in all_expand_ratios:
                for exp in exp_lst:
                    if exp not in expand_ratio_list:
                        expand_ratio_list.append(exp)
            expand_ratio_list.sort()
        else:
            expand_ratio_list = all_expand_ratios
        
        # extract information
        ln = None
        first_linear = None
        last_linear = None
        for op in self.main_path:
            if isinstance(op, LNOp) and ln is None:
                ln = op
            elif isinstance(op, LinearMatMulOp):
                if first_linear is None:
                    first_linear = op
                elif last_linear is None:
                    last_linear = op
                else:
                    raise NotImplementedError
        
        # print("Parsing FFN")
        # print(first_linear)
        expand_base_width = ln.features['in_channel']
        
        self.expand_ratio_list = expand_ratio_list
        self.features["trans_type"] = 'active'
        self.features["max_in_channel"] = ln.features["in_channel"]
        self.features["max_out_channel"] = ln.features["in_channel"]
        self.features["in_channel"] = ln.features["in_channel"]
        self.features["out_channel"] = ln.features["in_channel"]
        self.features["middle_width"] = first_linear.features["out_channel"]
        self.features['max_expand_ratio'] = closest(self.features['middle_width'] / expand_base_width, expand_ratio_list)
        self.features['expand_ratio'] = self.features['max_expand_ratio']
        self.features["divisor"] = divisor
        # print(self.features)
        
        
        self.dynamic_dimensions.append("expand_ratio")
        
    def transform(self, kv_features: dict) -> None:
        super().transform(kv_features)
        
        # print("Transfroming FFN")
        # print(self.features)
        if "expand_ratio" in kv_features.keys():
            cin = self.features["in_channel"]
            cout = self.features["out_channel"]
            expand_ratio = self.features["expand_ratio"]
            
            expand_base_width = cin
            middle_width = make_divisible(expand_base_width * expand_ratio,
                                          self.features["divisor"])
            # print(middle_width)
            self.features["middle_width"] = middle_width
            
            # update
            cur_width = cin
            cur_shape = self.features["in_shape"]
            
            ln = None
            first_linear = None
            last_linear = None
            for op in self.main_path:
                if isinstance(op, LNOp) and ln is None:
                    ln = op
                    op.update({"in_channel": cur_width,
                               "in_shape": cur_shape})
                elif isinstance(op, LinearMatMulOp):
                    if first_linear is None:
                        first_linear = op
                        op.update({"in_channel": cur_width,
                                   "in_shape": cur_shape,
                                   "out_channel": middle_width})
                    elif last_linear is None:
                        last_linear = op
                        op.update({"in_channel": cur_width,
                               "in_shape": cur_shape})
                    else:
                        raise NotImplementedError
                else:
                    op.update({"in_channel": cur_width,
                               "in_shape": cur_shape})
                cur_shape = op.features["out_shape"]
                cur_width = op.features["out_channel"]
            # print(f"Update ffn width: {middle_width}")
            # print(self.features)
