from loguru import logger

from .block import Block
from .convresidual import ConvResidualBlock
from .single_op import *
from .tailor_utils import make_divisible, closest


class CNResidualBlock(ConvResidualBlock):
    def __init__(self,
                 start_node,
                 last_node,
                 first_path,
                 second_path,
                 supernet_cfg_dict,
                 divisor=8):
        """Build residual block

        Args:
            start_node (TIR Node): the start node of the block
            last_node (TIR Node): the second node of the block
            first_path (list): the first path of nodes
            second_path (list): the second path of nodes
            supernet_cfg_dict: the dict of supernet configuration
        """
        super(CNResidualBlock, self).__init__(start_node,
                         last_node,
                         first_path,
                         second_path)
        self.type = 'CNResidualBlock'
        
        expand_base_on = supernet_cfg_dict["arch"]["CNResidualBlock"]["expand_base"]
        assert expand_base_on == "in"
        
        if "CNResidualBlock" in supernet_cfg_dict["var"]["block_vars"]:
            all_expand_ratios = supernet_cfg_dict["var"]["block_vars"]["CNResidualBlock"]["expand_ratio"]
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
        dw_conv = None
        first_matmul = None
        first_add = None
        last_matmul = None
        last_add = None
        mul_scaler = None
        for op in self.main_path:
            if isinstance(op, ConvOp):
                if dw_conv is None:
                    dw_conv = op
                else:
                    raise NotImplementedError
            elif isinstance(op, LinearMatMulOp):
                if first_matmul is None:
                    first_matmul = op
                elif last_matmul is None:
                    last_matmul = op
                else:
                    raise NotImplementedError
            elif isinstance(op, BiasAddOp):
                if first_add is None:
                    first_add = op
                elif last_add is None:
                    last_add = op
                else:
                    raise NotImplementedError
            elif isinstance(op, ScaleMulOp):
                if mul_scaler is None:
                    mul_scaler = op
                else:
                    raise NotImplementedError

        expand_base_width = dw_conv.features['in_channel']
        
        self.expand_ratio_list = expand_ratio_list
        self.features['expand_base_on'] = expand_base_on
        self.features['middle_width'] = first_matmul.features['out_channel']
        self.features['kernel_size'] = dw_conv.features['kernel_size']
        self.features['max_kernel_size'] = dw_conv.features['kernel_size']
        self.features['max_expand_ratio'] = closest(self.features['middle_width'] / expand_base_width, expand_ratio_list)
        self.features['expand_ratio'] = self.features['max_expand_ratio']
        self.features["divisor"] = divisor
        
        self.dynamic_dimensions.append('expand_ratio')
    
    def reorganize_weight(self, cfgs):
        dw_conv = None
        first_matmul = None
        first_add = None
        last_matmul = None
        last_add = None
        mul_scaler = None
        for op in self.main_path:
            if isinstance(op, ConvOp):
                if dw_conv is None:
                    dw_conv = op
                else:
                    raise NotImplementedError
            elif isinstance(op, LinearMatMulOp):
                if first_matmul is None:
                    first_matmul = op
                elif last_matmul is None:
                    last_matmul = op
                else:
                    raise NotImplementedError
            elif isinstance(op, BiasAddOp):
                if first_add is None:
                    first_add = op
                elif last_add is None:
                    last_add = op
                else:
                    raise NotImplementedError
            elif isinstance(op, ScaleMulOp):
                if mul_scaler is None:
                    mul_scaler = op
                else:
                    raise NotImplementedError
        
        if "expand_ratio" in cfgs:
            # reorganize weights for dynamic expand ratio in progressive training
            importance = torch.sum(
                torch.abs(last_matmul.super_weights.data), dim=(0)
            )
            
            width_list = []
            
            for expand_ratio in self.expand_ratio_list:
                base_width = self.features["max_in_channel"] if self.features["expand_base_on"] == "in" else self.features["max_out_channel"]
                width = make_divisible(base_width * expand_ratio,
                                      self.features["divisor"])
                width_list.append(width)
                
            mini_ratio_index = min(cfgs["expand_ratio"])
            mini_width = width_list[mini_ratio_index]
            
            right = len(importance)
            num_stages = len(width_list) - mini_ratio_index - 1
            base = -num_stages * 1e5
            for i in range(num_stages):
                left = width_list[len(width_list)-2-i]
                importance[left:right] += base
                logger.info(f"Reorganize weight of {self.type} from {left} to {right} with {base} offset")
                base += 1e5
                right = left
            
            sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
            
            # reorganize the first matmul in out dim
            logger.info("Reorganize first matmul")
            first_matmul.super_weights.data = torch.index_select(
                first_matmul.super_weights.data, 0, sorted_idx
            )
            
            first_add.super_bias.data = torch.index_select(
                first_add.super_bias.data, 0, sorted_idx
            )
            
            # reorganize the last matmul in in dim
            logger.info("Reorganize last matmul")
            last_matmul.super_weights.data = torch.index_select(
                last_matmul.super_weights.data, 1, sorted_idx
            )
                
    def update(self, kv_features: dict) -> None:
        """Update features of the block.
        
        Only in shape and width can be updated.
        
        Args:
          kv_features: the dictionary of to update features.
        
        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        for k, v in kv_features.items():
            if k in self.features and k in self.update_dimensions:
                self.features[k] = v
            else:
                raise KeyError
        inp_shape = self.features["in_shape"]
        
        cin = self.features["in_channel"]
        cout = self.features["out_channel"]
        
        expand_ratio = self.features["expand_ratio"]
        base_width = cin if self.features["expand_base_on"] == "in" else cout
        middle_width = make_divisible(base_width * expand_ratio,
                                      self.features["divisor"])
        self.features["middle_width"] = middle_width
        
        cur_width = cin
        dw_conv = None
        first_matmul = None
        first_add = None
        last_matmul = None
        last_add = None
        mul_scaler = None
        cur_shape = inp_shape
        for node in self.main_path:
            # update each node in flow
            if isinstance(node, ConvOp):
                if dw_conv is None:
                    dw_conv = node
                    node.update({"in_channel": cur_width,
                                "out_channel": cur_width,
                                "in_shape": cur_shape})
                else:
                    raise NotImplementedError
            elif isinstance(node, LinearMatMulOp):
                if first_matmul is None:
                    first_matmul = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": middle_width,
                                 "in_shape": cur_shape})
                elif last_matmul is None:
                    last_matmul = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": cout,
                                 "in_shape": cur_shape})
                else:
                    raise NotImplementedError
            elif isinstance(node, ScaleMulOp):
                if mul_scaler is None:
                    mul_scaler = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": cout,
                                 "in_shape": cur_shape})
            else:
                node.update({"in_channel": cur_width,
                             "in_shape": cur_shape})
            cur_width = node.features["out_channel"]
            cur_shape = node.features["out_shape"]
        
        cur_shape = inp_shape
        cur_width = cin
        for node in self.residual_path:
            # update each node in residual path
            if isinstance(node, ConvOp):
                node.update({"in_shape": cur_shape,
                             "in_channel": cur_width,
                             "out_channel": cout})
            else:
                node.update({'in_shape': cur_shape,
                            'in_channel': cur_width})
            cur_shape = node.features['out_shape']
            cur_width = node.features['out_channel']
        
        self.features['out_channel'] = cur_width
        self.features['out_shape'] = cur_shape
    
    def transform(self, kv_features: dict) -> None:
        """Transform features of the block.
        
        BottleneckResidual block can transform kernel size and expand ratio 
        
        Args:
          kv_features: the dictionary of to transform features.
        
        Raises:
          KeyError: the error occured in updating non exist feature or not
                    not support to transform this dimension.
        """
        super().transform(kv_features)
        
        if "expand_ratio" in kv_features.keys():
            cin = self.features['in_channel']
            cout = self.features['out_channel']
            expand_ratio = self.features['expand_ratio']
            
            expand_base_width = cin if self.features['expand_base_on'] == 'in' else cout
            
            middle_width = make_divisible(expand_base_width * expand_ratio,
                                          self.features["divisor"])
            # update middle width of middle conv in main_path
            self.features['middle_width'] = middle_width
            
            inp_shape = self.features["in_shape"]
            cur_shape = inp_shape
            cur_width = cin
            dw_conv = None
            first_matmul = None
            first_add = None
            last_matmul = None
            last_add = None
            mul_scaler = None
            for op in self.main_path:
                if isinstance(op, ConvOp):
                    if dw_conv is None:
                        dw_conv = op
                        op.update({"in_channel": cur_width,
                                "in_shape": cur_shape})
                    else:
                        raise NotImplementedError
                elif isinstance(op, LinearMatMulOp):
                    if first_matmul is None:
                        first_matmul = op
                        op.update({"in_channel": cur_width,
                                    "out_channel": middle_width,
                                    "in_shape": cur_shape})
                    elif last_matmul is None:
                        last_matmul = op
                        op.update({"in_channel": cur_width,
                                    "out_channel": cout,
                                    "in_shape": cur_shape})
                    else:
                        raise NotImplementedError
                elif isinstance(op, ScaleMulOp):
                    if mul_scaler is None:
                        mul_scaler = op
                        op.update({"in_channel": cur_width,
                                    "out_channel": cout,
                                    "in_shape": cur_shape})
                else:
                    op.update({"in_channel": cur_width,
                            "in_shape": cur_shape,
                            "out_channel": cur_width})
                cur_shape = op.features["out_shape"]
                cur_width = op.features["out_channel"]
    
