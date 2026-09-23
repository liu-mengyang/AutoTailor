from loguru import logger

from .block import Block
from .seresidual import SEResidualBlock
from .convresidual import ConvResidualBlock
from .single_op import *
from .tailor_utils import make_divisible, closest


class BottleneckResidualBlock(ConvResidualBlock):
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
        super(BottleneckResidualBlock, self).__init__(start_node,
                         last_node,
                         first_path,
                         second_path)
        self.type = 'BottleneckResidualBlock'
        
        expand_base_on = supernet_cfg_dict["arch"]["BottleneckResidualBlock"]["expand_base"]
        
        if "BottleneckResidualBlock" in supernet_cfg_dict["var"]["block_vars"]:
            all_expand_ratios = supernet_cfg_dict["var"]["block_vars"]["BottleneckResidualBlock"]["expand_ratio"]
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
        first_conv = None
        middle_conv = None
        last_conv = None
        for op in self.main_path:
            if isinstance(op, ConvOp):
                if first_conv is None:
                    first_conv = op
                elif middle_conv is None:
                    middle_conv = op
                elif last_conv is None:
                    last_conv = op
                else:
                    raise NotImplementedError
        
        expand_base_width = first_conv.features['in_channel'] if expand_base_on=='in' else last_conv.features['out_channel']
        
        self.expand_ratio_list = expand_ratio_list
        self.features['expand_base_on'] = expand_base_on
        self.features['middle_width'] = middle_conv.features['in_channel']
        self.features['kernel_size'] = middle_conv.features['kernel_size']
        self.features['max_kernel_size'] = middle_conv.features['kernel_size']
        self.features['max_expand_ratio'] = closest(self.features['middle_width'] / expand_base_width, expand_ratio_list)
        self.features['expand_ratio'] = self.features['max_expand_ratio']
        self.features["divisor"] = divisor
        
        self.dynamic_dimensions.append('expand_ratio')
    
    def reorganize_weight(self, cfgs):
        first_conv = None
        middle_conv = None
        last_conv = None
        for node in self.main_path:
            if isinstance(node, ConvOp):
                if first_conv is None:
                    first_conv = node
                    
                elif middle_conv is None:
                    middle_conv = node
                    
                elif last_conv is None:
                    last_conv = node
                else:
                    raise NotImplementedError
        assert first_conv and middle_conv and last_conv
        
        if "expand_ratio" in cfgs:
            # reorganize weights for dynamic expand ratio in progressive training
            importance = torch.sum(
                torch.abs(last_conv.super_weights.data), dim=(0, 2, 3)
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
            
            # reorganize the first conv in cout dim
            logger.info("Reorganize first conv")
            first_conv.super_weights.data = torch.index_select(
                first_conv.super_weights.data, 0, sorted_idx
            )
            if first_conv.features["has_bias"]:
                first_conv.super_bias.data = torch.index_select(
                    first_conv.super_bias.data, 0, sorted_idx
                )
            
            # reorganize the middle conv in channel dim
            logger.info("Reorganize middle conv")
            middle_conv.super_weights.data = torch.index_select(
                middle_conv.super_weights.data, 0, sorted_idx
            )
            if middle_conv.features["has_bias"]:
                middle_conv.super_bias.data = torch.index_select(
                    middle_conv.super_bias.data, 0, sorted_idx
                )
            
            # reorganize the last conv in cin dim
            logger.info("Reorganize last conv")
            last_conv.super_weights.data = torch.index_select(
                last_conv.super_weights.data, 1, sorted_idx
            )
            
            bn_cnt = 0
            for node in self.main_path:
                if isinstance(node, BNOp):
                    logger.info("Reorganize weight of bn")
                    bn_cnt += 1
                    if bn_cnt >= 3:
                        continue
                    node.super_weights.data = torch.index_select(node.super_weights.data, 0, sorted_idx)
                    node.super_bias.data = torch.index_select(node.super_bias.data, 0, sorted_idx)
                    node.super_mean.data = torch.index_select(node.super_mean.data, 0, sorted_idx)
                    node.super_var.data = torch.index_select(node.super_var.data, 0, sorted_idx)
                elif isinstance(node, SEResidualBlock):
                    node.reorganize_weight(sorted_idx)
                
                
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
        # if self.start_node.type == "Reshape":
        #     self.start_node.update({"in_channel": cin,
        #                             "in_shape": inp_shape})
        #     inp_shape = self.start_node.features["out_shape"]
        #     cin = inp_shape[1]
        
        expand_ratio = self.features["expand_ratio"]
        base_width = cin if self.features["expand_base_on"] == "in" else cout
        middle_width = make_divisible(base_width * expand_ratio,
                                      self.features["divisor"])
        self.features["middle_width"] = middle_width
        
        cur_width = cin
        first_conv = None
        middle_conv = None
        last_conv = None
        cur_shape = inp_shape
        for node in self.main_path:
            # update each node in flow
            if isinstance(node, ConvOp):
                if first_conv is None:
                    first_conv = node
                    node.update({"in_channel": cur_width,
                                "out_channel": middle_width,
                                "in_shape": cur_shape})
                elif middle_conv is None:
                    middle_conv = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": cur_width,
                                 "in_shape": cur_shape})
                elif last_conv is None:
                    last_conv = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": cout,
                                 "in_shape": cur_shape})
                else:
                    raise NotImplementedError
            else:
                node.update({"in_channel": cur_width,
                             "in_shape": cur_shape})
            cur_width = node.features["out_channel"]
            cur_shape = node.features["out_shape"]
        main_path_out_shape = cur_shape
        main_path_out_width = cur_width
        
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
        residual_path_out_shape = cur_shape
        residual_path_out_width = cur_width
        
        assert residual_path_out_shape == main_path_out_shape, f"{residual_path_out_shape}, {main_path_out_shape}"
        self.features['out_channel'] = cur_width
        self.features['out_shape'] = cur_shape
        self.last_node.features["in_shape"] = cur_shape
        self.last_node.features["out_shape"] = cur_shape
        self.last_node.features["in_channel"] = cur_shape[1]
        self.last_node.features["out_channel"] = cur_shape[1]
    
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

            # if self.start_node.type == "Reshape":
            #     self.start_node.update({"in_channel": cin, "in_shape": inp_shape})
            #     inp_shape = self.start_node.features["out_shape"]
            #     cin = inp_shape[1]
            
            cur_shape = inp_shape
            cur_width = cin
            first_conv = None
            middle_conv = None
            last_conv = None
            for op in self.main_path:
                if isinstance(op, ConvOp):
                    if first_conv is None:
                        first_conv = op
                        op.update({"in_channel": cur_width,
                                "in_shape": cur_shape,
                                "out_channel": middle_width})
                    elif middle_conv is None:
                        middle_conv = op
                        op.update({"in_channel": cur_width,
                                "in_shape": cur_shape,
                                "out_channel": cur_width})
                    elif last_conv is None:
                        last_conv = op
                        op.update({"in_channel": cur_width,
                                "in_shape": cur_shape,
                                "out_channel": cout})
                    else:
                        raise NotImplementedError
                else:
                    op.update({"in_channel": cur_width,
                            "in_shape": cur_shape,
                            "out_channel": cur_width})
                cur_shape = op.features["out_shape"]
                cur_width = op.features["out_channel"]
            assert first_conv and middle_conv and last_conv
    
