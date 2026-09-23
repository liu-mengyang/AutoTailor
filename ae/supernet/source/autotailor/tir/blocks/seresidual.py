from .convresidual import ConvResidualBlock
from .single_op import *
from .tailor_utils import make_divisible


class SEResidualBlock(ConvResidualBlock):
    def __init__(self, conv_residual_block, reduce_ratio=4, divisor=8):
        """Build residual block

        Args:
            conv_residual_block: the basic block.
            reduce_ratio: the ratio number for reducing dimension
        """
        super(SEResidualBlock, self).__init__(conv_residual_block.start_node,
                                    conv_residual_block.last_node,
                                    conv_residual_block.residual_path,
                                    conv_residual_block.main_path)
        self.type = 'SEResidualBlock'
        
        # extract information
        first_conv = None
        last_conv = None
        for op in self.main_path:
            if isinstance(op, ConvOp):
                if first_conv is None:
                    first_conv = op
                elif last_conv is None:
                    last_conv = op
                else:
                    raise NotImplementedError
        # if self.expandable:
            
        # else:
        self.features['trans_type'] = 'passive'
        self.features['middle_width'] = last_conv.features['in_channel']
        self.features['reduce_ratio'] = reduce_ratio
        self.features['divisor'] = divisor

        self.dynamic_dimensions = []
    
    def reorganize_weight(self, sorted_idx):
        first_conv = None
        last_conv = None
        for op in self.main_path:
            if isinstance(op, ConvOp):
                if first_conv is None:
                    first_conv = op
                elif last_conv is None:
                    last_conv = op
                else:
                    raise NotImplementedError
        
        # reorganize the last conv in cout dim
        last_conv.super_weights.data = torch.index_select(
            last_conv.super_weights.data, 0, sorted_idx
        )
        if last_conv.features["has_bias"]:
            last_conv.super_bias.data = torch.index_select(
                first_conv.super_bias.data, 0, sorted_idx
            )
        
        # reorganize the first conv in cin dim
        first_conv.super_weights.data = torch.index_select(
            first_conv.super_weights.data, 1, sorted_idx
        )
        
        # compute se index
        se_importance = torch.sum(torch.abs(last_conv.super_weights.data), dim=(0, 2, 3))
        se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

        last_conv.super_weights.data = torch.index_select(last_conv.super_weights.data, 1, se_idx)
        
        first_conv.super_weights.data = torch.index_select(first_conv.super_weights.data, 0, se_idx)
        if first_conv.features["has_bias"]:
            first_conv.super_bias.data = torch.index_select(
                first_conv.super_bias.data, 0, se_idx
            )
    
    def update(self, kv_features: dict) -> None:
        """Update features of the block.
        
        Extra update out_channel.
        
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
        if self.features["in_channel"] == 0:
            # Init channels
            self.features["in_channel"] = inp_shape[1]
            self.features["out_channel"] = inp_shape[1]
        
        cin = self.features['in_channel']
        self.features["out_channel"] = self.features["in_channel"] # SE constraint
        cout = self.features['out_channel']
        assert cin==cout
        reduce_ratio = self.features['reduce_ratio']
        
        middle_width = make_divisible(cin // reduce_ratio, self.features['divisor'])
        self.features['middle_width'] = middle_width # update middle width
        
        cur_shape = inp_shape
        cur_width = cin
        first_conv = None
        last_conv = None
        for op in self.main_path:
            if isinstance(op, ConvOp):
                if first_conv is None:
                    first_conv = op
                    op.update({"in_shape": cur_shape,
                               "in_channel": cur_width,
                               "out_channel": middle_width})
                    cur_shape = op.features["out_shape"]
                    cur_width = op.features["out_channel"]
                elif last_conv is None:
                    last_conv = op
                    op.update({"in_shape": cur_shape,
                               "in_channel": cur_width,
                               "out_channel": cout})
                    cur_shape = op.features["out_shape"]
                    cur_width = op.features["out_channel"]
            else:
                op.update({"in_shape": cur_shape,
                           "in_channel": cur_width,
                           "out_channel": cur_width})
                cur_shape = op.features["out_shape"]
                cur_width = op.features["out_channel"]
        assert first_conv and last_conv
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
            cur_shape = node.features["out_shape"]
            cur_width = node.features['out_channel']
        residual_path_out_shape = cur_shape
        residual_path_out_width = cur_width
        assert main_path_out_width == residual_path_out_width
        
        if self.last_node.type == "Mul":
            self.last_node.features["in_shape"] = (main_path_out_shape, residual_path_out_shape)
        else:
            self.last_node.features["in_shape"] = cur_shape
        # print(main_path_out_shape)
        # print(residual_path_out_shape)
        self.last_node.update({"in_shape": (main_path_out_shape, residual_path_out_shape),
                               "in_channel": cur_width,
                               "out_channel": cur_width})
        
        self.features['out_channel'] = self.last_node.features["out_channel"]
        self.features['out_shape'] = self.last_node.features["out_shape"]
    


