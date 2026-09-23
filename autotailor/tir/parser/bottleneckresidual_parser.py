import onnx_graphsurgeon as gs

from autotailor.tir.blocks import ConvOp
from autotailor.tir.blocks import BottleneckResidualBlock
from .single_op_parser import parse_single_node

__all__ = ["parse_bottleneckresidual"]


def parse_bottleneckresidual(
    cur_node: gs.Node,
    last_node: gs.Node,
    first_path: list,
    second_path: list,
    supernet_cfg_dict: dict,
) -> BottleneckResidualBlock:
    """Judge whether this is a bottleneck residual block and parse it if it is.
    
    Two requirements to judge:
      1. Three conv op;
      2. The kernel size of the first and last conv is 1 and the kernel size of
         the middle conv is not 1.
    
    Args:
      cur_node: the start node of the block.
      last_node: the last node of the block.
      first_path: the first path of the block.
      second_path: the second path of the block.
      supernet_cfg_dict: the dict of dynamic dimensions.
      
    Returns:
      None if it is not a bottleneck residual block.
      The TailorIR block of bottlneck residual if it is.
    """
    paths = [first_path, second_path]
    total_conv_count = 0
    for i, path in enumerate(paths):
        conv_count = 0
        first_conv = None
        middle_conv = None
        last_conv = None
        for op in path:
            if isinstance(op, ConvOp):
                conv_count += 1
                if first_conv is None:
                    first_conv = op
                elif middle_conv is None:
                    middle_conv = op
                elif last_conv is None:
                    last_conv = op
                else:
                    return None
        
        if (conv_count == 3 and first_conv.features['kernel_size'] == 1 and
            last_conv.features['kernel_size'] == 1 and
            middle_conv.features['kernel_size'] != 1):
            first_node = parse_single_node(cur_node)
            last_node = parse_single_node(last_node)
            block = BottleneckResidualBlock(first_node,
                                            last_node,
                                        first_path,
                                        second_path,
                                        supernet_cfg_dict)
            return block
    
    return None