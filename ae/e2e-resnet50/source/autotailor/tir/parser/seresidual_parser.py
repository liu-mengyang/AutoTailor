import onnx_graphsurgeon as gs

from autotailor.tir.stage import Stage
from autotailor.tir.blocks import ConvOp
from autotailor.tir.blocks import ConvResidualBlock, SEResidualBlock
from .single_op_parser import parse_single_node

__all__ = ["parse_seresidual"]


def parse_seresidual(
    block: ConvResidualBlock,
) -> SEResidualBlock:
    """Judge whether this is a se residual block and parse it if it is.
    
    Two requirements to judge:
      1. Have two conv ops;
      2. cin of first conv equal to cout of last conv.
    
    Args:
      cur_node: the start node of the block.
      last_node: the last node of the block.
      first_path: the first path of the block.
      second_path: the second path of the block.
    
    Returns:
      The original conv residual block if it is not a se block.
      The TailorIR block of se block if it is.
    """
    conv_count = 0
    first_conv = None
    last_conv = None
    for op in block.main_path:
        if isinstance(op, ConvOp):
            conv_count += 1
            if first_conv is None:
                first_conv = op
            elif last_conv is None:
                last_conv = op
            else:
                return block
    
    if (conv_count == 2 and 
            first_conv.features['in_channel'] == last_conv.features['out_channel'] and
            first_conv.features['in_channel'] != first_conv.features['out_channel']):
        block = SEResidualBlock(block)
        return block
        
    return block


