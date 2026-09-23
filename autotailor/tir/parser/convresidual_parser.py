import onnx_graphsurgeon as gs

from autotailor.tir.blocks import ConvOp
from autotailor.tir.blocks import ConvResidualBlock
from .single_op_parser import parse_single_node

__all__ = ["parse_convresidual"]


def parse_convresidual(
    cur_node: gs.Node,
    last_node: gs.Node,
    first_path: list,
    second_path: list,
) -> ConvResidualBlock:
    """Judge whether this is a conv residual block and parse it if it is.
    
    One requirements to judge: have conv op.
    
    Args:
      cur_node: the start node of the block.
      last_node: the last node of the block.
      first_path: the first path of the block.
      second_path: the second path of the block.
    
    Returns:
      None if it is not a conv residual block.
      The TailorIR block of conv residual if it is.
    """
    paths = [first_path, second_path]
    total_conv_count = 0
    for i, path in enumerate(paths):
        conv_count = 0
        for op in path:
            if isinstance(op, ConvOp):
                conv_count += 1
        
        if conv_count > 0:
            first_node = parse_single_node(cur_node)
            last_node = parse_single_node(last_node)
            # print(last_node)
            # print(first_path)
            # print(second_path)
            block = ConvResidualBlock(first_node,
                                      last_node,
                                      first_path,
                                      second_path)
            return block
    
    return None