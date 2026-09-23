import onnx_graphsurgeon as gs

from autotailor.tir.blocks import ResidualBlock
from .single_op_parser import parse_single_node

__all__ = ["parse_residual"]


def parse_residual(
    cur_node: gs.Node,
    last_node: gs.Node,
    first_path: list,
    second_path: list,
) -> ResidualBlock:
    """Parse into a residual block.
    
    No requirements.
    
    Args:
      cur_node: the start node of the block.
      last_node: the last node of the block.
      first_path: the first path of the block.
      second_path: the second path of the block.
    
    Returns:
      The TailorIR block of residual if it is.
    """
    block = ResidualBlock(parse_single_node(cur_node),
                          parse_single_node(last_node),
                          first_path,
                          second_path)
    return block