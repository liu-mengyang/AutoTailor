import onnx_graphsurgeon as gs

from autotailor.tir.stage import Stage
from autotailor.tir.graph_utils import is_last_node
from .single_op_parser import parse_single_node
from .diamond_parser import parse_diamond_block
from .triplet_parser import parse_triplet_block


__all__ = ["first_parse"]


def first_parse(
    first_node: gs.Node,
    supernet_cfg_dict: dict,
    inp_shape: tuple) -> Stage:
    """ Parse DAG into sequential blocks in TailorIR
    
    First parsing DAG into TailorIR in one stage.
    
    Args:
      first_node: The first node of the DAG.
      supernet_cfg_dict: The dictionary of supernet configuration.
      inp_shape: The shape of the input feature.
    
    Return:
      Initialized TailorIR stage.
    
    Raises:
      NotImplementedError: An error occured meeting unsupported subgraph.
    """
    initial_stage = Stage()
    cur_node = first_node

    # first block is a diamond block
    if isinstance(cur_node, list):
        # currently only support diamond block
        assert len(cur_node) == 2
        block, last_node = parse_diamond_block(cur_node, supernet_cfg_dict)
        initial_stage.add_block(block)
        cur_node = last_node
        
    while not is_last_node(cur_node):
        # Iterating all nodes until the last one.
        
        num_outputs = len(cur_node.outputs)
        
        if num_outputs == 1:
          num_outputs_path = len(cur_node.outputs[0].outputs)
          # Judge block or single op by number of outputs of node.

          if num_outputs_path == 1:
              # Single op block, building directly
              block = parse_single_node(cur_node)
              initial_stage.add_block(block)
              cur_node = cur_node.o()
          elif num_outputs_path == 2:
              # Two path block
              block, last_node = parse_diamond_block(cur_node, supernet_cfg_dict)
              if block.start_node.type != 'Add' and block.start_node.type != 'Mul':
                  # Avoid reducing the first node of the block other than they are reduce op.
                  initial_stage.add_block(block.start_node)
              initial_stage.add_block(block)
              cur_node = last_node
              if len(cur_node.outputs[0].outputs) == 1:
                  # Avoid access into a new block due to two path block are all
                  # diamond structure in this version.
                  cur_node = cur_node.o()
          else:
              # Not support for block with three or more paths
              raise NotImplementedError
        elif num_outputs == 3:
          # most possible is split node
          block, last_node = parse_triplet_block(cur_node, supernet_cfg_dict)
          initial_stage.add_block(block.start_node)
          initial_stage.add_block(block)
          cur_node = last_node
          if len(cur_node.outputs[0].outputs) == 1:
            # Avoid access into a new block due to two path block are all
            # diamond structure in this version.
            cur_node = cur_node.o()
        else:
          raise NotImplementedError
          
    # last node
    block = parse_single_node(cur_node)
    initial_stage.add_block(block)
    
    # Update shape information
    if len(inp_shape) == 4:
        cur_width = inp_shape[1]
    else:
        cur_width = inp_shape[-1]
    cur_width = inp_shape[-1]
    for block_id, block in initial_stage.flow.items():
        # print(block)
        # print(f"inp: {inp_shape}; in_channel: {cur_width}")
        block.update({"in_shape": inp_shape, "in_channel": cur_width})
        inp_shape = block.features['out_shape']
        cur_width = block.features["out_channel"]
        # print(f"out: {inp_shape}")
        
    return initial_stage
