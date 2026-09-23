import onnx_graphsurgeon as gs

from .single_op_parser import parse_single_node
from .diamond_parser import parse_diamond_block

__all__ = ["explore_path"]


def explore_path(
    start_node: gs.Node,
    end_node: gs.Node,
    supernet_cfg_dict: dict
) -> list:
    """Explore the path of a sequential flow to gather all nodes in the path

    Args:
      start_node: the first node of the path
      end_node: the last node of the path

    Returns:
      The path of the flow
    
    Raises:
      NotImplementedError: An error occured meeting unsupport block
    """
    
    def add_node(nodes_lst, cur_idx, block):
        block.idx = cur_idx
        cur_idx += 1
        nodes_lst.append(block)
        return nodes_lst, cur_idx
    
    nodes_lst = []
    cur_node = start_node
    cur_idx = 0
    while cur_node != end_node:
        # Iterate until meeting end node
        num_outputs_path = len(cur_node.outputs[0].outputs)
        num_outputs = len(cur_node.outputs)
        if num_outputs_path != 1:
            # Meet nested block
            if num_outputs_path == 2:
                # nested two path block
                block, last_node = parse_diamond_block(cur_node, supernet_cfg_dict)
                block.start_node.idx = cur_idx
                
                if block.start_node.type != 'Add' and block.start_node.type != 'Mul':
                    nodes_lst, cur_idx = add_node(nodes_lst,
                                                  cur_idx,
                                                  block.start_node)
                nodes_lst, cur_idx = add_node(nodes_lst,
                                              cur_idx,
                                              block)
                cur_node = last_node
                if len(cur_node.outputs[0].outputs) == 1:
                    cur_node = cur_node.o()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        else:
            node = parse_single_node(cur_node)
            nodes_lst, cur_idx = add_node(nodes_lst,
                                        cur_idx,
                                        node)
            cur_node = cur_node.o()
        
    return nodes_lst