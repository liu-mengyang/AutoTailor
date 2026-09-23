import onnx_graphsurgeon as gs

from autotailor.tir.graph_utils import explore_first_multi_input_node, is_last_node
from .single_op_parser import parse_single_node
from .bottleneckresidual_parser import parse_bottleneckresidual
from .convresidual_parser import parse_convresidual
from .seresidual_parser import parse_seresidual
from .residual_parser import parse_residual
from .triplet_parser import parse_triplet_block

__all__ = ["parse_diamond_block", "explore_path"]


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
    # print(f"Explore path from {start_node.name} to {end_node.name}")
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
        # print(f"{cur_node.name} has {num_outputs_path} output path and {num_outputs} outputs")
        if num_outputs == 1 and num_outputs_path < 3:
            if num_outputs_path == 1:
                node = parse_single_node(cur_node)
                nodes_lst, cur_idx = add_node(nodes_lst,
                                            cur_idx,
                                            node)
                cur_node = cur_node.o()
            elif num_outputs_path != 1:
                # Meet nested block
                if num_outputs_path == 2:
                    # nested two path block
                    # print("Dive into")
                    block, last_node = parse_diamond_block(cur_node, supernet_cfg_dict)
                    # print("Dive out")
                    # print(cur_node.name)
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
                raise NotImplementedError
                
        elif num_outputs == 3 or num_outputs_path == 3:
            # most possible is split node
            block, last_node = parse_triplet_block(cur_node, supernet_cfg_dict)
            block.start_node.idx = cur_idx
            nodes_lst, cur_idx = add_node(nodes_lst,
                                            cur_idx,
                                            block.start_node)
            # print(f"block start node: {block.start_node}")

            nodes_lst, cur_idx = add_node(nodes_lst,
                                        cur_idx,
                                        block)
            
            cur_node = last_node
            if len(cur_node.outputs[0].outputs) == 1:
                # Avoid access into a new block due to two path block are all
                # diamond structure in this version.
                cur_node = cur_node.o()
        else:
            raise NotImplementedError
        
    return nodes_lst


def parse_diamond_block(cur_node: gs.Node, supernet_cfg_dict: dict):
    """Parse a diamond block

    Args:
      cur_node: the first node of diamond block

    Returns:
      The built block
    
    Raises:
      NotImplementedError: An error occured meeting multi-path subgraph.
    """
    # print(f"Parsing diamond block from {cur_node.name}")
    block = None
    if isinstance(cur_node, list):
        first_node = cur_node[0]
        second_node = cur_node[1]
    else:
        first_node = cur_node.o(0, 0)
        second_node = cur_node.o(1, 0)
    
    last_node = explore_first_multi_input_node(first_node)
    
    second_last_node = explore_first_multi_input_node(second_node)
    if last_node is None or second_last_node is None:
        raise NotImplementedError(f"Meet unsupport multi-path subgraph from {cur_node.name}")
    elif last_node == second_last_node:
        pass
    else:
        # one last node is not the ture last node
        # try to find the second_last_node from the last_node
        # print(last_node.name)
        # print(second_last_node.name)
        res = explore_first_multi_input_node(last_node, exclude_itself=True)
        find_flag = False
        while res is not None:
            # print(f"Current res: {res.name}")
            if res == second_last_node:
                # second_last_node is true
                last_node = second_last_node
                find_flag = True
                break
            res = explore_first_multi_input_node(res, exclude_itself=True)
        if not find_flag:
            # check whether the last node is true
            res = explore_first_multi_input_node(second_last_node, exclude_itself=True)
            while res is not None:
                # print(f"Current res: {res.name}")
                if res == last_node:
                    # last node is true
                    find_flag = True
                    break
                res = explore_first_multi_input_node(res, exclude_itself=True)
        if not find_flag:
        # currently not considered nest and nest in one block that two nodes are both false last node
            raise NotImplementedError
    
    first_path = []
    second_path = []
    if first_node != last_node:
        first_path = explore_path(first_node, last_node, supernet_cfg_dict)
    if second_node != last_node:
        second_path = explore_path(second_node, last_node, supernet_cfg_dict)
    
    block = parse_bottleneckresidual(cur_node, last_node, first_path,
                                     second_path, supernet_cfg_dict)
    
    if block is None:
        block = parse_convresidual(cur_node, last_node, first_path, second_path)
        if block:
            # Check whether conv residual is se block
            block = parse_seresidual(block)
    if block is None:
        block = parse_residual(cur_node, last_node, first_path, second_path)
    return block, last_node
