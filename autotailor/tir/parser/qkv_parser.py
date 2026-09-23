import onnx_graphsurgeon as gs

from autotailor.tir.blocks import QKVBlock, ResidualBlock
from autotailor.tir.graph_utils import explore_first_multi_input_node
from .single_op_parser import parse_single_node

__all__ = ["parse_qkv"]


def explore_path(cur_node: gs.Node, end_node: gs.Node):
    nodes_lst = []
    while cur_node != end_node:
        # Iterate until meeting end node
        # print(cur_node.name)
        # print(end_node.name)
        num_outputs_path = len(cur_node.outputs[0].outputs)
        num_outputs = len(cur_node.outputs)
        assert num_outputs_path == 1 and num_outputs == 1
        node = parse_single_node(cur_node)
        nodes_lst.append(node)
        cur_node = cur_node.o()
        # print(f"Next: {cur_node.name}")
    # print("Found")
    return nodes_lst


def parse_qkv(start_node: gs.Node) -> QKVBlock:
    q_path = []
    k_path = []
    qk_path = []
    v_path = []
    if start_node.op == "Split":
        assert len(start_node.outputs) == 3 and start_node.op == "Split"
        
        q_trans_node = start_node.outputs[0].outputs[0]
        k_trans_node = start_node.outputs[1].outputs[0]
        v_reshape_node = start_node.outputs[2].outputs[0]
        
        matmul_node = q_trans_node.outputs[0].outputs[0]
        last_node = explore_first_multi_input_node(matmul_node.o())
        if q_trans_node.op == "Transpose" and k_trans_node.op == "Transpose" and v_reshape_node.op == "Reshape" and matmul_node.op == "MatMul":
            first_node = parse_single_node(start_node)
            if last_node and last_node.op == "MatMul":
                q_path = [parse_single_node(q_trans_node)]
                k_path = [parse_single_node(k_trans_node)]
                qk_residual = ResidualBlock(first_node,
                                            parse_single_node(matmul_node),
                                            q_path,
                                            k_path)
                qk_path_tail = explore_path(matmul_node.o(), last_node)
                qk_path = [qk_residual] + qk_path_tail
                v_path = explore_path(v_reshape_node, last_node)
                
                block = QKVBlock(first_node,
                                parse_single_node(last_node),
                                qk_path,
                                v_path)
                return block, last_node
            else:
                raise NotImplementedError
    elif start_node.op == "LayerNormalization":
        assert len(start_node.outputs[0].outputs) == 3
        v_start_node_index = 0
        for i in range(len(start_node.outputs[0].outputs)):
            node = start_node.outputs[0].outputs[i]
            while node.op != "Transpose":
                node = node.o()
            # q, k are multipled by sqrt l
            if node.o().op == "MatMul":
                v_start_node_index = i
                break
        
        # nodes: [q, k, v]
        qkv_start_nodes = []
        for i in range(len(start_node.outputs[0].outputs)):
            if i != v_start_node_index:
                qkv_start_nodes.append(start_node.outputs[0].outputs[i])
        qkv_start_nodes.append(start_node.outputs[0].outputs[v_start_node_index])
        # print(qkv_start_nodes)
        assert len(qkv_start_nodes) == 3

        last_node = explore_first_multi_input_node(qkv_start_nodes[2].o())
        qk_last_node = explore_first_multi_input_node(qkv_start_nodes[0].o())
        first_node = parse_single_node(start_node)
        if last_node and last_node.op == "MatMul":
            q_path = explore_path(qkv_start_nodes[0], qk_last_node)
            k_path = explore_path(qkv_start_nodes[1], qk_last_node)
            qk_residual = ResidualBlock(first_node,
                                        parse_single_node(qk_last_node),
                                        q_path,
                                        k_path)
            qk_path_tail = explore_path(qk_last_node.o(), last_node)
            qk_path = [qk_residual] + qk_path_tail
            v_path = explore_path(qkv_start_nodes[2], last_node)

                
            block = QKVBlock(first_node,
                            parse_single_node(last_node),
                            qk_path,
                            v_path)
            return block, last_node
        else:
            raise NotImplementedError
    else:
        return None, last_node
    
