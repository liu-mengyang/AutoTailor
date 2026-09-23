import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs
import numpy as np

from loguru import logger

import autotailor.tir.globvar as globvar

__all__ = ["get_node_from_name", "get_first_node",
           "is_last_node", "explore_first_multi_input_node"]


def get_node_from_name(graph, name):
    for node in graph.nodes:
        if node.name == name:
            return node
    return None

def get_first_node(graph):
    nodes = []
    for node in graph.nodes:
        if node.inputs[0].name == 'input':
            nodes.append(node)
    if len(nodes) == 1:
        return nodes[0]
    elif len(nodes) == 0:
        return None
    else:
        return nodes

def is_last_node(node):
    if node.outputs[0].name == 'output':
        return node
    return None

def is_multi_input_node(node):
    # pass BN for running mean and var
    if node.op == 'BatchNormalization':
        return False
    num_input_path = 0
    for inp in node.inputs:
        if len(inp.inputs) != 0:
            for inpinp in inp.inputs:
                # double loop for skipping constant
                if len(inpinp.inputs) != 0:
                    num_input_path += 1
    if num_input_path > 1:
        return True
    else:
        return False


def explore_first_multi_input_node(cur_node, exclude_itself=False):
    junction_node = None
    while True:
        # print(cur_node.name)
        if is_multi_input_node(cur_node) and not exclude_itself:
            junction_node = cur_node
            return junction_node
        elif len(cur_node.outputs[0].outputs) == 2:
            # meet nested block
            sub_junction_node = explore_first_multi_input_node(cur_node.o(0,0))
            cur_node = sub_junction_node.o()
        elif is_last_node(cur_node):
            return junction_node
        elif len(cur_node.outputs) == 3 and cur_node.op != "Split":
            print(f"Return None due to meet {cur_node.name} with 3 outputs")
            return None # skip QKV in transformer (3 paths)
        else:
            exclude_itself=False
            cur_node = cur_node.o()


def get_ops(graph):
    ops_lst = []
    for node in graph.nodes:
        op = node.op
        if op not in ops_lst:
            ops_lst.append(op)
    return ops_lst


def extract_sub_graph(nodes_name):
    # Nodes are in sequential (at least the first and the last nodes are correct)
    nodes = []
    inputs = []
    outputs = []
    for node_name in nodes_name:
        node = get_node_from_name(globvar.onnx_graph, node_name)
        nodes.append(node)
        inputs.append(node.inputs)
        outputs.append(node.outputs)
        inputs_name = []
        outputs_name = []
        for input_tensor in node.inputs:
            inputs_name.append(input_tensor.name)
        for output_tensor in node.outputs:
            outputs_name.append(output_tensor.name)
        # logger.debug(f'Inputs of {node.name}: {str(inputs_name)}')
        # logger.debug(f'Outpus of {node.name}: {str(outputs_name)}')
    new_tensors = []
    new_inputs = []
    new_outputs = []
    inp_idx = 0
    new_tensor_dict = {}
    for i, node_inputs in enumerate(inputs):
        node_new_inputs = []
        node_new_tensors = []
        for input_tensor in node_inputs:
            if isinstance(input_tensor, gs.Constant):
                new_input = gs.Constant(name=input_tensor.name, values=input_tensor.values, data_location=input_tensor.data_location)
            elif isinstance(input_tensor.dtype, np.dtype):
                if i == 0:
                    input_tensor_name = 'input'+str(inp_idx)
                    inp_idx+=1
                else:
                    input_tensor_name = input_tensor.name
                if input_tensor_name not in new_tensor_dict:
                    new_tensor = gs.Variable(name=input_tensor_name, dtype=input_tensor.dtype, shape=input_tensor.shape)
                    new_tensor_dict[input_tensor_name] = new_tensor
                else:
                    new_tensor = new_tensor_dict[input_tensor_name]
                new_input = new_tensor
                node_new_tensors.append(new_tensor)
            else:
                if i == 0:
                    input_tensor_name = 'input'+str(inp_idx)
                    inp_idx+=1
                else:
                    input_tensor_name = input_tensor.name
                tensor_shape = input_tensor.shape
                # if not the first node, this input should add shape
                if len(input_tensor.inputs) != 0:
                    pre_node = input_tensor.inputs[0]
                    tensor_shape = globvar.shape_dict[pre_node.name]
                    if isinstance(tensor_shape, list):
                        for idx, tensor_o in enumerate(pre_node.outputs):
                            if tensor_o == input_tensor:
                                tensor_shape = tensor_shape[idx]
                                break
                if input_tensor_name not in new_tensor_dict:
                    new_tensor = gs.Variable(name=input_tensor_name, dtype=np.float32, shape=tensor_shape)
                    new_tensor_dict[input_tensor_name] = new_tensor
                else:
                    new_tensor = new_tensor_dict[input_tensor_name]
                new_input = new_tensor
                node_new_tensors.append(new_tensor)
            node_new_inputs.append(new_input)
        new_inputs.append(node_new_inputs)
        new_tensors.append(node_new_tensors)
    # logger.debug(list(new_tensor_dict.keys()))
    all_outputs = []
    for node_outputs in outputs:
        for output_tensor in node_outputs:
            all_outputs.append(output_tensor)
    
    for i, node_outputs in enumerate(outputs):
        if i != len(outputs)-1:
            node_new_outputs = []
            for j, output_tensor in enumerate(node_outputs):
                for tensor_name, new_tensor in new_tensor_dict.items():
                    if output_tensor.name == tensor_name:
                        node_new_outputs.append(new_tensor)
                        break
            new_outputs.append(node_new_outputs)
        else:
            # The last node has output tensors
            node_new_outputs = []
            for j, output_tensor in enumerate(node_outputs):
                output_tensor_name = 'output'+str(j)
                if isinstance(output_tensor.dtype, np.dtype):
                    new_output = gs.Variable(name=output_tensor_name, dtype=output_tensor.dtype, shape=output_tensor.shape)
                else:
                    tensor_shape = globvar.shape_dict[nodes_name[i]]
                    if isinstance(tensor_shape, list):
                        tensor_shape = tensor_shape[j]
                    if len(tensor_shape)==0:
                        pre_node = output_tensor.inputs[0].i()
                        tensor_shape = globvar.shape_dict[pre_node.name]
                    new_output = gs.Variable(name=output_tensor_name, dtype=np.float32, shape=tensor_shape)
                node_new_outputs.append(new_output)
            new_outputs.append(node_new_outputs)
    # Find out extra inputs for handling multiin node
    extra_inputs = []
    for i, tensors in enumerate(new_tensors):
        for j, tensor in enumerate(tensors):
            if i == 0 and j == 0:
                continue
            # logger.info(f'checking {tensor.name} ...')
            match_flag = False
            for output_tensor in all_outputs:
                if tensor.name == output_tensor.name:
                    match_flag = True
                    # logger.debug(f'{tensor.name} is not input')
                    break
            if not match_flag:
                for k, inp_tensor in enumerate(new_inputs[i]):
                    if inp_tensor.name == tensor.name:
                        if len(tensor.shape)==0:
                            raise NotImplementedError
                            # if nodes[i].op=="Add": # gs bug: constant of add has not detected
                            #     inp_tensor.to_constant(np.array(3.0, dtype=np.float32)) #NOTICE: This is wrong, but currently only relu6 in
                            #     break
                            # elif nodes[i].op=="Div": # gs bug: constant of div has not detected
                            #     inp_tensor.to_constant(np.array(6.0, dtype=np.float32)) #NOTICE: This is wrong, but currently only relu6 in
                            #     break
                        tensor = gs.Variable(name='input'+str(inp_idx), dtype=tensor.dtype, shape=tensor.shape)
                        inp_idx += 1
                        extra_inputs.append(tensor)
                        new_inputs[i][k] = tensor
                        # logger.debug(f'add {tensor.name} into inputs')
                        break
    offset = 0
    for inp in new_inputs[0]:
        if isinstance(inp, gs.Constant):
            offset += 1
        else:
            break
    
    new_nodes = []
    for i, node in enumerate(nodes):
        new_node = gs.Node(op=node.op, name=node.name, attrs=node.attrs, inputs=new_inputs[i], outputs=new_outputs[i])
        new_nodes.append(new_node)
    graph = gs.Graph(nodes=new_nodes, inputs=new_inputs[0][offset:len(globvar.onnx_graph.inputs)+offset]+extra_inputs, outputs=new_outputs[-1])
    graph.cleanup().toposort()
    
    return graph


def shape_inference(onnx_model, input_shape=None):
    # create ort inputs
    # get input shape
    if input_shape is None:
        input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
        image_shape = [x.dim_value for x in input_shape]
    else:
        image_shape = input_shape
    
    # extend outputs and inputs
    for node in onnx_model.graph.node:
        for output in node.output:
            onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    
    ort_session = ort.InferenceSession(onnx_model.SerializeToString())
    ort_inputs = {}
    
    # transform 0 num to 1 dim
    image_shape_new = []
    
    for x in image_shape:
        if x == 0:
            image_shape_new.append(1)
        else:
            image_shape_new.append(x)
    image_shape = image_shape_new
    
    # construct input image
    img = np.array(np.random.random(image_shape), dtype = np.float32)
    for i, input_ele in enumerate(ort_session.get_inputs()):
        ort_inputs[input_ele.name] = img
    
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)
    shape_dict = {}
    shape_dict['input'] = tuple(image_shape)
    offset = 0
    for i, node in enumerate(onnx_model.graph.node):
        # print(f"{node.name}: {ort_outs[i+1].shape}")
        if node.op_type == "Split":
            shape_dict[node.name] = [
                ort_outs[i+1+offset].shape,
                ort_outs[i+2+offset].shape,
                ort_outs[i+3+offset].shape
            ]
            offset += 2
        else:
            shape_dict[node.name] = ort_outs[i+1+offset].shape
    
    assert len(onnx_model.graph.node)+offset+1 == len(ort_outs)

    return shape_dict


