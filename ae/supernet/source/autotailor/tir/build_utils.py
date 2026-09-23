import os
import sys

import torch
import onnx
import onnx_graphsurgeon as gs

from .graph_utils import extract_sub_graph, get_node_from_name

sys.path.append(os.getenv('AUTOTAILOR_HOME'))

from onnx2torch import convert
import autotailor.tir.globvar as globvar


def load_params_from_static(dynamic_module, static_module):
    new_kvpair=dynamic_module.state_dict()
    count = 0
    for key, value in new_kvpair.items():
        layer_name, weights = list(static_module.state_dict().items())[count]
        new_kvpair[key] = weights
        count += 1

    dynamic_module.load_state_dict(new_kvpair)


def generate_torch_block(sub_graph_nodes_names):
    model_name = globvar.model_name
    gs_graph_block = extract_sub_graph(sub_graph_nodes_names)
    names = []
    for name in sub_graph_nodes_names:
        names.append(name.replace("/", "|"))
    save_path = f"models/onnx/{model_name}/{'_'.join(names)}.onnx"
    os.makedirs(f"models/onnx/{model_name}", exist_ok=True)
    # print(names)
    # print(gs_graph_block)
    if not os.path.exists(save_path):
        onnx.save(gs.export_onnx(gs_graph_block), save_path)
    torch_block = convert(save_path)
    return torch_block


def sub_filter_start_end(kernel_size, sub_kernel_size):
    if kernel_size == sub_kernel_size:
        return 0, kernel_size
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def get_padding(ks, s, hw):
    """ choose padding value to make sure:
    if s = 1, out_hw = in_hw;
    if s = 2, out_hw = ceil(in_hw / 2);
    if s = 4, out_hw = ceil(in_hw / 4);

    (in_hw - ks + 2p) // s + 1 = out_hw
    p = ((out_hw - 1) * s - in_hw + ks) // 2
    """
    hw = int(hw)
    if hw % s == 0:
        pad = max(ks - s, 0)
    else:
        pad = max(ks - (hw % s), 0)
    if pad % 2 == 0:
        return pad // 2
    else:
        return pad // 2 + 1
