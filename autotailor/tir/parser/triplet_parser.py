import onnx_graphsurgeon as gs

from .qkv_parser import parse_qkv

__all__ = ["parse_triplet_block"]


def parse_triplet_block(cur_node: gs.Node, supernet_cfg_dict: dict):
    block, last_node = parse_qkv(cur_node)
    return block, last_node