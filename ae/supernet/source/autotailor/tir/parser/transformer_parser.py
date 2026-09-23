import copy

from autotailor.tir.stage import Stage
from autotailor.tir.blocks import ResidualBlock, AttentionBlock, FFNBlock, TransformerBlock


__all__ = ["parse_transformer"]


def parse_transformer(
    stage: Stage,
    supernet_cfg_dict: dict,
    inp_shape: tuple
) -> TransformerBlock:
    new_stage = Stage()
    
    block_queue = []
    
    meet_attn = False
    attn_block = None
    
    for block_id, block in stage.flow.items():
        if isinstance(block, AttentionBlock):
            meet_attn = True
            attn_block = block
        elif meet_attn and isinstance(block, FFNBlock):
            transformer_block = TransformerBlock(attn_block, block)
            new_stage.add_block(transformer_block)
            meet_attn = False
            attn_block = None
        elif meet_attn:
            meet_attn = False
            new_stage.add_block(copy.deepcopy(attn_block))
            attn_block = None
            new_stage.add_block(block)
        else:
            new_stage.add_block(block)
    
    # Update shape information
    for block_id, block in new_stage.flow.items():
        block.update({"in_shape": inp_shape})
        inp_shape = block.features['out_shape']
    
    return new_stage