from copy import deepcopy

from autotailor.tir.stage import Stage
from autotailor.tir.blocks import LinearMatMulOp, BiasAddOp

__all__ = ["parse_linearmatmul"]


def parse_linearmatmul(stage: Stage,
             supernet_cfg_dict: dict,
             inp_shape: tuple
) -> Stage:
    """Find all mb blocks in the stage.
    
    Strategy to parse: Detect two continuous conv op block and the first one is
      depth conv and the second one is point conv.
    
    Args:
      stage: the original stage with potential bottleneck blocks.
      inp_shape: the shape of input feature.
      
    Returns:
      A new stage with bottleneck blocks.
    """
    new_stage = Stage()
    matmul_tmp = None
    
    for block_id, block in stage.flow.items():
        if isinstance(block, LinearMatMulOp):
            if matmul_tmp is not None:
                new_stage.add_block(deepcopy(matmul_tmp))
            matmul_tmp = block
        elif isinstance(block, BiasAddOp) and matmul_tmp is not None:
            # print("Fuse matmul and biasadd")
            matmul_tmp.features["has_bias"] = True
            matmul_tmp.super_bias = block.super_bias
            new_stage.add_block(deepcopy(matmul_tmp))
            matmul_tmp = None
        else:
            if matmul_tmp is not None:
                new_stage.add_block(deepcopy(matmul_tmp))
                matmul_tmp = None
            new_stage.add_block(block)
    
    # Update shape information
    for block_id, block in new_stage.flow.items():
        block.update({"in_shape": inp_shape})
        inp_shape = block.features['out_shape']
    
    return new_stage