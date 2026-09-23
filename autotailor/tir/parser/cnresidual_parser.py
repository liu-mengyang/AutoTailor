import onnx_graphsurgeon as gs

from autotailor.tir.blocks import DepthConvOp, LinearMatMulOp
from autotailor.tir.blocks import CNResidualBlock, ConvResidualBlock
from autotailor.tir.stage import Stage

__all__ = ["parse_cnresidual"]


def parse_cnresidual(
    stage: Stage,
    supernet_cfg_dict: dict,
    inp_shape: tuple
) -> Stage:
    """Find all bottleneck blocks in the stage.
    
    Strategy to parse: Detect three continuous conv op block and they obbey the
      following rules: the kernel size of the first conv is 1, the kernel size
      of the second conv is not 1 , the kernel size of the last conv is 1, and
      the width of the second conv is the largest.
    
    Args:
      stage: the original stage with potential bottleneck blocks.
      supernet_cfg_dict: the dictionary of supernet configuration.
      inp_shape: the shape of input feature.
      
    Returns:
      A new stage with bottleneck blocks.
    """
    new_stage = Stage()
    
    for block_id, block in stage.flow.items():
        if isinstance(block, ConvResidualBlock):
            dw_conv = None
            first_matmul = None
            second_matmul = None
            main_path = block.main_path
            for node in main_path:
                if isinstance(node, DepthConvOp):
                    if dw_conv is None:
                        dw_conv = node
                    else:
                        dw_conv = None
                elif isinstance(node, LinearMatMulOp):
                    if first_matmul is None:
                        first_matmul = node
                    elif second_matmul is None:
                        second_matmul = node
                    else:
                        first_matmul = None
                        second_matmul = None
                
            if (dw_conv and first_matmul and second_matmul):
                first_node = block.start_node
                last_node = block.last_node
                first_path_tmp = block.main_path
                first_path = []
                tmp_matmul = None
                for i, node in enumerate(first_path_tmp):
                    # maunal fuse matmul and biasadd to linear
                    if node.type == "LinearMatMul":
                        tmp_matmul = node
                    elif node.type == "BiasAdd":
                        tmp_matmul.features["has_bias"] = True
                        tmp_matmul.super_bias = node.super_bias
                        first_path.append(tmp_matmul)
                    else:
                        first_path.append(node)
                second_path = block.residual_path
                cnblock = CNResidualBlock(first_node,
                                            last_node,
                                            first_path,
                                            second_path,
                                            supernet_cfg_dict)
                new_stage.add_block(cnblock)
            else:
                new_stage.add_block(block)
        else:
            new_stage.add_block(block)
        
    # Update shape information
    for block_id, block in new_stage.flow.items():
        block.update({"in_shape": inp_shape})
        inp_shape = block.features['out_shape']
    
    return new_stage