from copy import deepcopy

from autotailor.tir.stage import Stage
from autotailor.tir.blocks import DepthConvOp, ConvOp, SingleOp
from autotailor.tir.blocks import MBBlock

__all__ = ["parse_mb"]


def parse_mb(stage: Stage,
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
    mb_block_queue = []
    dw_conv = None
    point_conv = None
    
    for block_id, block in stage.flow.items():
        if isinstance(block, ConvOp):
            # Judge
            mb_block_queue.append(block)
            if dw_conv is None and isinstance(block, DepthConvOp):
                dw_conv = block
            elif (dw_conv and point_conv is None and
                  block.features["kernel_size"] == 1):
                # Find
                mobilenetblock = MBBlock(mb_block_queue)
                new_stage.add_block(mobilenetblock)
                
                mb_block_queue = []
                dw_conv = None
                point_conv = None
            else:
                # Not match, reset
                mb_block_queue_temp = []
                is_alive = False
                for subblock in mb_block_queue:
                    # Only queue out the first conv and then check whether alive
                    if not is_alive:
                        new_stage.add_block(subblock)
                        if isinstance(subblock, ConvOp):
                            # Handle if out a conv, it maybe alive
                            if isinstance(block, DepthConvOp):
                                dw_conv = block
                                point_conv = None
                                is_alive = True
                    else:
                        mb_block_queue_temp.append(subblock)
                mb_block_queue = mb_block_queue_temp
        elif not isinstance(block, SingleOp):
            # Judge the queue for meeting block
            # Meet new block, clear
            # Reset by clear
            for subblock in mb_block_queue:
                new_stage.add_block(subblock)
            new_stage.add_block(block)
            
            mb_block_queue = []
            dw_conv = None
            point_conv = None
        else:
            # Handle non-conv op.
            if len(mb_block_queue) > 0:
                block_copy = deepcopy(block)
                mb_block_queue.append(block_copy)
            else:
                new_stage.add_block(block)

    # Clear
    if len(mb_block_queue) > 0:
        for subblock in mb_block_queue:
            new_stage.add_block(subblock)
    
    # Update shape information
    for block_id, block in new_stage.flow.items():
        block.update({"in_shape": inp_shape})
        inp_shape = block.features['out_shape']
    
    return new_stage