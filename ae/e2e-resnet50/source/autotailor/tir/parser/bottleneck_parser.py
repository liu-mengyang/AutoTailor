from copy import deepcopy

from autotailor.tir.stage import Stage
from autotailor.tir.blocks import ConvOp, SingleOp
from autotailor.tir.blocks import (BottleneckBlock, SEResidualBlock,
                                   ResidualBlock)

__all__ = ["parse_bottleneck"]


def parse_bottleneck(stage: Stage,
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
    bottleneck_block_queue = []
    conv_count = 0
    first_conv = None
    middle_conv = None
    
    for block_id, block in stage.flow.items():
        if isinstance(block, ConvOp):
            # Judge
            bottleneck_block_queue.append(block)
            conv_count += 1
            if (conv_count == 1 and block.features['kernel_size'] == 1 and
                block.features['out_channel'] > block.features["in_channel"]):
                first_conv = block
            elif conv_count== 2 and block.features['kernel_size'] != 1:
                middle_conv = block
            elif conv_count == 3 and block.features['kernel_size'] == 1:
                # Find
                if "BottleneckBlock" in supernet_cfg_dict["var"]["block_vars"]:
                    all_expand_ratios = supernet_cfg_dict["var"]["block_vars"]["BottleneckBlock"]["expand_ratio"]
                else:
                    all_expand_ratios = None
                expand_base_on = supernet_cfg_dict["arch"]["BottleneckBlock"]["expand_base"]
                bottleneckblock = BottleneckBlock(deepcopy(bottleneck_block_queue),
                                                  all_expand_ratios,
                                                  expand_base_on)
                new_stage.add_block(bottleneckblock)
                
                bottleneck_block_queue = []
                conv_count = 0
            else:
                # Not match, reset
                bottleneck_block_queue_temp = []
                is_alive = False
                for subblock in bottleneck_block_queue:
                    # Only queue out the first conv and then check whether alive
                    if not is_alive:
                        new_stage.add_block(subblock)
                        if isinstance(subblock, ConvOp):
                            # Handle if out a conv, it maybe alive
                            conv_count -= 1
                            assert conv_count < 3
                            if conv_count == 2:
                                assert middle_conv
                                first_conv = middle_conv
                                middle_conv = block
                                if (first_conv.features['kernel_size'] == 1 and
                                     first_conv.features['out_channel'] >
                                     first_conv.features["in_channel"] and
                                    middle_conv.features['kernel_size'] != 1):
                                    is_alive = True
                            elif conv_count == 1:
                                first_conv = block
                                if (first_conv.features['kernel_size'] == 1 and
                                     first_conv.features['out_channel'] >
                                     first_conv.features["in_channel"]):
                                    is_alive = True
                    else:
                        bottleneck_block_queue_temp.append(subblock)
                bottleneck_block_queue = bottleneck_block_queue_temp
        elif not isinstance(block, SingleOp):
            # Judge the queue for meeting block
            if conv_count > 0:
                # Meet new block, judge
                if (isinstance(block, SEResidualBlock) or
                    isinstance(block, ResidualBlock)):
                    # Skip SEResidualBlock
                    bottleneck_block_queue.append(block)
                    continue
                else:
                    # Reset by clear
                    for subblock in bottleneck_block_queue:
                        new_stage.add_block(subblock)
                    new_stage.add_block(block)
                    
                    bottleneck_block_queue = []
                    conv_count = 0
            else:
                assert len(bottleneck_block_queue) == 0
                new_stage.add_block(block)
        else:
            # Handle non-conv op.
            if conv_count > 0:
                block_copy = deepcopy(block)
                bottleneck_block_queue.append(block_copy)
            else:
                assert len(bottleneck_block_queue) == 0
                new_stage.add_block(block)

    # Clear
    if len(bottleneck_block_queue) > 0:
        for subblock in bottleneck_block_queue:
            new_stage.add_block(subblock)
    
    # Update shape information
    for block_id, block in new_stage.flow.items():
        block.update({"in_shape": inp_shape})
        inp_shape = block.features['out_shape']
    
    return new_stage