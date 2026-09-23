import copy

from autotailor.tir.stage import Stage
from autotailor.tir.blocks import LNOp, LinearMatMulOp, FFNBlock, ResidualBlock, AttentionBlock
import autotailor.tir.globvar as globvar


__all__ = ["parse_ffn"]


def parse_ffn(
    stage: Stage,
    supernet_cfg_dict: dict,
    inp_shape: tuple
):
    new_stage = Stage()
    for block_id, block in stage.flow.items():
        if isinstance(block, ResidualBlock) and not isinstance(block, AttentionBlock):
            meet_ln = False
            meet_matmul = False
            for op in block.main_path:
                if isinstance(op, LNOp):
                    meet_ln = True
                elif isinstance(op, LinearMatMulOp):
                    meet_matmul = True
                if meet_ln and meet_matmul:
                    ffnblock = FFNBlock(block, supernet_cfg_dict)

                    main_path_temp = ffnblock.main_path
                    main_path = []
                    tmp_matmul = None
                    for i, node in enumerate(main_path_temp):
                        # maunal fuse matmul and biasadd to linear
                        if node.type == "LinearMatMul":
                            tmp_matmul = node
                        elif node.type == "BiasAdd":
                            # print("Fuse matmul and biasadd")
                            tmp_matmul.features["has_bias"] = True
                            globvar.super_weights[tmp_matmul.name]["bias"] = copy.deepcopy(globvar.super_weights[node.name]["bias"])
                            globvar.super_weights.pop(node.name)
                            tmp_matmul.super_bias = globvar.super_weights[tmp_matmul.name]["bias"]
                            main_path.append(tmp_matmul)
                        else:
                            main_path.append(node)
                    ffnblock.main_path = main_path
                    new_stage.add_block(ffnblock)
                    break
            if not (meet_ln and meet_matmul):
                new_stage.add_block(block)
        else:
            new_stage.add_block(block)
    
    # Update shape information
    for block_id, block in new_stage.flow.items():
        block.update({"in_shape": inp_shape})
        inp_shape = block.features['out_shape']
    
    return new_stage
