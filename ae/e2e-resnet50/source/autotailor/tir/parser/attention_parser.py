import copy

from autotailor.tir.stage import Stage
from autotailor.tir.blocks import LNOp, LinearMatMulOp, AttentionBlock, ResidualBlock, QKVBlock
import autotailor.tir.globvar as globvar


__all__ = ["parse_attention"]


def parse_attention(
    stage: Stage,
    supernet_cfg_dict: dict,
    inp_shape: tuple
) -> AttentionBlock:
    new_stage = Stage()
    for block_id, block in stage.flow.items():
        if isinstance(block, ResidualBlock):
            meet_qkv = False
            for op in block.main_path:
                # print(op.type)
                if isinstance(op, QKVBlock):
                    meet_qkv = True
                    attnblock = AttentionBlock(block)

                    main_path_temp = block.main_path
                    main_path = []
                    tmp_matmul = None
                    for i, node in enumerate(main_path_temp):
                        # maunal fuse matmul and biasadd to linear
                        if node.type == "LinearMatMul":
                            if tmp_matmul is not None:
                                main_path.append(copy.deepcopy(tmp_matmul))
                            tmp_matmul = node
                        elif node.type == "BiasAdd":
                            # print("Fuse matmul and biasadd")
                            tmp_matmul.features["has_bias"] = True
                            globvar.super_weights[tmp_matmul.name]["bias"] = copy.deepcopy(globvar.super_weights[node.name]["bias"])
                            # print(tmp_matmul.name)
                            globvar.super_weights.pop(node.name)
                            tmp_matmul.super_bias = globvar.super_weights[tmp_matmul.name]["bias"]
                            main_path.append(copy.deepcopy(tmp_matmul))
                            tmp_matmul = None
                        else:
                            if tmp_matmul is not None:
                                main_path.append(copy.deepcopy(tmp_matmul))
                                tmp_matmul = None
                            main_path.append(node)
                    attnblock.main_path = main_path
                    new_stage.add_block(attnblock)
                    break
            if not meet_qkv:
                new_stage.add_block(block)
        else:
            new_stage.add_block(block)
    
    # Update shape information
    for block_id, block in new_stage.flow.items():
        block.update({"in_shape": inp_shape})
        inp_shape = block.features['out_shape']
    
    return new_stage
