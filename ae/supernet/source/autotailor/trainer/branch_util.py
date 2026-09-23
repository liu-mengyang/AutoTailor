import copy

import torch


def generate_branch_configs(tailor, subnets_config):
    # create tirs
    tirs = []
    for subnet_config in list(subnets_config.values()):
        subnet_config["resolution"] = 224
        tailor.transform(subnet_config)
        # print(subnet_config)
        tirs.append(copy.deepcopy(tailor.tir))
    # compare tirs block by block
    branch_stage_id = 0
    branch_block_id = 0
    share_stage_id = -1
    share_block_id = -1
    find_flag = False
    for stage_id, stage in tailor.tir.stages.items():
        for block_id, block in stage.flow.items():
            feature_temp = None
            for tir_id, tir_temp in enumerate(tirs):
                block_temp = tir_temp.stages[stage_id].flow[block_id]
                if tir_id == 0:
                    if block_temp.is_active():
                        # use first feature as comparison baseline
                        # if is not active, keep None as baseline
                        feature_temp = block_temp.features
                else:
                    # comparing with baseline
                    # print(f"Comp {tir_id}")
                    # print(block_temp.features)
                    # print(feature_temp)
                    if block_temp.features != feature_temp:
                        # find branching out point
                        branch_stage_id = stage_id
                        branch_block_id = block_id
                        find_flag = True
                        # print(f"Find {stage_id}-{block_id}")
                        break
                    # print(f"Pass {stage_id}-{block_id}")
            if find_flag:
                break
            else:
                # to this pass this comparison and update share id
                share_stage_id = stage_id
                share_block_id = block_id
        if find_flag:
            break
    return share_stage_id, share_block_id, branch_stage_id, branch_block_id


class MultiBranchForwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data, backbone, heads, head_id=None, no_backbone=False):
        ctx.save_for_backward(input_data)
        if head_id is None:
            ys = []
            # normal sequential forward
            if not no_backbone:
                # Forward backbone
                output_backbone = backbone(input_data)

                # Save for backward pass
                ctx.output_backbone = output_backbone
                ctx.save_for_backward(output_backbone)

                # Forward heads
                for i, head in enumerate(heads):
                    output_head = head(output_backbone)
                    ys.append(output_head)
                    # Save for backward pass
                    ctx.save_for_backward(output_head)
            else:
                # No backbone
                for i, head in enumerate(heads):
                    output_head = head(input_data)
                    ys.append(output_head)
            return tuple(ys)
        elif head_id == -1:
            # Forward backbone
            if not no_backbone:
                output_backbone = backbone(input_data)

                # Save for backward pass
                ctx.save_for_backward(output_backbone)

            return None
        else:
            # Forward specific head
            if not no_backbone:
                output_backbone, = ctx.output_backbone
                y = heads[head_id](output_backbone)
            else:
                y = heads[head_id](input_data)
            return y

    @staticmethod
    def backward(ctx, heads_num, grad_output, head_id=None, no_backbone=False):
        # Retrieve saved tensors
        input_data = ctx.saved_tensors[0]
        if not no_backbone:
            output_backbone = ctx.saved_tensors[1]

        if head_id is None:
            # Normal backward for all heads
            grad_outputs = []
            for i in range(heads_num):
                if no_backbone:
                    # No backbone, directly use input_data
                    grad_output_head = torch.autograd.grad(
                        outputs=ctx.saved_tensors[i+1],
                        inputs=input_data,
                        grad_outputs=grad_output[i],
                    )
                else:
                    # Use output from backbone
                    grad_output_head = torch.autograd.grad(
                        outputs=ctx.saved_tensors[i+2],
                        inputs=output_backbone,
                        grad_outputs=grad_output[i],
                    )
                grad_outputs.append(grad_output_head[0])
            return tuple(grad_outputs)
        elif head_id == -1:
            # Backward for backbone only
            if not no_backbone:
                grad_output_backbone = torch.autograd.grad(
                    outputs=output_backbone,
                    inputs=input_data,
                    grad_outputs=grad_output,
                )
                return grad_output_backbone
            else:
                raise NotImplementedError("No backbone specified, cannot compute gradients for input_data.")
        else:
            # Backward for specific head
            if no_backbone:
                # No backbone, directly use input_data
                grad_output_head = torch.autograd.grad(
                    outputs=ctx.saved_tensors[head_id+1],
                    inputs=input_data,
                    grad_outputs=grad_output,
                )
            else:
                # Use output from backbone
                grad_output_head = torch.autograd.grad(
                    outputs=ctx.saved_tensors[head_id+2],
                    inputs=output_backbone,
                    grad_outputs=grad_output,
                )
            return grad_output_head[0]
