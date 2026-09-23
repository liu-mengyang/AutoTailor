# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Code source: https://github.com/microsoft/Moonlit/blob/main/ElasticViT/process.py
import copy
import math
import operator
import time

import dill as pickle
from loguru import logger
import torch
from torch.autograd.profiler import record_function
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from torch.distributed import get_world_size, all_reduce, barrier
import torchvision.transforms as transforms

import autotailor.tir.globvar as globvar
from autotailor.trainer.utils import get_current_memory_usage, GiB_bytes
from autotailor.trainer.branch_util import MultiBranchForwardBackward
from .ft_utils import AverageMeter, calib_bn
from .my_random_resize_crop import MyRandomResizedCrop

__all__ = ['train_one_epoch', 'validate', 'PerformanceScoreboard']


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



def bn_cal(model, train_loader, args, num_batches=100, mixup_fn=None):
    model.eval()

    for _, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.training = True
            module.momentum = None

            module.reset_running_stats()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx > num_batches:
            break

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)

        model(inputs)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_meter(meter, loss, acc1, acc5, size, batch_time, world_size):

    barrier()
    r_loss = loss.clone().detach()
    all_reduce(r_loss)
    r_loss /= world_size

    meter['loss'].update(r_loss.item(), size)
    meter['batch_time'].update(batch_time)


def teacher_inference(teacher_model, inputs, tailor=None, T=0.2):
    with torch.no_grad():
        if isinstance(teacher_model, list):
            teacher_outputs_0 = (teacher_model[0](inputs) / T).softmax(dim=-1)
            teacher_outputs_1 = (teacher_model[1](inputs) / T).softmax(dim=-1)
            teacher_outputs = teacher_outputs_0 / 2. + teacher_outputs_1 / 2.
        elif tailor is not None:
            # update resolution
            supercode = teacher_model.supercode
            supercode["resolution"] = inputs.shape[-1]
            teacher_model.transform(supercode)
            teacher_model.tir.build()
            t_model = teacher_model.tir.torch_nn
            t_model.eval()
            teacher_outputs = t_model(inputs).softmax(dim=-1)
        else:
            teacher_outputs = teacher_model(inputs).softmax(dim=-1)

    return teacher_outputs


def compute_dist_loss(labels, outputs, criterion, teacher_outputs=None, distill_criterion=None, multi_teachers=False, ALPHA=.5):
    distill_loss = None
    if teacher_outputs is not None and distill_criterion:
        distill_loss = distill_criterion(outputs, teacher_outputs)

    if multi_teachers:
        return distill_loss
    else:
        if distill_loss:
            return ALPHA * criterion(outputs, labels) + (1-ALPHA) * distill_loss
        else:
            return criterion(outputs, labels)


def train_subnets_one_epoch(tailor,
                            subnets_configs,
                            train_loader,
                            criterion,
                            optimizer,
                            dynamic_batch_size,
                            lr_scheduler,
                            epoch,
                            pymonitor,
                            args,
                            distillation_loss,
                            teacher_model,
                            mixup_fn,
                            sampling_strategy=None,
                            dataset_path="/datasets/imagene",
                            record_one_epoch=False,
                            hard_distillation=False,
                            force_random=False,
                            sandwich=False,
                            prof=None,
                            prof_ending=25,
                            enable_caching=True,
                            enable_overlapping=True,
                            enable_half_overlapping=False,
                            sharing_computation=False,
                            disable_rescheduling=False,
                            branch_samples=None,
                            caches=None):
    MyRandomResizedCrop.EPOCH = epoch
    meters = {
        'loss': AverageMeter(),
        'batch_time': AverageMeter()
    }

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size

    steps_per_epoch = math.ceil(total_sample / batch_size)
    steps_per_epoch = torch.tensor(steps_per_epoch).to(args.device)
    all_reduce(steps_per_epoch)
    steps_per_epoch = int(steps_per_epoch.item() // get_world_size())

    if args.rank == 0:
        logger.info(f'Training: {total_sample} samples ({batch_size} per mini-batch) with {sampling_strategy} strategy')

    if teacher_model:
        if isinstance(teacher_model, list):
            for tm in teacher_model:
                tm.eval()
        if isinstance(teacher_model, type(tailor)):
            teacher_model.tir.torch_nn.eval()
        else:
            teacher_model.eval()

    teacher_outputs = None
    multi_teachers = isinstance(teacher_model, list)

    resolution = subnets_configs[str(epoch)]['0']['0']["resolution"]
    lst = []
    for i in range(len(subnets_configs[str(epoch)])):
        resolution = subnets_configs[str(epoch)][str(i)]['0']['resolution']
        lst.append(resolution)
    MyRandomResizedCrop.IMAGE_SIZE_LIST = lst


    next_model = None
    next_branch_sample = None
    next_sample = None
    for batch_idx, (original_inputs, original_targets) in enumerate(train_loader):
        if prof:
            logger.info("Profiling next batch")
        start_time = time.time()
        MyRandomResizedCrop.BATCH = batch_idx + 1
        loss_of_subnets = []
        if prof:
            with record_function("## data movement ##"):
                original_inputs = original_inputs.to(args.device)
                original_targets = original_targets.to(args.device)
                mem_consumed = get_current_memory_usage(f'cuda:{dist.get_rank()}')/GiB_bytes
                logger.info(f"Move tensors took {mem_consumed:.4f} GiB")
        else:
            original_inputs = original_inputs.to(args.device)
            original_targets = original_targets.to(args.device)

        # manually zero out super gradients
        if prof:
            # TODO: use optimizer zero_grad
            with record_function("## zero out gradients ##"):
                for param_dict in globvar.super_weights.values():
                    for param in param_dict.values():
                        if isinstance(param, nn.Parameter):
                            param.grad.zero_()
            with record_function("## data augment ##"):
                if mixup_fn is not None:
                    inputs, targets = original_inputs.clone(), original_targets.clone()
                    inputs, targets = mixup_fn(inputs, targets)
                else:
                    inputs, targets = original_inputs, original_targets

                tmp_batch = inputs.shape[0]
                if inputs.shape[0] != batch_size:
                    # Fix bs related tir updating problem
                    tiled_tensor = inputs.repeat(batch_size//inputs.shape[0]+1, 1, 1, 1)

                    inputs = tiled_tensor[:batch_size]
            if teacher_model:
                with record_function("## teacher inference ##"):
                    if isinstance(teacher_model, type(tailor)):
                        teacher_outputs = teacher_inference(teacher_model=teacher_model, inputs=inputs, tailor=tailor)
                    else:
                        teacher_outputs = teacher_inference(teacher_model=teacher_model, inputs=inputs)
        else:
            # TODO: use optimizer zero_grad
            for param_dict in globvar.super_weights.values():
                for param in param_dict.values():
                    if isinstance(param, nn.Parameter):
                        param.grad.zero_()
            if mixup_fn is not None:
                inputs, targets = original_inputs.clone(), original_targets.clone()
                inputs, targets = mixup_fn(inputs, targets)
            else:
                inputs, targets = original_inputs, original_targets

            tmp_batch = inputs.shape[0]
            if inputs.shape[0] != batch_size:
                # Fix bs related tir updating problem
                tiled_tensor = inputs.repeat(batch_size//inputs.shape[0]+1, 1, 1, 1)

                inputs = tiled_tensor[:batch_size]
            if teacher_model:
                if isinstance(teacher_model, type(tailor)):
                    teacher_outputs = teacher_inference(teacher_model=teacher_model, inputs=inputs, tailor=tailor)
                else:
                    teacher_outputs = teacher_inference(teacher_model=teacher_model, inputs=inputs)

        num_updates = 0
        if not sharing_computation:
            next_model = None
        torch.autograd.set_detect_anomaly(True)
        if sharing_computation:
            # TODO: Currently not support for sandwich and only support for timm_resnet50
            if branch_samples is not None:
                branch_config = branch_samples[str(epoch)][str(batch_idx)]
                samples = branch_config["subnets"]
                dynamic_batch_size = branch_config["num_heads"]
            else:
                samples = subnets_configs[str(epoch)][str(batch_idx)]
                share_stage_id, share_block_id, branch_stage_id, branch_block_id = generate_branch_configs(tailor, samples)
                branch_config = {
                    "share_stage_id": share_stage_id,
                    "share_block_id": share_block_id,
                    "branch_stage_id": branch_stage_id,
                    "branch_block_id": branch_block_id,
                    "num_heads": len(samples)
                }
                dynamic_batch_size = len(samples)

            build_st = time.time()
            for i in range(branch_config["num_heads"]):
                code = samples[str(i)]
                code["resolution"] = lst[batch_idx]
                tailor.transform(code)
                model = tailor.tir.build(
                    branching=True,
                    share_stage_id=branch_config["share_stage_id"],
                    share_block_id=branch_config["share_block_id"],
                    branch_stage_id=branch_config["branch_stage_id"],
                    branch_block_id=branch_config["branch_block_id"],
                    branching_id=i,
                    num_heads=branch_config["num_heads"],
                    cache=caches
                )

            model.zero_grad(set_to_none=False)
            backbone = model.backbone
            heads = model.heads
            if not model.no_backbone:
                backbone.to(f"cuda:{dist.get_rank()}")
                ddp_backbone = DistributedDataParallel(backbone, device_ids=[args.local_rank])
            ddp_heads = []
            for head in heads:
                head.to(f"cuda:{dist.get_rank()}")
                ddp_head = DistributedDataParallel(head, device_ids=[args.local_rank])
                ddp_heads.append(ddp_head)
            if not disable_rescheduling:
                # Forward backbone
                backbone_outputs = []
                head_outputs = []
                output_backbone = None
                if not model.no_backbone:
                    output_backbone = ddp_backbone(inputs)
                    for i, head in enumerate(ddp_heads):
                        output_backbone_detached = copy.deepcopy(output_backbone.detach().requires_grad_(True))
                        backbone_outputs.append(output_backbone_detached)
                        head_output = head(backbone_outputs[i])
                        head_outputs.append(head_output)
                        branch_loss = compute_dist_loss(labels=targets,
                                                    outputs=head_output,
                                                    teacher_outputs=teacher_outputs,
                                                    criterion=criterion,
                                                    distill_criterion=distillation_loss,
                                                    multi_teachers=multi_teachers)
                        branch_loss.backward()
                else:
                    for head in ddp_heads:
                        head_output = head(inputs)
                        head_outputs.append(head_output)
                        branch_loss = compute_dist_loss(labels=targets,
                                                    outputs=head_output,
                                                    teacher_outputs=teacher_outputs,
                                                    criterion=criterion,
                                                    distill_criterion=distillation_loss,
                                                    multi_teachers=multi_teachers)
                        branch_loss.backward()  
            else:
                # Forward
                if prof:
                    forward_total_t = 0
                    torch.cuda.synchronize()
                    forward_st = time.time()
                # TODO: Support for disable_rescheduling
                backbone_outputs = []
                output_backbone = None
                if not model.no_backbone:
                    output_backbone = ddp_backbone(inputs)
                    for i, head in enumerate(ddp_heads):
                        output_backbone_detached = copy.deepcopy(output_backbone.detach().requires_grad_(True))
                        backbone_outputs.append(output_backbone_detached)
                        head_output = head(backbone_outputs[i])
                        branch_loss = compute_dist_loss(labels=targets,
                                                    outputs=head_output,
                                                    teacher_outputs=teacher_outputs,
                                                    criterion=criterion,
                                                    distill_criterion=distillation_loss,
                                                    multi_teachers=multi_teachers)
                        branch_loss.backward()
                else:
                    for head in ddp_heads:
                        head_output = head(inputs)
                        # if args.rank == 0:
                        #     print(head_output)
                        branch_loss = compute_dist_loss(labels=targets,
                                                    outputs=head_output,
                                                    teacher_outputs=teacher_outputs,
                                                    criterion=criterion,
                                                    distill_criterion=distillation_loss,
                                                    multi_teachers=multi_teachers)
                        branch_loss.backward()
                if prof:
                    torch.cuda.synchronize()
                    forward_t = time.time() - forward_st
                    forward_total_t += forward_t
            torch.cuda.synchronize()
                    
            if prof:
                torch.cuda.synchronize()
                backward_st = time.time()
            # Backward backbone
            if not model.no_backbone:
                ## sum up inter gradients, almost zero cost
                sum_of_grad = None
                for i in range(dynamic_batch_size):
                    if sum_of_grad is None:
                        sum_of_grad = backbone_outputs[i].grad
                    else:
                        sum_of_grad += backbone_outputs[i].grad
                assert output_backbone.shape == sum_of_grad.shape, \
                    f"Shape mismatch: output_backbone has shape {output_backbone.shape} " \
                    f"but sum_of_grad has shape {sum_of_grad.shape}"
                torch.autograd.backward(tensors=output_backbone, grad_tensors=sum_of_grad)

            if prof:
                torch.cuda.synchronize()
                backward_t = time.time()-backward_st
                backward_total_t += backward_t
                if enable_overlapping:
                    extra_str = "+build"
                else:
                    extra_str = ""
                logger.info(f"Backward{extra_str} finished in {backward_total_t} s")
            # logger.info("Backbone backwarded")
            # print(branch_config)
            if prof:
                torch.cuda.synchronize()
                update_st = time.time()
            branch_stages = []
            branch_lengths = []
            for i in range(dynamic_batch_size):
                branch_lengths.append(len(model.heads[i].stages))
            for i in range(dynamic_batch_size):
                branch_stages.append(max(branch_lengths) - branch_lengths[i])
            # update gradient of super weight
            # transform and build subnets and update them subnet by subnet
            if caches is not None:
                for i in range(dynamic_batch_size):
                    code = samples[str(i)]
                    code["resolution"] = lst[batch_idx]
                    tailor.transform(code)
                    model = tailor.tir.build(
                        branching=True,
                        share_stage_id=branch_config["share_stage_id"],
                        share_block_id=branch_config["share_block_id"],
                        branch_stage_id=branch_config["branch_stage_id"],
                        branch_block_id=branch_config["branch_block_id"],
                        branching_id=i,
                        num_heads=branch_config["num_heads"],
                        cache=caches
                    )
                    tailor.tir.update_grad()
            else:
                tailor.tir.update_grad()
            if prof:
                torch.cuda.synchronize()
                logger.info(f"Update finished in {time.time()-update_st} s")

            update_meter(meters, branch_loss, None, None, inputs.size(0), time.time() - start_time, args.world_size)

        else:
            for i in range(dynamic_batch_size):
                start_time = time.time()

                if not enable_overlapping:
                    subnet_code = subnets_configs[str(epoch)][str(batch_idx)][str(i)]
                    tailor.transform(subnet_code)
                    model = tailor.tir.build(cache=caches)
                elif enable_overlapping:
                    if next_model:
                        model = next_model
                    else:
                        # first model in this iteration
                        # TODO: solve conflict with pointer-based update gradient
                        subnet_code = subnets_configs[str(epoch)][str(batch_idx)][str(i)]
                        tailor.transform(subnet_code)
                        model = tailor.tir.build(cache=caches)
                    if (i-1)!=len(subnets_configs[str(epoch)][str(batch_idx)]):
                        next_subnet_code = subnets_configs[str(epoch)][str(batch_idx)][str(i-1)]
                        tailor.transform(next_subnet_code)
                else:
                    tailor.transform(subnet_code)
                    model = tailor.tir.build(cache=caches)

                if prof:
                    with record_function("## model movement ##"):
                        model.to(f"cuda:{dist.get_rank()}")
                    mem_consumed = get_current_memory_usage(f'cuda:{dist.get_rank()}')/GiB_bytes
                    logger.info(f"Move one subnet took {mem_consumed:.4f} GiB")
                else:
                    model.to(f"cuda:{dist.get_rank()}")
                if prof:
                    with record_function("## DDP wrapping ##"):
                        model = DistributedDataParallel(
                            model, device_ids=[args.local_rank])
                    mem_consumed = get_current_memory_usage(f'cuda:{dist.get_rank()}')/GiB_bytes
                    logger.info(f"Wrap one subnet as DDP took {mem_consumed:.4f} GiB")
                else:
                    model = DistributedDataParallel(
                        model, device_ids=[args.local_rank])
                model.train()

                if prof:
                    with record_function("## Forward ##"):
                        outputs = model(inputs)[:tmp_batch]
                        if enable_half_overlapping and enable_overlapping and i!=1 and (i-1)!=len(subnets_configs[str(epoch)][str(batch_idx)]):
                            tailor.transform(next_subnet_code)
                            tailor.tir.build(cache=caches, half_start=True)
                else:
                    outputs = model(inputs)[:tmp_batch]

                    if enable_half_overlapping and enable_overlapping and i!=1 and (i-1)!=len(subnets_configs[str(epoch)][str(batch_idx)]):
                        tailor.transform(next_subnet_code)
                        tailor.tir.build(cache=caches, half_start=True)

                loss = None
                loss = compute_dist_loss(labels=targets, outputs=outputs, teacher_outputs=teacher_outputs, criterion=criterion, distill_criterion=distillation_loss,
                                            multi_teachers=multi_teachers)

                if loss is None:
                    raise NotImplementedError

                loss_of_subnets.append(loss)
                if prof:
                    with record_function("## Backward ##"):
                        loss.backward() # DDP sync

                        # skip for mininet and the last net
                        if enable_overlapping and i!=1 and (i-1)!=len(subnets_configs[str(epoch)][str(batch_idx)]):
                            if enable_half_overlapping:
                                next_model = tailor.tir.build(cache=caches, half_end=True)
                            else:
                                tailor.transform(next_subnet_code)
                                next_model = tailor.tir.build(cache=caches)
                        mem_consumed = get_current_memory_usage(f'cuda:{dist.get_rank()}')/GiB_bytes
                        logger.info(f"Train one subnet took {mem_consumed:.4f} GiB")
                else:
                    loss.backward() # DDP sync

                    # skip for mininet and the last net
                    if enable_overlapping and i!=1 and (i-1)!=len(subnets_configs[str(epoch)][str(batch_idx)]):
                        if enable_half_overlapping:
                            next_model = tailor.tir.build(cache=caches, half_end=True)
                        else:
                            tailor.transform(next_subnet_code)
                            next_model = tailor.tir.build(cache=caches)

                # Update gradient of super weights
                if prof:
                    with record_function("## Gradient update ##"):
                        tailor.tir.update_grad()
                else:
                    tailor.tir.update_grad()
                # set grads on wrapped model to none to avoid repeated gradient aggregation
                model.zero_grad(set_to_none=False)

                update_meter(meters, loss, None, None, inputs.size(0), time.time() - start_time, args.world_size)

        # Update super weights
        optimizer.step()

        num_updates += 1

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                num_updates=num_updates, metric=meters['loss'].avg)

        if args.rank == 0 and (batch_idx + 1) % args.print_freq == 0:
            pymonitor.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                'Loss': meters['loss'],
                'BatchTime': meters['batch_time'],
                'LR': optimizer.param_groups[0]['lr'],
                'GPU memory': round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            })
            logger.info(
                "--------------------------------------------------------------------------------------------------------------")

        if prof:
            if batch_idx == prof_ending:
                return

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    if 'top1' in meters.keys():
        return meters['top1'].avg, meters['top5'].avg, meters['loss'].avg
    else:
        return meters['loss'].avg


def validate(tailor, data_loader, code, criterion, args, existed_model=None, cache=None):
    meters = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    }

    batch_size = data_loader.batch_size
    logger.info(f"Validating with {batch_size} batch size and {code['resolution']} resolution")
    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(
            256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(code["resolution"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if existed_model:
        model = existed_model
    else:
        tailor.transform(code)
        model = tailor.tir.build(cache=cache)
        model.cuda()

    calib_bn(model, args.path, code["resolution"], batch_size)
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            tmp_batch = inputs.shape[0]
            # Tiling for the last batch to avoid TailorIR inference error
            if inputs.shape[0] != batch_size:
                # Fix bs related tir updating problem
                tiled_tensor = inputs.repeat(batch_size//inputs.shape[0]+1, 1, 1, 1)

                # Slice the tensor to the desired shape [128, 3, 256, 256]
                inputs = tiled_tensor[:batch_size]

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            start_time = time.time()

            outputs = model(inputs)[:tmp_batch]
            loss = None
            if criterion:
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            if criterion:
                update_meter(meters, loss, acc1, acc5, inputs.size(0),
                             time.time() - start_time, args.world_size)
            else:
                meters["top1"].update(acc1.item(), inputs.size(0))
                meters["top5"].update(acc5.item(), inputs.size(0))

    return meters['top1'].avg, meters['top5'].avg, meters['loss'].avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info(f"Scoreboard best {idx + 1} ==> Epoch [{score['epoch']}][Top1: {score['top1']}   Top5: {score['top5']}]")

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch