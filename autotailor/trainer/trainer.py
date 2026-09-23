import ast
import copy
import json
import os

from loguru import logger
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Mixup

import autotailor.tir.globvar as globvar
from autotailor.trainer.utils import get_current_memory_usage, GiB_bytes
from .checkpoint import save_checkpoint, load_checkpoint
from .ddp_utils import (setup_print, load_data_dist)
from .ft_utils import ProgressMonitor, PerformanceScoreboard
from .loss_ops import SoftTargetCrossEntropyNoneSoftmax
from .process import train_subnets_one_epoch, validate


class Trainer:
    def __init__(self,
                 tailor,
                 training_config,
                 image_size_list,
                 task="classify",
                 teacher_tailor=None):
        self.image_size_list = list(map(int, image_size_list))
        self.tailor = tailor
        self.task = task
        self.teacher_tailor = teacher_tailor
        self.training_config = training_config
        self.val_loader = None

    def train(self,
              subnets_configs,
              dimension_index_dict=None,
              prof=None,
              branch_samples=None,
              sharing_comp=False,
              total_op_dict=None,
              disable_rescheduling=False,
              max_num_heads=1):
        # Init tuning
        output_dir = self.training_config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        pymonitor = None
        if self.training_config.rank == 0:
            with open(f"{self.training_config.output_dir}/self.training_config.json", "w") as self.training_config_file:  # dump experiment config
                json.dump(vars(self.training_config), self.training_config_file, indent=4)
            pymonitor = ProgressMonitor(logger)

        ## Configure for DDP
        assert self.training_config.rank >= 0, 'ERROR IN RANK'
        assert self.training_config.distributed

        setup_print(is_master=self.training_config.rank == 0)

        if self.training_config.rank == 0:
            print(self.training_config)

        ## Configure for LR scheduler
        scaled_linear_lr = self.training_config.lr * dist.get_world_size() * \
            self.training_config.batch_size / 2048
        scaled_linear_min_lr = self.training_config.min_lr * \
            dist.get_world_size() * self.training_config.batch_size / 2048
        if self.training_config.warmup_lr < 0:
            self.training_config.warmup_lr = self.training_config.lr
        scaled_linear_warmup_lr = self.training_config.warmup_lr * \
            dist.get_world_size() * self.training_config.batch_size / 2048

        self.training_config.lr = scaled_linear_lr
        self.training_config.min_lr = scaled_linear_min_lr
        self.training_config.warmup_lr = scaled_linear_warmup_lr

        if dimension_index_dict:
            # Configure for progressive tuning
            self.tailor.dimension_index_dict = dimension_index_dict
            sampling_strategy = dimension_index_dict["sampling_strategy"]
        else:
            sampling_strategy = None

        # Load data
        train_loader, val_loader, _, training_sampler = load_data_dist(
            self.training_config,
            self.image_size_list,
            self.tailor.supercode["resolution"])

        # Init mixup function
        mixup_fn = None
        if self.training_config.mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=self.training_config.mixup,
                cutmix_alpha=self.training_config.cutmix,
                cutmix_minmax=self.training_config.cutmix_minmax,
                prob=self.training_config.mixup_prob,
                switch_prob=self.training_config.mixup_switch_prob,
                mode=self.training_config.mixup_mode,
                label_smoothing=self.training_config.smoothing,
                num_classes=self.training_config.num_classes)

        # Create loss function
        if self.training_config.mixup:
            criterion = SoftTargetCrossEntropyNoneSoftmax()
        else:
            criterion = LabelSmoothingCrossEntropy(self.training_config.smoothing)
        criterion = criterion.cuda()

        # First building the model and move its weights to GPU
        self.tailor.transform(self.tailor.supercode)
        super_model = self.tailor.tir.build()
        super_model = super_model.to(f"cuda:{dist.get_rank()}")

        # Create super gradients for super weights.
        # Super weights has been created and malloced memory when TailorIR
        # parsed.
        cnt_weights = 0
        for param_dict in globvar.super_weights.values():
            for param in param_dict.values():
                param.to(f"cuda:{dist.get_rank()}")
                cnt_weights += 1
                if isinstance(param, nn.Parameter):
                    param.grad = torch.zeros(param.shape, device=f"cuda:{dist.get_rank()}")
        mem_consumed = get_current_memory_usage(f'cuda:{dist.get_rank()}')/GiB_bytes
        logger.info(f"Create {cnt_weights} super gradients took {mem_consumed:.4f} GiB")
        # Rebind
        self.tailor.tir.bind_weight()
        # Create optimizer on top of super weights
        super_params = []
        for param_dict in globvar.super_weights.values():
            for param in param_dict.values():
                if isinstance(param, nn.Parameter):
                    super_params.append(param)
        logger.info(f"Length of params: {len(super_params)}")
        optimizer = create_optimizer(self.training_config, super_params)

        # Create scheduler
        lr_scheduler, num_epochs = create_scheduler(self.training_config, optimizer)

        # Auto resume
        chkp_file = self.training_config.resume_path if (self.training_config.resume_path is not None and os.path.exists(self.training_config.resume_path)) else os.path.join(output_dir, self.training_config.name + '_checkpoint.pth.tar')
        if os.path.exists(chkp_file):
            print("load checkpoint from", chkp_file)
            # super_weights, start_epoch, _ = load_checkpoint(
            #     chkp_file=chkp_file,
            #     optimizer=optimizer if not self.training_config.eval else None)

            checkpoint = torch.load(chkp_file, map_location=lambda storage, loc: storage)
            checkpoint_epoch = checkpoint.get('epoch', None)
            start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
            super_weights = checkpoint.get('state_dict', None)
            # Update super weights
            # self.tailor.tir.update_weights(super_weights)
            for param_dict_k, param_dict_v in globvar.super_weights.items():
                for param_k, param_v in param_dict_v.items():
                    if param_k == "weight" or param_k == "bias" or param_k == "scale":
                        globvar.super_weights[param_dict_k][param_k] = \
                            nn.Parameter(super_weights[param_dict_k][param_k].to(f"cuda:{dist.get_rank()}"))
                    elif param_k == "mean" or param_k == "var":
                        globvar.super_weights[param_dict_k][param_k] = \
                            super_weights[param_dict_k][param_k].to(f"cuda:{dist.get_rank()}")
                    else:
                        raise NotImplemented(f"{param_k} not supported")
            del super_weights
            torch.cuda.empty_cache()
            # Rebind
            self.tailor.tir.bind_weight()

            # Reinit super gradients, optimizer and scheduler
            cnt_weights = 0
            for param_dict in globvar.super_weights.values():
                for param in param_dict.values():
                    param.to(f"cuda:{dist.get_rank()}")
                    if isinstance(param, nn.Parameter):
                        cnt_weights += 1
                        param.grad = torch.zeros(param.shape, device=f"cuda:{dist.get_rank()}")
            super_params = []
            for param_dict in globvar.super_weights.values():
                for param in param_dict.values():
                    if isinstance(param, nn.Parameter):
                        super_params.append(param)
            optimizer = create_optimizer(self.training_config, super_params)
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler, num_epochs = create_scheduler(self.training_config,
                                                        optimizer)

            if start_epoch > 0:
                lr_scheduler.step(start_epoch)
        else:
            start_epoch = 0
            assert not self.training_config.eval

        # After autoresume, we need to rebuild the model
        caches = None
        if total_op_dict is not None:
            caches = []
            for i in range(max_num_heads):
                output_dict = {}
                for op_type, feature_dict in total_op_dict.items():
                    output_dict[op_type] = {}
                    print(f"Building {op_type} operators ...")
                    for feature_info, op in tqdm(feature_dict.items()):
                        # Build from op dict
                        op_id = None
                        ids_str = feature_info.split("(")[0]
                        start_idx = len(ids_str)
                        feature = feature_info[start_idx:]
                        split_results = ids_str.split("-")
                        if len(split_results) == 3:
                            stage_id, block_id, _ = split_results
                        elif len(split_results) == 4:
                            stage_id, block_id, op_id, _ = split_results
                        elif len(split_results) > 4:
                            stage_id = split_results[0]
                            block_id = split_results[1]
                            op_id = split_results[2:-1]
                        else:
                            raise NotImplementedError
                        feature = ast.literal_eval(feature)
                        if op_type in ["Conv", "DepthConv"]:
                            assert len(feature) == 8
                            feature_dict = {
                                "kernel_size": int(feature[0]),
                                "stride": int(feature[1]),
                                "group": int(feature[2]),
                                "has_bias": bool(feature[3]),
                                "in_shape": feature[4],
                                "in_channel": int(feature[6]),
                                "out_channel": int(feature[7])
                            }
                            out_shape = feature[5]
                        elif op_type in ["MaxPool"]:
                            assert len(feature) == 6
                            feature_dict = {
                                "kernel_size": feature[0],
                                "stride": feature[1],
                                "in_shape": feature[2],
                                "in_channel": int(feature[4]),
                                "out_channel": int(feature[5])
                            }
                            out_shape = feature[3]
                        elif op_type == "Linear":
                            assert len(feature) == 5
                            feature_dict = {
                                "has_bias": feature[0],
                                "in_shape": feature[1],
                                "in_channel": int(feature[3]),
                                "out_channel": int(feature[4])
                            }
                            out_shape = feature[2]
                        elif op_type == "LinearMatMul":
                            assert len(feature) == 5
                            feature_dict = {
                                "has_bias": feature[0],
                                "in_shape": feature[1],
                                "out_shape": feature[2],
                                "in_channel": int(feature[3]),
                                "out_channel": int(feature[4])
                            }
                            out_shape = feature[2]
                        elif op_type == "Reshape":
                            assert len(feature) == 5
                            feature_dict = {
                                "shape": feature[0],
                                "in_shape": feature[1],
                                "out_shape": feature[2],
                                "in_channel": int(feature[3]),
                                "out_channel": int(feature[4])
                            }
                            out_shape = feature[2]
                        elif op_type == "Transpose":
                            assert len(feature) == 5
                            feature_dict = {
                                "perm": feature[0],
                                "in_shape": feature[1],
                                "out_shape": feature[2],
                                "in_channel": int(feature[3]),
                                "out_channel": int(feature[4])
                            }
                            out_shape = feature[2]
                        elif op_type == "Concat":
                            assert len(feature) == 6
                            feature_dict = {
                                "axis": feature[0],
                                "extra_dims": feature[1],
                                "in_shape": feature[2],
                                "out_shape": feature[3],
                                "in_channel": int(feature[4]),
                                "out_channel": int(feature[5])
                            }
                            out_shape = feature[3]
                        elif op_type in ["BatchNormalization", "LayerNormalization", "Relu", "Add","Flatten", "GlobalAveragePool", "Clip", "Gelu", "BiasAdd", "MatMul", "Gather"]:
                            assert len(feature) == 4
                            feature_dict = {
                                "in_shape": feature[0],
                                "in_channel": int(feature[2]),
                                "out_channel": int(feature[3])
                            }
                            out_shape = feature[1]
                        elif op_type == "Mul":
                            assert len(feature) == 4
                            feature_dict = {
                                "in_shape": feature[0][0],
                                "in_channel": int(feature[2]),
                                "out_channel": int(feature[3])
                            }
                            out_shape = feature[1]
                        elif op_type == "Softmax":
                            assert len(feature) == 5
                            feature_dict = {
                                "in_shape": feature[1],
                                "in_channel": int(feature[3]),
                                "out_channel": int(feature[4])
                            }
                            out_shape = feature[2]
                        else:
                            raise NotImplementedError(f"{op_type} is not supported")
                        block = None
                        if isinstance(op_id, list):
                            op_ids = op_id
                            tmp_block = None
                            block = self.tailor.tir.stages[int(stage_id)].flow[int(block_id)]
                            for i, op_id in enumerate(op_ids):
                                op_id = int(op_id)
                                if hasattr(block, "flow"):
                                    op = block.flow[int(op_id)]
                                else:
                                    if op_id == len(block.main_path)+len(block.residual_path):
                                        op = block.last_node
                                    elif op_id > len(block.main_path)+len(block.residual_path):
                                        raise KeyError(f"Op {op_id} is invalid")
                                    elif len(block.main_path) <= op_id:
                                        op = block.residual_path[op_id-len(block.main_path)]
                                    else:
                                        op = block.main_path[op_id]
                                if i == len(op_ids)-1:
                                    pass
                                else:
                                    block = op
                            op.update(feature_dict)
                        elif op_id is None:
                            op = self.tailor.tir.stages[int(stage_id)].flow[int(block_id)]
                            assert op.type == op_type
                            op.update(feature_dict)
                        else:
                            op_id = int(op_id)
                            block = self.tailor.tir.stages[int(stage_id)].flow[int(block_id)]
                            if hasattr(block, "flow"):
                                op = block.flow[int(op_id)]
                            else:
                                if op_id == len(block.main_path)+len(block.residual_path):
                                    op = block.last_node
                                elif op_id > len(block.main_path)+len(block.residual_path):
                                    raise KeyError(f"Op {op_id} is invalid")
                                elif len(block.main_path) <= op_id:
                                    op = block.residual_path[op_id-len(block.main_path)]
                                else:
                                    op = block.main_path[op_id]
                            assert op.type == op_type
                            op.update(feature_dict)
                        assert op.features["out_shape"] == out_shape
                        output_dict[op_type][feature_info] = op.build()
                caches.append(output_dict)


        num_epochs = self.training_config.epochs
        if int(start_epoch) == int(num_epochs):
            # training has finished
            return

        # Validation mode
        if self.training_config.eval:
            top1_eval_acc = validate(
                self.tailor,
                val_loader,
                self.tailor.supercode,
                None,
                self.training_config)

            if self.training_config.rank == 0:
                logger.info(
                    f"[Eval mode] evaluation top-1 accuracy {top1_eval_acc} (%)")
            return

        # Configure for knowledge distillation
        teacher_model = None
        distillation_loss = None

        pretrained_teacher_path = self.training_config.teacher_path

        if self.training_config.distillation:
            # Teacher model must be stored as an ONNX model.
            # In this version, TailorIR-based teacher building is removed for
            # reducing mixed shared memory management.
            # TODO: A sepearate distillation worker
            teacher_model = self.build_teachers(
                teacher_path=pretrained_teacher_path)

            if self.training_config.distillation:
                distillation_loss = SoftTargetCrossEntropyNoneSoftmax().cuda()

            if not prof:
                top1_eval_acc = validate(
                    self.tailor,
                    val_loader,
                    self.tailor.supercode,
                    None,
                    self.training_config,
                    existed_model=teacher_model)
                if self.training_config.rank == 0:
                    logger.info(
                        f"[Teacher] evaluation top-1 accuracy {top1_eval_acc} (%)")

        if self.training_config.rank == 0:
            logger.info((f"Optimizer: {optimizer}").replace(
                '\n', '\n' + ' ' * 11))
            logger.info(f"Total epoch: {num_epochs}, Start epoch {start_epoch}, Val cycle: {self.training_config.val_cycle}")

        perf_scoreboard = PerformanceScoreboard(
            self.training_config.num_best_scores)

        v_top1, v_top5, v_loss = 0, 0, 0

        if not prof:
            top1_eval_acc = validate(
                self.tailor,
                val_loader,
                self.tailor.supercode,
                None,
                self.training_config)
            if self.training_config.rank == 0:
                logger.info(
                    f"[Supernet] evaluation top-1 accuracy {top1_eval_acc} (%)")
            top1_eval_acc = validate(
                self.tailor,
                val_loader,
                self.tailor.min_sample(),
                None,
                self.training_config)
            if self.training_config.rank == 0:
                logger.info(
                    f"[Mininet] Evaluation accuracy {top1_eval_acc} (%)")

        for epoch in range(start_epoch, num_epochs):
            # Configure for dynamic resolution of loaded data
            if self.training_config.distributed:
                training_sampler.set_epoch(epoch)

            if self.training_config.rank == 0:
                logger.info(f'>>>>>>>> Epoch {epoch}')

            train_loss = train_subnets_one_epoch(
                self.tailor,
                subnets_configs,
                train_loader,
                criterion,
                optimizer,
                self.training_config.dynamic_batch_size,
                lr_scheduler,
                epoch,
                pymonitor,
                self.training_config,
                teacher_model=teacher_model,
                distillation_loss=distillation_loss,
                mixup_fn=mixup_fn,
                sandwich=self.training_config.sandwich,
                sampling_strategy=sampling_strategy,
                prof=prof,
                enable_overlapping=not self.training_config.disable_overlapping,
                branch_samples=branch_samples,
                sharing_computation=sharing_comp,
                caches=caches,
                disable_rescheduling=disable_rescheduling)

            if prof:
                # Profiling mode only tuning for one epoch
                return

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)

            # Validate after each epoch
            supernet_top1_eval_acc = validate(
                self.tailor,
                val_loader,
                self.tailor.supercode,
                None,
                self.training_config)
            mininet_top1_eval_acc = validate(
                self.tailor,
                val_loader,
                self.tailor.min_sample(),
                None,
                self.training_config)

            if self.training_config.rank == 0:
                logger.info(
                    f"[Supernet] Evaluation accuracy [{epoch}/{num_epochs}] {supernet_top1_eval_acc} (%)")
                logger.info(
                    f"[Mininet] Evaluation accuracy [{epoch}/{num_epochs}] {mininet_top1_eval_acc} (%)")
                perf_scoreboard.update(v_top1, v_top5, epoch)
                is_best = perf_scoreboard.is_best(epoch)
                save_checkpoint(
                    epoch,
                    globvar.super_weights,
                    {'top1': v_top1, 'top5': v_top5},
                    is_best,
                    self.training_config.name,
                    output_dir,
                    optimizer=optimizer)

                if epoch % 10 == 0:
                    save_checkpoint(
                        epoch,
                        globvar.super_weights,
                        {'top1': v_top1, 'top5': v_top5},
                        False,
                        self.training_config.name + f'_{epoch}epochs_',
                        output_dir,
                        optimizer=optimizer)

        if self.training_config.rank == 0:
            logger.info('Program completed successfully ... exiting ...')

    def validate(self, dimension_index_dict, chkp_file, cache=None):
        self.tailor.dimension_index_dict = dimension_index_dict

        self.tailor.transform(self.tailor.supercode)
        self.tailor.tir.build(cache=cache)
        super_model = copy.deepcopy(self.tailor.tir.torch_nn)
        # calib_bn(model, self.training_config.path, self.tailor.supercode["resolution"], 128)
        super_model = super_model.to(f"cuda:{dist.get_rank()}")

        if os.path.exists(chkp_file):
            logger.info("load checkpoint from", chkp_file)
            checkpoint = torch.load(chkp_file, map_location=lambda storage, loc: storage)
            checkpoint_epoch = checkpoint.get('epoch', None)
            start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
            super_weights = checkpoint.get('state_dict', None)
            # Update super weights
            # self.tailor.tir.update_weights(super_weights)
            for param_dict_k, param_dict_v in globvar.super_weights.items():
                for param_k, param_v in param_dict_v.items():
                    globvar.super_weights[param_dict_k][param_k] = \
                        copy.deepcopy(super_weights[param_dict_k][param_k])

            # data loader
            if self.val_loader is None:
                train_loader, val_loader, test_loader, training_sampler = load_data_dist(self.training_config, self.image_size_list, self.tailor.supercode["resolution"])
                self.val_loader = val_loader
            else:
                val_loader = self.val_loader

            supernet_eval_acc = validate(
                self.tailor,
                val_loader,
                self.tailor.supercode,
                None,
                self.training_config)
            if self.training_config.rank == 0:
                logger.info(
                    f"[Supernet] evaluation top-1 accuracy {supernet_eval_acc} (%)")

            mininet_eval_acc = validate(
                self.tailor,
                val_loader,
                self.tailor.min_sample(),
                None,
                0,
                self.training_config)
            if self.training_config.rank == 0:
                logger.info(
                    f"[Mininet] Evaluation accuracy {mininet_eval_acc} (%)")
            return supernet_eval_acc[0], supernet_eval_acc[1], mininet_eval_acc[0], mininet_eval_acc[1]
        else:
            raise FileNotFoundError

    def build_teachers(self, teacher_path=None):
        teacher_onnx_path = teacher_path

        import onnx2torch

        teacher_model = onnx2torch.convert(teacher_onnx_path)

        return teacher_model.cuda()
