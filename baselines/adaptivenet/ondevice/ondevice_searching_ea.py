import argparse
import os

import logging
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Subset
import numpy as np
import torch
import time
import torchvision
from mytimm.models import create_model
from mytimm.utils import *
from tools.evolution_finder import EvolutionFinder
import torchvision.transforms as transforms
import timm
from tools import mytools
from tools import global_var
import sys

from utils import evaluate_latency, evaluate_accuracy

parser = argparse.ArgumentParser(description='evolution finder', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR', default='ValForDevice/ValForDevice3k',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='imagenet',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')

parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.05, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
#parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   # help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

parser.add_argument('--model_path', metavar='DIR',
                    help='path to trained model')

parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--GPU', action='store_true', default=False,
                    help='Use GPU')
parser.add_argument("--log_interval", default=200, type=int)
parser.add_argument("--warmupbatches", default=10, type=int)
parser.add_argument('--pths_path', metavar='DIR',
                    help='path to trained model')
parser.add_argument('--slim', action='store_true', default=False)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument("--batch_size_for_lat", default=4, type=int)
parser.add_argument('--pruned', action='store_true', default=False)
parser.add_argument('--save_path', default='./npy/batch-', type=str)
parser.add_argument('--baseline_save_path', default='./npy/baseline-batch-', type=str)
parser.add_argument("--num_workers", default=8, type=int)


parser.add_argument("--data_len", default=500, type=int)
parser.add_argument("--pre_len", default=3, type=int) 
parser.add_argument("--searching_times", default=10, type=int)
parser.add_argument("--population_size", default=200, type=int)

parser.add_argument("--load_times", default=2, type=int) 
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--baseline_load_times", default=1, type=int)
parser.add_argument("--baseline_batch_size", default=500, type=int)
parser.add_argument('--method', default='AdaptiveNet',type=str)
parser.add_argument('--time-budget', default=40, type=int)
parser.add_argument('--framework', default='ncnn', type=str)
parser.add_argument('--device', default='samsung1', type=str)
parser.add_argument("--command", default="ncnn_cpu_f0_fp16", type=str)

args = parser.parse_args()

print("""Usage:
python3 ondevice_searchin_ea.py \\
    --model {resnet50|mbv2} \\
    --time-budget {integer in milliseconds} \\
    --framework {tflite|ncnn} \\
    --device {samsung0|samsung1|redmi0|honor0}""")

if args.model == 'resnet50':
    args.model_path = 'weights/resnet1epoch59acc69.pth'
elif args.model == 'mbv2':
    args.model = 'mobilenetv2_100'
    args.model_path = 'weights/mbv2_100_1epoch44_best.pth'
else:
    raise ValueError("args.model")

if 'samsung' in args.device:
    subcommand = 'f0'
else:
    subcommand = '70'

if args.framework == 'ncnn':
    BACKEND_CONFIG = f'{args.device}'
    COMMAND_CONFIG = f'{args.command}'
    EVALUATE_CONFIG = 'im1k_gpu0_128'
elif args.framework == 'tflite':
    BACKEND_CONFIG = f'{args.device}'
    COMMAND_CONFIG = f'{args.command}'
    EVALUATE_CONFIG = 'im1k_gpu0_128'
else:
    raise ValueError("args.framework")

print('BACKEND_CONFIG', BACKEND_CONFIG)
print('COMMAND_CONFIG', COMMAND_CONFIG)

def test_lat(block, input, test_times, block_idx=0, choice_idx=0):
    print(f"block_idx:{block_idx} choice_idx:{choice_idx}")
    return evaluate_latency(
        block,
        input.shape,
        BACKEND_CONFIG,
        COMMAND_CONFIG
    )

def get_resnet_lats(model, batchsize, test_times=500):
    model.eval()
    x = torch.rand(batchsize, 3, 224, 224)
    if arg.GPU:
        model.cuda()
        x.cuda()
    lats = []
    layers = [model.conv1, model.bn1, model.act1, model.maxpool]
    former_layers = nn.Sequential(*layers)
    if args.GPU:
        former_layers.cuda()
    lats.append(test_lat(former_layers, x, test_times, 99, 98))
    x = former_layers(x)
    for blockidx in range(len(model.multiblocks)):
        lats.append([])
        for choiceidx in range(len(model.multiblocks[blockidx])):
            lats[-1].append(test_lat(model.multiblocks[blockidx][choiceidx], x, test_times, blockidx, choiceidx))
        x = model.multiblocks[blockidx][0](x)
    f_layers = [model.global_pool, model.fc]
    latter_layers = nn.Sequential(*f_layers)
    lats.append(test_lat(latter_layers, x, test_times, 99, 99))
    return lats

def get_mbv_lats(model, batchsize, test_times=1000):
    model.eval()
    x = torch.rand(batchsize, 3, 224, 224)
    if args.GPU:
        model.cuda()
        x.cuda()
    lats = []
    layers = [model.conv_stem, model.bn1, model.act1]
    former_layers = nn.Sequential(*layers)
    if args.GPU:
        former_layers.cuda()
    lats.append(test_lat(former_layers, x, test_times))
    x = former_layers(x)
    for blockidx in range(len(model.multiblocks)):
        lats.append([])
        for choiceidx in range(len(model.multiblocks[blockidx])):
            lats[-1].append(test_lat(model.multiblocks[blockidx][choiceidx], x, test_times))
        x = model.multiblocks[blockidx][0](x)
    f_layers = [model.conv_head, model.bn2, model.act2, model.global_pool, model.classifier]
    latter_layers = nn.Sequential(*f_layers)
    lats.append(test_lat(latter_layers, x, test_times))
    return lats

def validate_baseline(model, loader, subnet, args, loss_fn, lats):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    lat = lats[0]
    for blockidx in range(len(subnet)):
        if subnet[blockidx] != 99:
            lat += lats[blockidx+1][subnet[blockidx]]
    lat += lats[-1]
    if args.GPU:
        model.cuda()
    model.eval()
    with torch.no_grad():
       for batch_idx, (input, target) in enumerate(loader):
            if args.GPU:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            # with amp_autocast():
            output = model(input, subnet,batch_idx=batch_idx,infer_type=0)
            if args.GPU:
                torch.cuda.synchronize()

            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            
            if args.GPU:
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
    return top1_m.avg, lat

def validate(model, loader, subnet, args, loss_fn, lats,infer_type=2):
    validate_time = time.time()
    get_subnet_time = 0
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    lat = lats[0]
    for blockidx in range(len(subnet)):
        if subnet[blockidx] != 99:
            lat += lats[blockidx+1][subnet[blockidx]]
    lat += lats[-1]
    if args.GPU:
        model.cuda()
    model.eval()
    check_validated_feature_time = time.time()
    if infer_type == 2:
        model.check_validated_feature()
    check_validated_feature_time = time.time()-check_validated_feature_time
    end = time.time()
    last_idx = len(loader) - 1
    total_time = 0
    infer_time = 0 
    infer_after_time = 0
    to_gpu_time = 0
    with torch.no_grad():
        for batch_idx in range(len(loader)):
            input = loader[batch_idx][0]
            target = loader[batch_idx][1]
            last_batch = batch_idx == last_idx
            # if args.GPU:
            #     input = input.cuda()
            #     target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            # with amp_autocast():
            s2 = time.time()
            t1 = time.time()
            output = model(input, subnet,batch_idx=batch_idx,infer_type=infer_type)
            if args.GPU:
                torch.cuda.synchronize()
            infer_time += time.time()-s2
            s3 = time.time()
            if batch_idx >= args.warmupbatches:
                total_time += (time.time() - t1)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            if args.GPU:
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
    update_validated_feature_time = time.time()
    if infer_type == 2:
        model.update_validated_feature()
    update_validated_feature_time = time.time()-update_validated_feature_time
    validate_time = time.time()-validate_time
    return top1_m.sum, top1_m.count, lat

def warmup(model, args, warmuptime, teachermodel=False, subnet=None):
    x = torch.randn(args.batch_size, 3, 224, 224)
    if args.GPU:
        x.cuda()
    for _ in range(warmuptime):
        if teachermodel:
            output = model(x)
        else:
            output = model(x, subnet)


def main():
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    args.world_size = 1
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=1000,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if 'resnet' in args.model:
        model.get_skip_blocks(more_options=False)
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = mytools.prune_model(teachermodel, [0.2, 0.4])
            model.get_pruned_module(prune_modules)
            del teachermodel
    if 'mobilenetv2' in args.model:
        model.get_multi_blocks()
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = mytools.prune_mbv2(teachermodel, [0.25, 0.5])
            model.get_pruned_module(prune_modules)
            del teachermodel

    global_var._init()
    global_var.set_value('validated_feature', set())
    global_var.set_value('need_save_feature', set())
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=True)
    if args.GPU:
        model.cuda()
    if args.method == "AdaptiveNet":
        mytools.load_data(model,args.model,args.data_dir,args.save_path, method = args.method, load_times=args.load_times,batch_size=args.batch_size,data_len=args.data_len)
    elif args.method == "BaseLine0":
        mytools.load_data(model,args.model,args.data_dir,args.baseline_save_path, method = args.method, load_times=args.load_times,batch_size=args.batch_size,data_len=args.data_len)
    
    if args.slim:
        model.adjust_channels()
        model.get_multi_blocks()
    elif 'mobilenetv2' in args.model:
        layer_lens = []
        for layeridx in range(len(model.multiblocks)):
            layerlen = len(model.multiblocks[layeridx])
            for blockidx in range(len(model.multiblocks[layeridx])): # note that we did not prune the last block
                layer_lens.append(layerlen)
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules, prune_checkpoint, prune_block_nums = mytools.prune_model_v2(teachermodel, [0.25, 0.5])
            model.get_pruned_module(prune_modules, prune_checkpoint, prune_block_nums)
            del teachermodel

    def profile(model, subnet):
        model.apply_subnet(subnet)
        dummy_input = torch.randn(1, 3, 224, 224)
        from thop import profile
        if args.GPU:
            dummy_input.cuda()
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        latency = evaluate_latency(model, (1, 3, 224, 224), BACKEND_CONFIG, COMMAND_CONFIG)
        accuracy = evaluate_accuracy(model, EVALUATE_CONFIG)
        return flops, latency, accuracy

    max_subnet = model.generate_random_subnet()
    max_subnet = [0 for _ in range(len(max_subnet))]
    min_subnet = model.generate_random_subnet()
    min_subnet = [2 for _ in range(len(min_subnet))]
    min_subnet[-2] = 1
    min_subnet[-1] = 0
    
    max_subnet_flops, max_subnet_latency, max_subnet_accuracy = profile(model, max_subnet)
    min_subnet_flops, min_subnet_latency, min_subnet_accuracy = profile(model, min_subnet)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    baseline_subnet = model.generate_random_subnet()
    baseline_subnet = [0 for _ in range(len(baseline_subnet))]
    model.apply_subnet(baseline_subnet)
    baseline_latency = evaluate_latency(model, (1, 3, 224, 224), BACKEND_CONFIG, COMMAND_CONFIG)
    baseline_accuracy = evaluate_accuracy(model, EVALUATE_CONFIG)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    t1 = time.time()
    lats_cache = f"./npy/{args.model}_{args.framework}_{args.device}_lats.npy"
    if os.path.exists(lats_cache):
        print("Cache exists, loading lats from cache!")
        lats = np.load(lats_cache, allow_pickle=True)
    else:
        if "resnet" in args.model:
            lats = get_resnet_lats(model, batchsize=args.batch_size_for_lat)
        else:
            lats = get_mbv_lats(model, batchsize=args.batch_size_for_lat)
        np.save(lats_cache, lats)
    loss_fn = nn.CrossEntropyLoss()
    if args.GPU:
        loss_fn.cuda()
    lut_time = time.time() - t1
    
    t1 = time.time()

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset_eval = torchvision.datasets.ImageFolder(root=args.data_dir,transform=data_transform)
    idxs = np.load('./npy/idxs.npy').tolist()[:args.data_len]
    eval_set = Subset(dataset_eval, idxs)
    loader_eval = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    
    lens = layer_lens if "mobilenetv2" in args.model else None
    finder = EvolutionFinder(batch_size=args.batch_size, population_size=args.population_size, branch_choices=model.block_choices, time_budget=args.time_budget/1000, searching_times=args.searching_times, lats=lats, model_lens=lens)
    if args.method == "AdaptiveNet":
        _, best_info = finder.evolution_search(model, validate, args, loss_fn)
    elif args.method == "BaseLine0":
        _, best_info = finder.evolution_search_baseline1(model, validate, args, loss_fn)

    best_subnet = best_info[1]
    t2 = time.time()
    evolution_time = time.time() - t1

    


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    model.apply_subnet(best_subnet)
    best_latency = evaluate_latency(model, (1, 3, 224, 224), BACKEND_CONFIG, COMMAND_CONFIG)
    best_accuracy = evaluate_accuracy(model, EVALUATE_CONFIG)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def flatten_list(l):
        flat_l = []
        for item in l:
            if type(item) == list:
                flat_l.extend(flatten_list(item))
            else:
                flat_l.append(item)
        return flat_l
    n_blocks = len(flatten_list(lats))
    n = len(best_subnet)
    f = [1] * (n + 5)
    f[2] = 2
    for i in range(3, n + 1):
        f[i] = f[i - 1] + f[i - 2] + f[i - 3]
    n_subnets = f[n]

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Baseline Latency\t", baseline_latency*1000)
    print("Baseline Accuracy\t", baseline_accuracy)
    print("LUT Time Cost\t", lut_time)
    print("Evolution Time Cost (excluded LUT)\t", evolution_time)
    print("Time Constraint:", args.time_budget)
    print("Best Latency:", best_latency*1000)
    print("Best Accuracy:", best_accuracy)
    with open("results.log", "a") as f:
        f.write(f"\n\n=== Model: {args.model}, Framework: {args.framework}, Device: {args.device}, Latency Constraint: {args.time_budget} ===\n")
        f.write(f"Baseline Latency\t{baseline_latency*1000}\n")
        f.write(f"Baseline Accuracy\t{baseline_accuracy}\n\n")
        f.write(f"Max Subnet Latency\t{max_subnet_latency*1000}\n")
        f.write(f"Max Subnet Accuracy\t{max_subnet_accuracy}\n")
        f.write(f"Max Subnet Flops\t{max_subnet_flops}\n\n")
        f.write(f"Min Subnet Latency\t{min_subnet_latency*1000}\n")
        f.write(f"Min Subnet Accuracy\t{min_subnet_accuracy}\n")
        f.write(f"Min Subnet Flops\t{min_subnet_flops}\n\n")
        f.write(f"LUT Time Cost\t{lut_time}\n")
        f.write(f"Evolution Time Cost (excluded LUT)\t{evolution_time}\n\n")
        f.write(f"Best Latency\t{best_latency*1000}\n")
        f.write(f"Best Accuracy\t{best_accuracy}\n\n")
        f.write(f"#Blocks\t{n_blocks}\t\t#Subnets\t{n_subnets}\n")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


if __name__ == '__main__':
    main()