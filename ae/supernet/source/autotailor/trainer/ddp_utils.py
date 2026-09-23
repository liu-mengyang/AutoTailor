
import math
import os
import random
import sys

from loguru import logger
import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.data import create_transform

from .mydataloader import myDataLoader
from .my_random_resize_crop import MyRandomResizedCrop


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_distributed_training(args):
    
    args.device = "cuda"

    args.world_size = 1
    args.rank = 0
    args.distributed = True

    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)


def init_logger(exp_name, output_dir):
    log_path = f"{output_dir}/{exp_name}.log"
    logger.add(log_path)
    

def setup_print(is_master):
        
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def load_data_dist(cfg, image_size_list, val_image_size, searching_set=False):
    assert cfg.dataset == 'imagenet' or cfg.dataset == "coco"
    
    _valid_transform_dict = {}

    assert isinstance(image_size_list, list)
    image_size_list.sort()  # e.g., 160 -> 224
    MyRandomResizedCrop.IMAGE_SIZE_LIST = image_size_list.copy()
    MyRandomResizedCrop.ACTIVE_SIZE = max(image_size_list)

    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    for img_size in image_size_list:
        _valid_transform_dict[img_size] = transforms.Compose(
            [
                transforms.Resize(int(math.ceil(img_size / 0.875))),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    active_img_size = max(image_size_list)  # active resolution for test
    valid_transforms = _valid_transform_dict[active_img_size]
    
    annotations_dir = None
    train_annotations_file = None
    val_annotations_file = None
    if cfg.dataset == "imagenet":
        traindir = os.path.join(cfg.path, 'train')
        valdir = os.path.join(cfg.path, 'val')
    elif cfg.dataset == "coco":
        traindir = os.path.join(cfg.path, "train2017")
        valdir = os.path.join(cfg.path, "val2017")
        annotations_dir = os.path.join(cfg.path, 'annotations')
        train_annotations_file = os.path.join(annotations_dir, 'instances_train2017.json')
        val_annotations_file = os.path.join(annotations_dir, 'instances_val2017.json')
    print("Train dir:", traindir)

    aug_type = getattr(cfg, 'aug_type', 'none')
    
    resize_transform_class = MyRandomResizedCrop
    # random_resize_crop -> random_horizontal_flip
    train_transforms = [
        resize_transform_class(image_size_list, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    
    # color augmentation (optional)
    color_transform = None
    # if self.distort_color == "torch":
        # color_transform = transforms.ColorJitter(
        #     brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        # )
    # elif self.distort_color == "tf":
    color_transform = transforms.ColorJitter(
        brightness=32.0 / 255.0, saturation=0.5
    )
    # if color_transform is not None:
    #     train_transforms.append(color_transform)

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
    ]

    train_transforms = transforms.Compose(train_transforms)
    collate_fn = None
    if cfg.dataset == "imagenet":
        train_set = datasets.ImageFolder(
            traindir,
            train_transforms
        )
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_image_size),
                transforms.ToTensor(),
                normalize,
            ]))
    elif cfg.dataset == "coco":
        train_set = datasets.CocoDetection(root=traindir, annFile=train_annotations_file,
                                  transform=train_transforms)
        val_set = datasets.CocoDetection(root=valdir, annFile=val_annotations_file,
                                transform=transforms.Compose([
                                    transforms.Resize(
                                        256, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(val_image_size),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        collate_fn = lambda x: tuple(zip(*x))
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = myDataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=False, drop_last=True,
        sampler=train_sampler, collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=False, drop_last=False,
        sampler=val_sampler, collate_fn=collate_fn)

    if searching_set:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(cfg.path, 'search'), transforms.Compose([
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(val_image_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.workers, pin_memory=False, drop_last=False,
            collate_fn=collate_fn)
    else:
        val_loader = test_loader

    return train_loader, val_loader, test_loader, train_sampler