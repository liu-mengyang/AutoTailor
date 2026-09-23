# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from loguru import logger
import os
import torch as t


def save_checkpoint(epoch, super_weights, extras=None, is_best=None, name=None,
                    output_dir='.', optimizer=None):
    """Save a pyTorch training checkpoint.
    In this version, model_ema is removed.
    Args:
        epoch: current epoch number
        super_weights: sharing weight of supernet
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        output_dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(output_dir):
        raise IOError('Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    filepath = os.path.join(output_dir, filename)
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    filepath_best = os.path.join(output_dir, filename_best)
    
    checkpoint = {
        'epoch': epoch,
        'state_dict': super_weights,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'extras': extras,
    }

    msg = 'Saving checkpoint to:\n'
    msg += '             Current: %s\n' % filepath
    t.save(checkpoint, filepath)
    if is_best:
        msg += '                Best: %s\n' % filepath_best
        t.save(checkpoint, filepath_best)
    logger.info(msg)


def load_checkpoint(chkp_file, optimizer=None):
    """Load a pyTorch training checkpoint.
    Args:
        chkp_file: the checkpoint file
    :returns: super_weights, optimizer, start_epoch
    """
    if not os.path.isfile(chkp_file):
        raise NotImplementedError

    checkpoint = t.load(chkp_file, map_location=lambda storage, loc: storage)

    if 'state_dict' not in checkpoint:
        raise ValueError('Checkpoint must contain model parameters')

    extras = checkpoint.get('extras', None)

    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
    super_weights = checkpoint.get('state_dict', None)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return super_weights, start_epoch, extras

