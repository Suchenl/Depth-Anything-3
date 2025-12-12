import torch
from typing import Literal
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os

def make_deterministic(seed):
    # 🌟 Must be retained for reproducibility
    # Ensure that all Rank models have the same initialization weights and module ordering 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # ⚠️ These settings typically increase memory overhead or decrease speedups
    # Remove deterministic coercion from cuDNN to allow more optimized (but non-deterministic) algorithms to run
    # torch.backends.cudnn.deterministic = True 
    # Recommended: keep benchmark True to choose the fastest algorithm (and usually the most memory-efficient)
    torch.backends.cudnn.benchmark = True # PyTorch defaults to False
    
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

import torch

def optimize_params(loss, optimizer, scheduler=None, retain_graph=False, gradient_clip=None, train_param=None, scaler=None):
    """
    Performs a complete optimization step: backward pass, gradient clipping,
    optimizer step, scheduler step, and zeroing gradients.

    This function is now compatible with Automatic Mixed Precision (AMP) training.

    Args:
        loss (torch.Tensor): The loss tensor to perform backpropagation on.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler.
        retain_graph (bool, optional): If True, the graph will be retained after backward pass.
        gradient_clip (float, optional): The max norm for gradient clipping.
        train_param (iterable, optional): The parameters to be clipped. Required if gradient_clip is set.
        scaler (torch.cuda.amp.GradScaler, optional): 
            If a GradScaler instance is provided, the optimization step will be
            performed using automatic mixed precision. If None, a standard
            single-precision step is performed. Defaults to None.
    """
    if scaler is not None:
        # --- Automatic Mixed Precision (AMP) Path ---
        # 1. Scale the loss and call backward() to create scaled gradients.
        scaler.scale(loss).backward(retain_graph=retain_graph)
        # 2. (Optional) Unscale the gradients for clipping.
        if gradient_clip is not None:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_param, max_norm=gradient_clip)
        # 3. Scaler's step() calls optimizer.step() internally.
        #    It skips the step if gradients containinfs or NaNs.
        scaler.step(optimizer)
        # 4. Update the scale for next iteration.
        scaler.update()
    else:
        # --- Standard Precision Path (Original Logic) ---
        loss.backward(retain_graph=retain_graph)
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(train_param, max_norm=gradient_clip)
        optimizer.step()
    # --- Common Steps for Both Paths ---
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()

def initialize_tensorboard_writer(log_dir=None, 
                                  ckpt_path=None,
                                  ckpt_savedir=None, 
                                  repeat_maxnum=5):
    if log_dir is not None:
        if log_dir == 'resume':
            assert ckpt_path is not None, "If log_dir is set to 'resume', ckpt_path must be provided"
            ckpt_savedir = Path(ckpt_path).parent
            i = 0
            while not os.path.exists(ckpt_savedir / 'logs') and i < repeat_maxnum:
                ckpt_savedir = ckpt_savedir.parent
                i += 1
            if i == repeat_maxnum:
                raise ValueError(f"Cannot find logs folder in {ckpt_path} or its parent folders")
            else:
                log_dir = ckpt_savedir / 'logs'
        else:
            log_dir = str(Path(log_dir))

    else:  # log_dir is None
        assert ckpt_savedir is not None, "If log_dir is None, ckpt_savedir must be provided"
        log_dir = str(Path(ckpt_savedir) / f'logs')
    print('Save tensorboard log dir:', log_dir)
    writer = SummaryWriter(log_dir)
    return writer, log_dir

def initialize_checkpoint():
    checkpoint = {
        'epoch_num': 0,
        'step_num': 0,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'loss': 0
    }
    return checkpoint

def update_checkpoints_savedir(log_dir=None,
                               ckpt_path=None,
                               ckpt_savedir=None, 
                               exp_name=None,
                               repeat_maxnum=5):
    """ update checkpoint dir with time suffix """
    # Case 1: resume from existing checkpoint
    if ckpt_path is not None and log_dir == 'resume':
        ckpt_savedir = Path(ckpt_path).parent
        i = 0
        while not os.path.exists(ckpt_savedir / 'logs') and i < repeat_maxnum:
            ckpt_savedir = ckpt_savedir.parent
            i += 1
        if i == repeat_maxnum:
            raise ValueError(f"Cannot find logs folder in {ckpt_path} or its parent folders")
        else:
            print(f"Resuming from checkpoint directory: {ckpt_savedir}")
            
    # Case 2: create new save directory
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate current timestamp (formatted as YYYYMMDD_HHMMSS)
        ckpt_savedir = ckpt_savedir + '/' + exp_name + '/' + timestamp
        # create dir to save weights
        os.makedirs(ckpt_savedir, exist_ok=True)
        
    print(f"Save dir: {ckpt_savedir}")
    return ckpt_savedir
