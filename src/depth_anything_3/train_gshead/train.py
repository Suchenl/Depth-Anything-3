# ===================== Deepspeed ZeRO ===================== 
import deepspeed
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# from deepspeed.runtime.zero.stage3 import ZeRO_Init
# ==========================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
import argparse
from PIL import Image
from datetime import datetime
import random
import os
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
import shutil
from typing import *
import matplotlib.pyplot as plt
import math
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import PyTorchModelHubMixin

# import models
from .model import DepthAnything3
from .dataset import TrainDA3Static3DGSDataset, split_val_dataset

# import utils
from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

# import my utils
from .dataset import ...

from .utils.train_utils import (
    make_deterministic, 
    set_requires_grad,
    optimize_params, 
    initialize_tensorboard_writer, 
    initialize_checkpoint, 
    update_checkpoints_savedir
    )

from .utils.io_utils import (
    parse_args,
    save_namespace_to_json,
    tensor_to_pil,
    namespace_to_dict
    )

from .utils.loss_functions import LPIPSLoss, DiceLoss, MixtureOfLaplaceLoss


def handler(signum, frame):
    print(f"⚠️ Received signal {signum} on PID {os.getpid()}", flush=True)
    sys.exit(1)

for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGKILL, signal.SIGABRT]:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass
    
def get_filtered_model_state_dict(model):
    # If the model is DataParallel wrapped, access the .module
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    original_state_dict = model.state_dict()
    # Define the parameter name prefix you don't want to save, it can be multiple
    prefixes_to_exclude = ['vae.'] 
    filtered_state_dict = {
        key: value for key, value in original_state_dict.items()
        if not any(key.startswith(prefix) for prefix in prefixes_to_exclude)
    }
    return filtered_state_dict


def prepare_datas(args, generator: Optional[torch.Generator] = None):
    # Define dataset and dataloader
    train_dataset = DA3Training3DGSDataset(
        dataset_name=args.train_config.train_data.dataset_name, 
        dataset_path=args.train_config.train_data.dataset_path,
        process_res=args.train_config.train_data.process_res,
        process_res_method=args.train_config.train_data.process_res_method
        )

    if args.train_config.validation.enabled:
        if args.train_config.validation.split_from_train:
            val_dataset, train_dataset = split_val_dataset(
                dataset=train_dataset,
                split_num=args.train_config.validation.split_num,
                split_ratio=args.train_config.validation.split_ratio,
                generator=generator)
        else: raise NotImplementedError("Not implemented yet")
        
        val_num = len(val_dataset)
        print('val_num:', val_num)
    
    train_num = len(train_dataset)
    print('train_num:', train_num)

    
    # Calculate the number of total_steps
    num_gpus = dist.get_world_size()
    per_gpu_batch_size = args.deepspeed_config['train_micro_batch_size_per_gpu']
    global_batch_size = per_gpu_batch_size * num_gpus * args.deepspeed_config['gradient_accumulation_steps']
    steps_per_epoch = math.ceil(train_num / global_batch_size)
    total_num_steps = steps_per_epoch * args.train_config.num_epochs
    if "scheduler" in args.deepspeed_config.keys():
        args.deepspeed_config["scheduler"]["params"]["total_num_steps"] = total_num_steps
        print(f"Dynamically setting scheduler 'total_num_steps' to {total_num_steps}")
        
    # DeepSpeed Need: Use DistributedSampler
    ## Use DistributedSampler for the training set
    train_sampler = DistributedSampler(train_dataset)
    # Use sampler in DataLoader and shuffle 'must be' False
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=per_gpu_batch_size, # The batch size of per-GPU 
                                  sampler=train_sampler,
                                  shuffle=False, # Sampler handles shuffle.
                                  num_workers=args.train_config.num_workers)
    args.train_config.total_steps = len(train_dataloader) * args.train_config.num_epochs

    ## Use DistributedSampler for the validation set as well
    if args.train_config.validation.enabled:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) # shuffle=False for consistent evaluation
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=per_gpu_batch_size, 
                                    sampler=val_sampler, # Use the sampler
                                    num_workers=args.train_config.num_workers)
    else:
        val_dataloader = None
        
    return train_dataloader, val_dataloader

def prepare_model_engine(args, device):
    # Initialize the model
    model = DepthAnything3(model_name=args.model_config.model_name, align_3dgs=args.model_config.align_3dgs)

    # Freeze the specific parameters
    if args.model_config.freeze_main:
        train_param = []
        for name, param in model.named_parameters():
            if 'gs_head' in name:
                train_param.append(param)
            else:
                param.requires_grad = False
        print(f"Freeze the parameters except gs_head")
    else: 
        train_param = model.parameters()
        
    # Load the model from normal checkpoint if specified
    if args.model_config.load_ckpt.enabled and args.model_config.load_ckpt.choice == 'normal':
        assert args.model_config.normal.ckpt_path is not None, "normal_ckpt_path must be specified when load_ckpt.enabled is 'true' & load_ckpt.choice is 'normal'"
        ckpt = torch.load(args.model_config.normal_ckpt_path, map_location=device)
        if "model_state_dict" in ckpt.keys(): print(f"Load normal checkpoint for model | from {args.model_config.normal_ckpt_path}:\n\t", model.load_state_dict(ckpt["model_state_dict"], strict=False))
        else: print(f"Load normal checkpoint for model | from {args.model_config.normal_ckpt_path}:\n\t", model.load_state_dict(ckpt, strict=False))

    # Use deepspeed.initialize to wrap all components uniformly
    # It automatically handles model and data movement to the GPU, blended accuracy, ZeRO optimization
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=train_param,
        optimizer=None,
        lr_scheduler=None,
        config=args.deepspeed_config)

    # The DeepSpeed way to load checkpoint
    start_epoch = 0
    start_step = 0
    if args.model_config.load_ckpt.enabled and args.model_config.load_ckpt.choice == 'deepspeed':
        # load_checkpoint handles model, optimizer, and scheduler states automatically.
        # It also returns a client_state dictionary for things like epoch or step number.
        load_path, client_state = model_engine.load_checkpoint(
            args.model_config.load_ckpt.deepspeed.ckpt_path, 
            load_module_strict=False
            )
        if client_state is not None:
            start_epoch = client_state.get('epoch_num', 0)
            start_step = client_state.get('step_num', 0)
            print(f"Loaded checkpoint from {load_path}. Resuming from epoch {start_epoch}, step {start_step}")
        else: print(f"Loaded checkpoint from {args.model_config.load_ckpt.deepspeed.ckpt_path}, but client_state is None")

    return model_engine, optimizer, scheduler, start_epoch, start_step

def prepare_loss_functions(args, device):
    class LossFunctions:
        def __init__(self, args, device):
            funcs = {}
            # Rendering Loss
            if args.loss_config.rendering_loss.enabled:
                if args.loss_config.rendering_loss.lpips_loss.enabled:
                    funcs['render_lpips_loss']['func'] = LPIPSLoss(net='alex', device=device)
                    funcs['render_lpips_loss']['weight'] = args.loss_config.rendering_loss.lpips_loss.weight
                if args.loss_config.rendering_loss.mse_loss.enabled:
                    funcs['render_mse_loss']['func'] = nn.MSELoss().to(device)
                    funcs['render_mse_loss']['weight'] = args.loss_config.rendering_loss.mse_loss.weight

            # Depth Loss
            if args.loss_config.depth_loss.enabled:
                if args.loss_config.depth_loss.l1_loss.enabled:
                    funcs['depth_l1_loss']['func'] = nn.L1Loss().to(device)
                    funcs['depth_l1_loss']['weight'] = args.loss_config.depth_loss.l1_loss.weight
                if args.loss_config.depth_loss.grad_loss.enabled:
                    funcs['depth_grad_loss']['func'] = nn.L1Loss().to(device)
                    funcs['depth_grad_loss']['weight'] = args.loss_config.depth_loss.grad_loss.weight

            # Dynamic Loss
            if args.loss_config.dynamic_loss.enabled:
                if args.loss_config.dynamic_loss.dice_loss.enabled:
                    funcs['dynamic_dice_loss']['func'] = DiceLoss().to(device)
                    funcs['dynamic_dice_loss']['weight'] = args.loss_config.dynamic_loss.dice_loss.weight

            # Motion Loss
            if args.loss.config.motion_loss.enabled:
                if args.loss_config.motion_loss.mol_loss.enabled:
                    funcs['motion_mol_loss']['func'] = MixtureOfLaplaceLoss().to(device)
                    funcs['motion_mol_loss']['weight'] = args.loss_config.motion_loss.mol_loss.weight
                if args.loss_config.motion_loss.l1_loss.enabled:
                    funcs['motion_l1_loss']['func'] = nn.L1Loss().to(device)
                    funcs['motion_l1_loss']['weight'] = args.loss_config.motion_loss.l1_loss.weight

            self.funcs = funcs

        def __call__(self, loss_name, *args, **kwargs):
            return self.funcs[loss_name]['func'](*args, **kwargs) * self.funcs[loss_name]['weight']
                
    return LossFunctions(args, device)

def run_one_epoch(args, epoch, model_engine, optimizer, train_dataloader, val_dataloader, loss_functions, device, writer):

    
def train(args, device):
    # Initialize the random generator
    g_cuda = torch.Generator(device=device)
    
    # Prepare Datas
    train_dataloader, val_dataloader = prepare_datas(args, g_cuda)
    
    # Initialize the model, optimizer and loss functions
    model_engine, optimizer, scheduler, start_epoch, start_step = prepare_model_engine(args, device)

    # Prepare the loss functions
    loss_functions = prepare_loss_functions(args, device)

    # Initialize the bar and writer
    # Only the master process (rank 0) creates the TensorBoard writer and tqdm progress bars.
    if model_engine.global_rank == 0:
        bar = tqdm(train_dataloader, desc='Training Steps')
        if args.train_config.ckpt.enable_tensorboard:
            writer, log_dir = initialize_tensorboard_writer(log_dir=args.train_config.ckpt.log_dir, 
                                                            ckpt_path=args.train_config.ckpt.read_path,
                                                            ckpt_savedir=args.train_config.ckpt.save_dir)
            args.train_config.ckpt.log_dir = log_dir
    else: bar = train_dataloader # Other processes only iterate over the data, no progress bar is displayed

    # Cycle epoch nums
    for epoch in range(start_epoch, start_epoch + args.train_config.num_epochs):
        g_cuda.manual_seed(args.train_config.random_seed + epoch)
    
        # Training
        train_sampler.set_epoch(epoch) # <--- 11. Ensure that the shuffle is different for each epoch.
        epoch_loss = 0
        disc_epoch_loss = 0
        loss_add_times = 0
        model_engine.train()

        if args.train_config.print_memory:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                
        for step, datas in enumerate(bar):
            # Get data
            cover_img, init_flow = datas[0].to(device), datas[1].to(device)
            B, C, H, W = cover_img.shape

            if args.deepspeed_config.get('fp16', False).get('enabled', False):
                cover_img, init_flow = cover_img.half(), init_flow.half()
            elif args.deepspeed_config.get('bf16', False).get('enabled', False):
                cover_img, init_flow = cover_img.bfloat16(), init_flow.bfloat16()
            output = model_engine(image,
                                  extrinsics,
                                  intrinsics,
                                  export_feat_layers, 
                                  infer_gs=True, 
                                  use_ray_pose=True, 
                                  ref_view_strategy="first")
            
            enc_img = output['enc_img']
            
            # # ============= test motion enhancer ================
            # true_motion = output['true_motion'].detach().float()
            # viz_motions(true_motion, args.train_config.checkpoints_savedir, 'viz', f"motion_true_rank{model_engine.global_rank}", epoch, step, split_set='train', val_step=None, viz_with='both')
            # continue
            # # ===================================================
            
            # 1.train main model
            ## (1) Calculate the image losses
            if args.loss_config.img_sim_loss.enabled:
                loss_img_sim = img_sim_loss_fn(enc_img, cover_img)
            else:
                loss_img_sim = torch.tensor(0.0, device=device)
                
            if args.loss_config.lpips_loss.enabled:
                loss_lpips = lpips_loss_fn(enc_img, cover_img)
            else:
                loss_lpips = torch.tensor(0.0, device=device)
                
            if args.loss_config.gan_loss.enabled:
                disc_fake_out = disc_engine(enc_img)
                loss_disc_enc = gan_loss_fn(disc_fake_out['global_disc'], True) * args.model_config.disc_model.global_weight + \
                                gan_loss_fn(disc_fake_out['local_disc'], True) * args.model_config.disc_model.local_weight
            else:
                loss_disc_enc = torch.tensor(0.0, device=device)
                
            loss_img = loss_img_sim * args.loss_config.img_sim_loss.weight + \
                        loss_lpips * args.loss_config.lpips_loss.weight + \
                        loss_disc_enc * args.loss_config.gan_loss.weight 
                        
            ## (2) Calculate the motion losses
            if args.loss_config.motion_loss.enabled:
                loss_motion = output['loss_motion'] * args.loss_config.motion_loss.weight
            else:
                loss_motion = torch.tensor(0.0, device=device)
                
            ## (3) Calculate the content losses
            if args.loss_config.content_loss.enabled:
                loss_content = output['loss_content'] * args.loss_config.content_loss.weight
            else:
                loss_content = torch.tensor(0.0, device=device)
                
            ## (4) Calculate the template loss
            if args.loss_config.template_loss.enabled:
                loss_tpl = output['loss_tpl']
            else:
                loss_tpl = torch.tensor(0.0, device=device)
            ## (5) Calculate the total loss
            loss = loss_img + loss_motion + loss_content + loss_tpl * args.loss_config.template_loss.weight
            
            model_engine.zero_grad()
            model_engine.backward(loss)
            model_engine.step()
            ## If EMA is enabled in the config, this will overwrite the optimizer's update
            model_engine.perform_ema_update()
            if args.loss_config.gan_loss.enabled:
                # 2. train net_disc
                # We need to freeze the main model (generator) parameters
                set_requires_grad(model_engine.module, False)
                set_requires_grad(disc_engine.module, True)
                # The .train() call should also be on the engine
                model_engine.eval()
                disc_engine.train()
                disc_real_out = net_disc(cover_img)
                disc_fake_out = net_disc(enc_img.detach())
                loss_disc_real = gan_loss_fn(disc_real_out['global_disc'], True) * args.model_config.disc_model.global_weight + \
                                gan_loss_fn(disc_real_out['local_disc'], True) * args.model_config.disc_model.local_weight
                loss_disc_fake = gan_loss_fn(disc_fake_out['global_disc'], False) * args.model_config.disc_model.global_weight + \
                                gan_loss_fn(disc_fake_out['local_disc'], False) * args.model_config.disc_model.local_weight
                loss_net_disc = (loss_disc_real + loss_disc_fake) * 0.5
                # Update params and learning rate
                disc_engine.zero_grad()
                disc_engine.backward(loss_net_disc)
                disc_engine.step()
                # Update checkpoint
                set_requires_grad(model_engine.module, True)
                set_requires_grad(disc_engine.module, False)
                model_engine.train()
                disc_engine.eval()
            else:
                loss_net_disc = torch.tensor(0.0, device=device)
            # ===============================================================================================
            # Accumulate loss
            epoch_loss += loss.item()
            disc_epoch_loss += loss_net_disc.item()
            loss_add_times += 1
            
            # Logging and printing is performed only on the master process (rank 0) to avoid duplicate printing by multiple processes.
            if model_engine.global_rank == 0:
                # Visualize training process
                bar.set_description("Train epoch[{}/{}] step:{} loss_model:{:.3f} loss_sim:{:.3f} loss_lpips:{:.3f} loss_motion:{:.3f} loss_content:{:.3f} loss_tpl:{:.3f} loss_disc_enc:{:.3f} loss_net_disc:{:.3f}"
                                    .format(epoch,
                                            start_epoch + args.train_config.num_epochs - 1, 
                                            model_engine.global_steps - 1,
                                            loss.item(),
                                            loss_img_sim.item(),
                                            loss_lpips.item(),
                                            loss_motion.item(),
                                            loss_content.item(),
                                            loss_tpl.item(),
                                            loss_disc_enc.item(),
                                            loss_net_disc.item()
                                            )
                                    )
                # Update tensorboard
                writer.add_scalar(f'Training (step) -model- Loss', loss.item(), model_engine.global_steps - 1)
                if args.loss_config.img_sim_loss.enabled:
                    writer.add_scalar(f'Training (step) -model- Loss_img_sim', loss_img_sim.item(), model_engine.global_steps - 1)
                if args.loss_config.lpips_loss.enabled:
                    writer.add_scalar(f'Training (step) -model- Loss_lpips', loss_lpips.item(), model_engine.global_steps - 1)
                if args.loss_config.motion_loss.enabled:
                    writer.add_scalar(f'Training (step) -model- Loss_motion', loss_motion.item(), model_engine.global_steps - 1)
                if args.loss_config.content_loss.enabled:
                    writer.add_scalar(f'Training (step) -model- Loss_content', loss_content.item(), model_engine.global_steps - 1)
                if args.loss_config.gan_loss.enabled:
                    writer.add_scalar(f'Training (step) -Net_Disc- Loss', loss_net_disc.item(), disc_engine.global_steps - 1)
                    writer.add_scalar(f'Training (step) -model- Loss_disc_enc', loss_disc_enc.item(), model_engine.global_steps - 1)
                if args.loss_config.template_loss.enabled:
                    writer.add_scalar(f'Training (step) -model- Loss_template', loss_tpl.item(), model_engine.global_steps - 1)

                # Visualize samples for rank 0
                if model_engine.global_steps % args.train_config.viz_freq_train_step == 0:
                    save_viz(model_engine, epoch, cover_img, output, args, split_set="train")
                    
                # Print memory usage
                if args.train_config.print_memory:
                    print_interval = 100
                    if torch.cuda.is_available() and (step % print_interval == 0 or step == len(bar) - 1):
                        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        current_memory_mb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        print(f"\n[Epoch {epoch}, Step {step}] "
                                f"Peak Memory: {peak_memory_mb:.4f} GB | "
                                f"Current Memory: {current_memory_mb:.4f} GB | "
                                f"Loss: {loss.item():.4f}")
                        torch.cuda.reset_peak_memory_stats(device)
                        
            # --- Distributed operations must be invoked on all Ranks --- #
            if model_engine.is_gradient_accumulation_boundary(): # Save only after an effective step
                current_step = model_engine.global_steps # Use deepspeed's global step counter
                # Logic for saving checkpoints is visible to all processes
                if current_step % args.train_config.save_freq_step == 0:
                    save_checkpoints(model_engine, disc_engine, args, epoch, current_step, split_set="train")
                    
        # Calculate average loss for this epoch
        epoch_loss = epoch_loss / loss_add_times
        disc_epoch_loss = disc_epoch_loss / loss_add_times
        if model_engine.global_rank == 0:
            writer.add_scalar(f'Training (epoch) -model- Loss', epoch_loss, epoch)
            if args.loss_config.gan_loss.enabled:
                writer.add_scalar(f'Training (epoch) -Net_Disc- Loss', disc_epoch_loss, epoch)
            bar.set_description("Train epoch[{}/{}] loss_model:{:.3f} loss_net_disc:{:.3f}".format(epoch, 
                                                                                                start_epoch + args.train_config.num_epochs - 1, 
                                                                                                epoch_loss,
                                                                                                disc_epoch_loss)
                                )
            if epoch % args.train_config.viz_freq_train_epoch == 0:
                save_viz(model_engine, epoch, cover_img, output, args, split_set="train")

        # Save the checkpoint and visualize samples
        if model_engine.is_gradient_accumulation_boundary(): # Save only after an effective step
            current_step = model_engine.global_steps # Use deepspeed's global step counter
            if current_step % args.train_config.save_freq_epoch == 0:
                save_checkpoints(model_engine, disc_engine, args, epoch, current_step, split_set="train")
        
        # 2. Validation
        # Only rank 0 should run validation and print/log results
        if args.data_config.validation and epoch % args.train_config.val_freq_epoch == 0:
            def validate_model():
                model_engine.eval()
                if args.loss_config.gan_loss.enabled:
                    disc_engine.eval()
                # These accumulators are now on EACH process
                total_loss_model = 0.0
                total_loss_disc = 0.0
                total_samples = 0
                # Use a progress bar only on the main process
                if model_engine.global_rank == 0:
                    bar = tqdm(val_dataloader, desc='Validating Steps')
                else:
                    bar = val_dataloader
                # Iterate over the validation dataset
                with torch.no_grad():
                    for step, datas in enumerate(bar):
                        # Get data
                        cover_img, init_flow = datas[0].to(device), datas[1].to(device)
                        B, C, H, W = cover_img.shape
                        # Forward
                        output = model_engine(cover_img, init_flow=init_flow, content_mask=None, calc_loss=True, vae=vae, generator=g_cuda)
                        enc_img = output['enc_img'] 
                        ## (1) Calculate the image losses
                        loss_img_sim = img_sim_loss_fn(enc_img, cover_img)
                        loss_lpips = lpips_loss_fn(enc_img, cover_img)
                        if args.loss_config.gan_loss.enabled:
                            disc_fake_out = disc_engine(enc_img)
                            loss_disc_enc = gan_loss_fn(disc_fake_out['global_disc'], True) * args.model_config.disc_model.global_weight + \
                                            gan_loss_fn(disc_fake_out['local_disc'], True) * args.model_config.disc_model.local_weight
                        else:
                            loss_disc_enc = torch.tensor(0.0, device=device)
                        loss_img = loss_img_sim * args.loss_config.img_sim_loss.weight + \
                                    loss_lpips * args.loss_config.lpips_loss.weight + \
                                    loss_disc_enc * args.loss_config.gan_loss.weight 
                        ## (2) Calculate the motion losses
                        loss_motion = output['loss_motion'] * args.loss_config.motion_loss.weight
                        ## (3) Calculate the content losses
                        loss_content = output['loss_content'] * args.loss_config.content_loss.weight
                        ## (4) Calculate the template loss
                        if args.loss_config.template_loss.enabled:
                            loss_tpl = output['loss_tpl']
                        else:
                            loss_tpl = torch.tensor(0.0, device=device)
                        ## (5) Calculate the total loss
                        loss = loss_img + loss_motion + loss_content + loss_tpl * args.loss_config.template_loss.weight
                        if args.loss_config.gan_loss.enabled:
                            disc_real_out = net_disc(cover_img)
                            disc_fake_out = net_disc(enc_img.detach())
                            loss_disc_real = gan_loss_fn(disc_real_out['global_disc'], True) * args.model_config.disc_model.global_weight + \
                                            gan_loss_fn(disc_real_out['local_disc'], True) * args.model_config.disc_model.local_weight
                            loss_disc_fake = gan_loss_fn(disc_fake_out['global_disc'], False) * args.model_config.disc_model.global_weight + \
                                            gan_loss_fn(disc_fake_out['local_disc'], False) * args.model_config.disc_model.local_weight
                            loss_net_disc = (loss_disc_real + loss_disc_fake) * 0.5 
                        else:
                            loss_net_disc = torch.tensor(0.0, device=device)
                        if model_engine.global_rank == 0:
                            current_global_step = model_engine.global_steps
                            writer.add_scalar('Validate (step) / Model_Loss_local', loss.item(), current_global_step + step)
                            if args.loss_config.gan_loss.enabled:
                                writer.add_scalar('Validate (step) / Disc_Loss_local', loss_net_disc.item(), current_global_step + step)
                            if (step + 1) % args.train_config.viz_freq_val_step == 0:
                                viz_meta = {'epoch_num': epoch, 'step_num': current_global_step}
                                save_viz(model_engine, epoch, cover_img, output, args, split_set="val", val_step=step)
                        # --- Step 3: Accumulate local results for the final aggregation ---
                        total_loss_model += loss.item() * B
                        if args.loss_config.gan_loss.enabled:
                            total_loss_disc += loss_net_disc.item() * B
                        total_samples += B
                # --- Step 4: Perform ONE aggregation at the end of the epoch ---
                local_stats = torch.tensor([total_loss_model, total_loss_disc, total_samples], dtype=torch.float64, device=device)
                dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
                # --- Step 5: Log the final, precise, aggregated result (Rank 0) ---
                if model_engine.global_rank == 0:
                    if local_stats[2].item() > 0:
                        global_avg_loss_model = local_stats[0].item() / local_stats[2].item()
                        global_avg_loss_disc = local_stats[1].item() / local_stats[2].item()
                    else:
                        global_avg_loss_model = 0.0
                        global_avg_loss_disc = 0.0
                    print(f"\n--- Validation Summary for Epoch {epoch} (Global Avg) ---")
                    print(f"   Average Model Loss: {global_avg_loss_model:.4f}")
                    print(f"   Average Disc Loss: {global_avg_loss_disc:.4f}")
                    writer.add_scalar('Validate (epoch) / Model_Loss_global_avg', global_avg_loss_model, model_engine.global_steps)
                    if args.loss_config.gan_loss.enabled:
                        writer.add_scalar('Validate (epoch) / Disc_Loss_global_avg', global_avg_loss_disc, model_engine.global_steps)
            validate_model()

def save_checkpoints(model_engine, disc_engine, args, epoch_num, step_num, split_set: Literal['train', 'val']):
    try:
        # Define a tag for the checkpoint, e.g., the step number
        tag = f"epochnum-{epoch_num}_trainstep-{step_num}"
        # Create a dictionary to save additional metadata
        client_state = {'epoch_num': epoch_num, 'step_num': step_num}
        # Use the DeepSpeed save API
        save_model_dir = Path(args.train_config.checkpoints_savedir) / split_set / 'model'
        model_engine.save_checkpoint(save_dir=str(save_model_dir), tag=tag, client_state=client_state, exclude_frozen_parameters=False)
        if args.loss_config.gan_loss.enabled and disc_engine is not None:
            save_net_disc_dir = Path(args.train_config.checkpoints_savedir) / split_set / 'net_disc'
            # The discriminator engine saves its own state
            disc_engine.save_checkpoint(save_dir=str(save_net_disc_dir), tag=tag, client_state=client_state, exclude_frozen_parameters=False)
        # Only log on the main process
        if model_engine.global_rank == 0:
            print(f"Saved DeepSpeed checkpoint for {tag}")
    except Exception as e:
        print(f"Error: save checkpoint failed. Reason: {e}")

def save_viz(model_engine, epoch, cover_img, output, args, split_set: Literal['train', 'val'], val_step=None):
    try:
        save_enc_img_dir = Path(args.train_config.checkpoints_savedir) / split_set / 'viz'
        os.makedirs(save_enc_img_dir, exist_ok=True)
        for i, enc_img_i in enumerate(output['enc_img']):
            enc_img_i_pil = tensor_to_pil(enc_img_i.float())
            if split_set == "train":
                save_enc_img_i_path = str(save_enc_img_dir / f"epochnum-{epoch}_trainstep-{model_engine.global_steps}_sample-{i:02d}-img_enc.png")
            elif split_set == "val":
                save_enc_img_i_path = str(save_enc_img_dir / f"epochnum-{epoch}_trainstep-{model_engine.global_steps}_valstep-{val_step}_sample-{i:02d}-img_enc.png")
            enc_img_i_pil.save(save_enc_img_i_path)

        for i, enc_cf_mf_ds_img_i in enumerate(output['enc_cf_mf_ds_img']):
            enc_cf_mf_ds_img_i_pil = tensor_to_pil(enc_cf_mf_ds_img_i.float())
            if split_set == "train":
                save_enc_cf_mf_ds_img_i_path = str(save_enc_img_dir / f"epochnum-{epoch}_trainstep-{model_engine.global_steps}_sample-{i:02d}-img_enc_cf_mf_ds.png")
            elif split_set == "val":
                save_enc_cf_mf_ds_img_i_path = str(save_enc_img_dir / f"epochnum-{epoch}_trainstep-{model_engine.global_steps}_valstep-{val_step}_sample-{i:02d}-img_enc_cf_mf_ds.png")
            enc_cf_mf_ds_img_i_pil.save(save_enc_cf_mf_ds_img_i_path)
            
        if cover_img is not None:
            save_cover_img_dir = Path(args.train_config.checkpoints_savedir) / split_set / 'viz'
            os.makedirs(save_cover_img_dir, exist_ok=True)
            for i, cover_img_i in enumerate(cover_img):
                cover_img_i_pil = tensor_to_pil(cover_img_i.float())
                if split_set == "train":
                    save_cover_img_i_path = str(save_cover_img_dir / f"epochnum-{epoch}_trainstep-{model_engine.global_steps}_sample-{i:02d}-img_in.png")
                elif split_set == "val":
                    save_cover_img_i_path = str(save_cover_img_dir / f"epochnum-{epoch}_trainstep-{model_engine.global_steps}_valstep-{val_step}_sample-{i:02d}-img_in.png")
                cover_img_i_pil.save(save_cover_img_i_path)
        
        if output is not None:
            pred_motion = output['pred_motions'][-1].detach().float()
            true_motion = output['true_motion'].detach().float()

            # save motions (pred and true)
            viz_motions(true_motion, args.train_config.checkpoints_savedir, 'viz', "motion_true", epoch, model_engine.global_steps, split_set, val_step, viz_with='both')
            viz_motions(pred_motion, args.train_config.checkpoints_savedir, 'viz', "motion_pred", epoch, model_engine.global_steps, split_set, val_step, viz_with='both')

            # save confidences (pred and true)
            if 'pred_infos' in output:
                pred_conf = model_engine.module.calc_confidence(output['pred_infos'][-1].detach())
                viz_tensors_to_heatmap(pred_conf.float(), args.train_config.checkpoints_savedir, 'viz', "conf_pred", epoch, model_engine.global_steps, split_set, val_step, cmap_name='viridis')

            # save content forgery (pred and true)
            if 'pred_content' in output:
                pred_content = output['pred_content'].detach().float()
                true_content = output['true_content'].detach().float()
                viz_tensor_to_grey(true_content, args.train_config.checkpoints_savedir, 'viz', "content_true", epoch, model_engine.global_steps, split_set, val_step)
                viz_tensor_to_grey(pred_content, args.train_config.checkpoints_savedir, 'viz', "content_pred", epoch, model_engine.global_steps, split_set, val_step, binarize_thre=0.5)

            # viz valid_motion when valid_motion is not zero (for better visualization)
            if 'valid_motion' in output and not args.model_config.choices.non_valid_motion_to_zero:
                valid_motion = output['valid_motion'].detach().float()
                viz_tensor_to_grey(valid_motion, args.train_config.checkpoints_savedir, 'viz', "valid_motion", epoch, model_engine.global_steps, split_set, val_step)

            # save the templates
            if args.model_config.unified_template.enabled:
                if args.model_config.unified_template.viz_in_train:
                    viz_template(model_engine.module.unified_template.detach().float(), args.train_config.checkpoints_savedir, 'ut', "ut", epoch, model_engine.global_steps, split_set, val_step, args.model_config.unified_template.viz_mode)
            else:
                if args.model_config.motion_template.viz_in_train:
                    viz_template(model_engine.module.motion_template.detach().float(), args.train_config.checkpoints_savedir, 'mt', "mt", epoch, model_engine.global_steps, split_set, val_step, args.model_config.motion_template.viz_mode)
                if args.model_config.content_template.viz_in_train:
                    viz_template(model_engine.module.content_template.detach().float(), args.train_config.checkpoints_savedir, 'ct', "ct", epoch, model_engine.global_steps, split_set, val_step, args.model_config.content_template.viz_mode)
    except Exception as e:
        print(f"ERROR: Failed to save visualization. Reason: {e}")

def main(args):
    # 1. Set Environment Variables
    ## We suspect a higher-level configuration is overriding our NCCL settings.
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    
    ## Initialize the distributed environment and devices (all processes)
    deepspeed.init_distributed()
    world_size = dist.get_world_size()     # Get the total number of processes involved in training (i.e., the total number of GPUs used)
    global_rank = dist.get_rank()    # Get the global rank of the current process
    local_rank = int(os.environ['LOCAL_RANK'])    # Get the rank of the current process on the current machine (GPU number)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    ## make deterministic
    make_deterministic(args.train_config.random_seed)
    
    # 2. Update savedir
    ## Update the ckpt_savedir
    ckpt_savedir = update_checkpoints_savedir(log_dir=args.train_config.ckpt.log_dir,
                                              ckpt_path=args.train_config.ckpt.read_path,
                                              ckpt_savedir=args.train_config.ckpt.save_dir,
                                              exp_name=args.train_config.ckpt.exp_name)
    args.train_config.ckpt.save_dir = ckpt_savedir

    ## 3. Save the args
    if global_rank == 0:    # Use "global_rank == 0" rather than "local_rank == 0" for more generality
        print_all_args(args)
        save_args_path = Path(args.train_config.checkpoints_savedir) / "config.json"
        if save_args_path.exists(): print(f"config.json already exists in {args.train_config.checkpoints_savedir}.")
        else: save_namespace_to_json(args, save_args_path)
    dist.barrier()
    
    train(args, device)
    
    if global_rank == 0:
        print('Finish Training')
        
def set_args():
    parser = argparse.ArgumentParser()
    # experiment settings
    # parser.add_argument('--data_config', default='core_v8_1/configs/data_config.json')
    parser.add_argument('--loss_config', default='core_v8_1/configs/loss_config.json')
    parser.add_argument('--model_config', default='core_v8_1/configs/model_config.json')
    parser.add_argument('--train_config', default='core_v8_1/configs/train_config.json')
    parser.add_argument('--deepspeed_config', default='core_v8_1/configs/deepspeed_config.json')
    parser.add_argument('--unified_config', default=None)
    # Adding 'local_rank' here to prevent argparse from throwing an "unrecognized arguments" error.
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    args = parse_args(parser)
    # Set up the config
    if args.unified_config is not None:
        # args.data_config = args.unified_config.data_config
        args.loss_config = args.unified_config.loss_config
        args.model_config = args.unified_config.model_config
        args.train_config = args.unified_config.train_config
        args.deepspeed_config = args.unified_config.deepspeed_config
        
    if not args.loss_config.template_loss.enabled:
        args.model_config.unified_template.calc_loss = False
        args.model_config.motion_template.calc_loss = False
        args.model_config.content_template.calc_loss = False

    args.deepspeed_config = namespace_to_dict(args.deepspeed_config)
    
    return args
    
    
if __name__ == '__main__':
    args = set_args()
    main(args)