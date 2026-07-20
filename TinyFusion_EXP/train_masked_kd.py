# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import random
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import wandb

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, labels_dir, flip=0):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.flip = flip

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))

        if self.flip>0:
            if random.random() < self.flip:
                features = features[1:]
            else:
                features = features[:1]
        return torch.from_numpy(features), torch.from_numpy(labels)
    
# ==== MemmapDataset (features.npy / labels.npy 사용) ====
class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, feat_path, labl_path, flip=0.0):
        # 하나의 큰 npy를 mmap으로 read-only 매핑 (OS 페이지캐시 공유)
        self.F = np.load(feat_path, mmap_mode="r", allow_pickle=False)  # (N, ...)
        self.L = np.load(labl_path, mmap_mode="r", allow_pickle=False)  # (N, ...)
        assert len(self.F) == len(self.L), "features/labels length mismatch"
        self.flip = float(flip)

    def __len__(self): return self.F.shape[0]

    def __getitem__(self, idx):
        f = self.F[idx]  # view
        l = self.L[idx]
        if self.flip > 0.0:
            f = f[1:] if random.random() < self.flip else f[:1]
        # zero-copy 텐서 (안전하게 copy=False)
        return torch.from_numpy(np.array(f, copy=False)), torch.from_numpy(np.array(l, copy=False))

# ==== 재현성: 각 worker에 결정적 seed 셋 ====
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed); torch.manual_seed(worker_seed)

# ==== 헬퍼: memmap 존재하면 True ====
def has_memmap(data_path: str):
    fp = os.path.join(data_path, "features.npy")
    lp = os.path.join(data_path, "labels.npy")
    return os.path.isfile(fp) and os.path.isfile(lp)

class CustomDataset_v2(torch.utils.data.Dataset):
    def __init__(self, features_dir, labels_dir, flip=0):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.flip = flip

        self._feature_cache = {}
        self._label_cache = {}

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        if idx in self._feature_cache:
            features = self._feature_cache[idx]
            labels = self._label_cache[idx]
        else:
            feature_file = self.features_files[idx]
            label_file = self.labels_files[idx]

            features = np.load(os.path.join(self.features_dir, feature_file))
            labels = np.load(os.path.join(self.labels_dir, label_file))

            self._feature_cache[idx] = features
            self._label_cache[idx] = labels
        if self.flip > 0:
            if random.random() < self.flip:
                features = features[1:]
            else:
                features = features[:1]

        return torch.from_numpy(features), torch.from_numpy(labels)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def print_rank_0(msg):
    if dist.get_rank() == 0:
        print(msg)

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    # print("RANK", os.getenv("RANK"), "LOCAL_RANK", os.getenv("LOCAL_RANK"),
    #   "cuda:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)    
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    # Setup an experiment folder:
    
    if args.teacher == 'DiT-XL/2' and args.model == 'DiT-D7/2':
        args.prefix = args.prefix + '_KD_from_D28'
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}/{args.prefix}"  # Create an experiment folder
        args.experiment_dir=experiment_dir
        
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        print_rank_0(f"Experiment directory created at {experiment_dir}")
    
        # if rank != 0:
        #     model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        #     experiment_dir = f"{args.results_dir}/{model_string_name}/{args.prefix}"  # Create an experiment folder
        #     args.experiment_dir=experiment_dir

    if rank == 0:
        wandb_prefix = args.prefix.replace('/','_')
        if args.prefix is not None:
            wandb_resume = os.path.join(experiment_dir, wandb_prefix +'.wandb_id')
            if os.path.exists(wandb_resume):
                # read the wandb id from the txt file
                with open(wandb_resume, 'r') as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with open(wandb_resume, 'w') as f:
                    f.write(wandb_id)
            wandb.init(project="TinyFusion", name=wandb_prefix , id=wandb_id, resume="allow", config=args)
        else:
            wandb.init(project="TinyFusion", name=wandb_prefix , config=args)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    print("latent_size is", latent_size)
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    teacher = DiT_models[args.teacher](
        input_size=latent_size,
        num_classes=args.num_classes
    ).eval()

    for p in teacher.parameters():
        p.requires_grad = False

    def feature_hook(module, input, output):
        module.feature = output
    
    if args.model == 'DiT-D14/2':
        student_layers = list(range(0, 14))
        teacher_layers = list(range(1, 28, 2))
    elif args.model == 'DiT-D19/2':
        student_layers = [2, 5, 7, 10, 12, 15, 18]
        teacher_layers = [3, 7, 11, 15, 19, 23, 27]
    elif args.model == 'DiT-D7/2':
        student_layers = list(range(0, 7))
        if args.teacher == 'DiT-XL/2':
            print("Using D28 teacher")
            # teacher_layers = list(range(1, 28, 4))
            teacher_layers = list(range(3, 28, 4))
        else:
            teacher_layers = list(range(1, 14, 2))
    elif args.model == 'DiT-D4/2':
        student_layers = list(range(0, 4))
        if args.teacher == 'DiT-XL/2':
            teacher_layers = list(range(1, 28, 7))
        else:
            teacher_layers = [1,3,4,6]
        # teacher_layers = [0, 2, 4, 5] # [1,3,5,6]
    #student_layers = [13]
    #teacher_layers = [27]
    print(student_layers)
    print(teacher_layers)

    for slayer, tlayer in zip(student_layers, teacher_layers):
        model.blocks[slayer].target_layer = tlayer

    if args.load_weight is not None:
        initial_ckpt = torch.load(args.load_weight, map_location='cpu')

        if 'ema' in initial_ckpt:
            model.load_state_dict(initial_ckpt['ema'])
            print_rank_0(f"Loaded initial EMA weights from {args.load_weight}")
        elif 'model' in initial_ckpt:
            model.load_state_dict(initial_ckpt['model'])
            print_rank_0(f"Loaded initial weights from {args.load_weight}")
        else:
            model.load_state_dict(initial_ckpt)
            print_rank_0(f"Loaded plain weights from {args.load_weight}")

    if args.load_teacher is not None:
        initial_ckpt = torch.load(args.load_teacher, map_location='cpu',weights_only=False)
        if 'ema' in initial_ckpt:
            teacher.load_state_dict(initial_ckpt['ema'])
            print_rank_0(f"Loaded Teacher initial EMA weights from {args.load_teacher}")
        elif 'model' in initial_ckpt:
            teacher.load_state_dict(initial_ckpt['model'])
            print_rank_0(f"Loaded Teacher initial weights from {args.load_teacher}")
        else:
            teacher.load_state_dict(initial_ckpt)
            print_rank_0(f"Loaded Teacher plain weights from {args.load_teacher}")

    for i, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
        model.blocks[student_layer].register_forward_hook(feature_hook)
        teacher.blocks[teacher_layer].register_forward_hook(feature_hook)
        
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    teacher = teacher.to(device) # no ddp for teacher
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print_rank_0(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    features_dir = f"{args.data_path}/imagenet256_features"
    labels_dir = f"{args.data_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir, flip=0.5)
    # if has_memmap(args.data_path):
    #     dataset = MemmapDataset(
    #         os.path.join(args.data_path, "features.npy"),
    #         os.path.join(args.data_path, "labels.npy"),
    #         flip=0.5
    #     )
    # else:
    #     features_dir = f"{args.data_path}/imagenet256_features"
    #     labels_dir   = f"{args.data_path}/imagenet256_labels"
    #     dataset = CustomDataset(features_dir, labels_dir, flip=0.5)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True
    )
    print_rank_0(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = 2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader), eta_min=lr*0.5)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    if args.resume is not None:
        from packaging import version
        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            checkpoint = torch.load(args.resume, map_location=f'cuda:{device}', weights_only=False)
        else:
            checkpoint = torch.load(args.resume, map_location=f'cuda:{device}')
        # checkpoint = torch.load(args.resume, map_location=f'cuda:{device}')
        # print(checkpoint.keys())
        if args.results_dir == 'outputs_resume':
            start_epoch = 0
            train_steps = 0
            model.module.load_state_dict(checkpoint)
        else:
            model.module.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            opt.load_state_dict(checkpoint["opt"])
            sched.load_state_dict(checkpoint["sched"])
            
            train_steps = checkpoint["train_steps"]
            start_epoch = train_steps // len(loader) 
        if rank == 0:
                print_rank_0(f"Resume training from {args.resume}")
        del checkpoint
    else:
        start_epoch = 0
        train_steps = 0
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_feat_loss = 0
    running_kd_loss = 0
    running_gt_loss = 0
    running_lift_place = 0
    running_outkd = 0
    running_sched = 0
    running_recon = 0
    running_slope = 0
    running_intercept = 0
    start_time = time()

    print_rank_0(f"Training for {args.epochs} epochs ({args.epochs * len(loader)} steps)...")
    scaler = torch.cuda.amp.GradScaler()
    init_beta_feature = 1e-2
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        print_rank_0(f"Beginning epoch {epoch}...")
        beta_feat = init_beta_feature * (1 - epoch/(args.epochs-1))
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)

            opt.zero_grad()
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            args.cur_step = train_steps
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = diffusion.training_losses_kd_rep(model, teacher, x, t, model_kwargs, beta_feat=beta_feat, args=args)
                loss = loss_dict["loss"].mean()
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            update_ema(ema, model.module, decay=args.ema_decay)

            # Log loss values:
            running_loss += loss.item()
            running_kd_loss += loss_dict["kd_loss"].mean().item()
            running_feat_loss += loss_dict["feat_loss"].mean().item()
            running_gt_loss += loss_dict["gt"].mean().item()
            # for schedule ours
            if args.loss_type != 'logit':
                running_lift_place += loss_dict["LIFT_PLACE"].mean().item()
                running_outkd += loss_dict["OutKD"].mean().item()
                # running_sched += loss_dict["Schedule_lambda"].mean().item()
                running_recon += loss_dict["recon"].mean().item()
                running_slope += loss_dict["slope"].mean().item()
                running_intercept += loss_dict["intercept"].mean().item()

            log_steps += 1
            train_steps += 1
            # if train_steps % 50 == 0:
            #     print(f"[rank {rank}] step {train_steps} alive", flush=True)
            if train_steps % args.log_every == 0 or train_steps <=200:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_kd_loss = torch.tensor(running_kd_loss / log_steps, device=device)
                avg_feat_loss = torch.tensor(running_feat_loss / log_steps, device=device)
                avg_gt_loss = torch.tensor(running_gt_loss / log_steps, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_kd_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_feat_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_gt_loss, op=dist.ReduceOp.SUM)
                
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_kd_loss = avg_kd_loss.item() / dist.get_world_size()
                avg_feat_loss = avg_feat_loss.item() / dist.get_world_size()
                avg_gt_loss = avg_gt_loss.item() / dist.get_world_size()

                # for schedule ours
                avg_lift_place = torch.tensor(running_lift_place / log_steps, device=device)
                avg_outkd = torch.tensor(running_outkd / log_steps, device=device)
                # avg_sched = torch.tensor(running_sched / log_steps, device=device)
                avg_recon = torch.tensor(running_recon / log_steps, device=device)
                avg_slope = torch.tensor(running_slope / log_steps, device=device)
                avg_intercept = torch.tensor(running_intercept / log_steps, device=device)
                
                dist.all_reduce(avg_lift_place, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_outkd, op=dist.ReduceOp.SUM)
                # dist.all_reduce(avg_sched, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_recon, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_slope, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_intercept, op=dist.ReduceOp.SUM)
                
                avg_lift_place = avg_lift_place.item() / dist.get_world_size()
                avg_outkd = avg_outkd.item() / dist.get_world_size()
                # avg_sched = avg_sched.item() / dist.get_world_size()
                avg_recon = avg_recon.item() / dist.get_world_size()
                avg_slope = avg_slope.item() / dist.get_world_size()
                avg_intercept = avg_intercept.item() / dist.get_world_size()
                if args.loss_type in ["interpolate_patch_1_1"]:
                    print_rank_0(f"(step={train_steps:07d}) Train Loss: {avg_gt_loss:.4f}, Total Loss: {avg_loss:.4f}, KD Loss: {avg_kd_loss:.4f}, Intercept: {avg_intercept:.4f}, Slope: {avg_slope:.4f}, LIFT/PLACE Loss: {avg_lift_place:.4f}, OutKD Loss: {avg_outkd:.4f}, lamb_sched: {avg_sched:.4f}, Feat Loss: {avg_feat_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {opt.param_groups[0]['lr']:.6f}")  
                elif args.loss_type in ['patch', 'reg_const', 'l2_reg_const','l1_reg_const','l2_reg_const_conv']:
                    print_rank_0(f"(step={train_steps:07d}) Train Loss: {avg_gt_loss:.4f}, Total Loss: {avg_loss:.4f}, KD Loss: {avg_kd_loss:.4f}, Intercept: {avg_intercept:.4f}, Slope: {avg_slope:.4f}, LIFT/PLACE Loss: {avg_lift_place:.4f}, Recon Loss: {avg_recon:.4f}, Feat Loss: {avg_feat_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {opt.param_groups[0]['lr']:.6f}")                
                elif 'reg_const' in args.loss_type:
                    print_rank_0(f"(step={train_steps:07d}) Train Loss: {avg_gt_loss:.4f}, Total Loss: {avg_loss:.4f}, KD Loss: {avg_kd_loss:.4f}, Intercept: {avg_intercept:.4f}, Slope: {avg_slope:.4f}, LIFT/PLACE Loss: {avg_lift_place:.4f}, Recon Loss: {avg_recon:.4f}, Feat Loss: {avg_feat_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {opt.param_groups[0]['lr']:.6f}")                
                else:
                    print_rank_0(f"(step={train_steps:07d}) Train Loss: {avg_gt_loss:.4f}, Total Loss: {avg_loss:.4f}, KD Loss: {avg_kd_loss:.4f}, Feat Loss: {avg_feat_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {opt.param_groups[0]['lr']:.6f}")
                # Reset monitoring variables:
                running_loss = 0
                running_feat_loss = 0
                running_kd_loss = 0
                running_gt_loss = 0
                
                running_lift_place = 0
                running_outkd = 0
                running_sched = 0
                running_recon = 0
                running_slope = 0
                running_intercept = 0

                log_steps = 0
                start_time = time()

                if rank == 0:
                    try:
                        wandb.log({"train_loss": avg_gt_loss, "total_loss": avg_loss, "kd_loss": avg_kd_loss, "feat_loss": avg_feat_loss, "misc/steps_per_sec": steps_per_sec, "misc/lr": opt.param_groups[0]['lr']}, step=train_steps)
                    except:
                        pass

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "train_steps": train_steps,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print_rank_0(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "train_steps": train_steps,
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print_rank_0(f"Saved checkpoint to {checkpoint_path}")
    dist.barrier()

    print_rank_0("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--teacher", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--load-weight", type=str, default=None)
    parser.add_argument("--load-teacher", type=str, default=None)
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--strong-aug", action='store_true', default=False)
    parser.add_argument("--loss_type",type=str,default="okd",help=("distill loss type"),)
    parser.add_argument("--lamb",type=float,default=1.,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--nseg",type=int,default=4,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    args = parser.parse_args()
    main(args)