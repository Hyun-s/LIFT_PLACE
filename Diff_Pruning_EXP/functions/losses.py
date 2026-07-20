import torch
from torch import nn

# print(teacher.device)
import logging
import time 
import math, json, os
import numpy as np
import matplotlib.pyplot as plt

def get_patch_size(cur_iter: int,
                   max_iter: int,
                   start_size: int = 64,
                   beta: float = 0.5, use_outputkd=False) -> int:
    """
    cur_iter   : 현재 iteration (0 ≤ cur_iter ≤ max_iter)
    max_iter   : 총 iteration 수
    start_size : 초기 patch size (반씩 줄여서 1까지)
    beta       : <1: 초반 급감→후반 완만, >1: 후반 급감
    """
    # 1) levels 자동 생성: start_size → 1까지 절반으로
    levels = []
    s = start_size
    if use_outputkd:
        while s >= 1:
            levels.append(int(s))
            s //= 2
    else:
        while s > 1:
            levels.append(int(s))
            s //= 2
    # 2) 변곡점(thresholds) 계산
    L = len(levels)
    thresholds = [
        int((1 - (1 - i / L) ** beta) * max_iter)
        for i in range(L)
    ]
    # 3) 뒤에서부터 threshold 넘는 첫 레벨 리턴
    for level, th in reversed(list(zip(levels, thresholds))):
        if cur_iter >= th:
            return level
    return levels[0]


def _infer_total_steps(args):
    if args.pruning_ratio == 0.3:
        return 100_000 if "celeba" in args.config else 200_000
    elif args.pruning_ratio == 0.5:
        return 200_000 if "celeba" in args.config else 500_000
    elif args.pruning_ratio == 0.7:
        return 240_000 if "celeba" in args.config else 500_000
    elif args.pruning_ratio == 0.9:
        return 500_000
    else:
        raise ValueError("unexpected pruning_ratio")
    
def save_schedule_and_args(args):
    # ── (a) 로그 폴더 준비
    os.makedirs(args.log_path, exist_ok=True)

    # ── (b) args JSON 저장
    json_path = os.path.join(args.log_path, "args.json")
    with open(json_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f"[ARGS]       saved → {json_path}")

    # ── (c) 전체 λ 스케줄 샘플 & 플롯
    total_steps = _infer_total_steps(args)
    xs = np.arange(0, total_steps + 1, max(1, total_steps // 600))
    ys = [sigmoid_decay_schedule(int(s), args=args) for s in xs]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, linewidth=2)
    plt.title(f"λ schedule (prune {args.pruning_ratio}, {args.config})")
    plt.xlabel("step"); plt.ylabel("λ"); plt.grid(True); plt.tight_layout()

    png_path = os.path.join(args.log_path, "lambda_schedule.png")
    plt.savefig(png_path)
    plt.close()
    logging.info(f"[λ‑schedule] saved → {png_path}")
    
    
def sigmoid_decay_schedule(step: int,
                           min_lambda: float = 0.1,
                           max_lambda: float = 1.0,
                           sharpness: float = 12.0,
                           args=0) -> float:
    if args.pruning_ratio == 0.3:
        if 'celeba' in args.config:
            total_steps=100000
        elif 'bedroom' in args.config:
            total_steps=200000
    elif args.pruning_ratio == 0.5:
            if 'celeba' in args.config:
                total_steps=200000
            elif 'bedroom' in args.config:
                total_steps=500000
    elif args.pruning_ratio == 0.7:
            if 'celeba' in args.config:
                total_steps=240000
            elif 'bedroom' in args.config:
                total_steps=500000
    elif args.pruning_ratio == 0.9:
            if 'celeba' in args.config:
                total_steps=500000
            elif 'bedroom' in args.config:
                total_steps=500000
        
    midpoint = total_steps / 2
    k  = sharpness / total_steps                # 기울기 조절
    sig = 1 / (1 + math.exp(-k * (step - midpoint)))   # 0→1
    return max_lambda - (max_lambda - min_lambda) * sig # 1→0.1



def distill(teacher, model, t_modules, s_modules, noisy_images, timesteps):
    t_features = []
    s_features = []
    def t_forward_hook(module, input, output):
        t_features.append(output.detach())
    def s_forward_hook(module, input, output):
        s_features.append(output)
    hooks = []
    for t_module, s_module in zip(t_modules, s_modules):
        hooks.append(t_module.register_forward_hook(t_forward_hook))
        hooks.append(s_module.register_forward_hook(s_forward_hook))
        
    model_output = model(noisy_images, timesteps).sample
    with torch.no_grad():
        t_model_output = teacher(noisy_images, timesteps).sample

    for hook in hooks:
        hook.remove()    
    return model_output,t_model_output, s_features, t_features


def distill_(teacher, model, noisy_images, timesteps, t_modules=None, s_modules=None):
    model_output = model(noisy_images, timesteps)#.sample
    with torch.no_grad():
        t_model_output = teacher(noisy_images, timesteps)#.sample        
    return model_output, t_model_output



class FitNetLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.adapters = nn.ModuleList()

    def construct_adapter(self, teacher, model,
                          t_modules, s_modules, 
                          noisy_images, timesteps):
        # _, _, t_feat, s_feat = distill(teacher, model, t_modules, s_modules, noisy_images, timesteps)
        _, _, s_feat, t_feat = hook_forward(teacher,model, noisy_images, timesteps)
        self.adapters = nn.ModuleList()

        for tf, sf in zip(t_feat, s_feat):
            c_out = sf.shape[1]
            c_in = tf.shape[1]
            conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
            conv.weight.data.uniform_(-0.005, 0.005)
            self.adapters.append(conv)
        self.to(self.device)

    def forward(self, s_feat, t_feat):
        loss = 0
        # print(len(s_feat),len(t_feat))
        for k,(s_f, t_f, adapter) in enumerate(zip(s_feat, t_feat, self.adapters)):
            adapted_s_f = adapter(s_f)
            loss += ((adapted_s_f - t_f)**2).sum(dim=(1, 2, 3)).mean(dim=0)
            # print(k, ":",((adapted_s_f - t_f).pow(2)).mean())
        return loss.mean()




class distill_loss():
    def __init__(self, loss_type, lamb, device,
                 teacher, model,
                 t_modules, s_modules, 
                 noisy_images, timesteps,args=None):
        self.loss_type = loss_type
        self.lamb = lamb
        self.device= device
        self.args = args
        
        print(self.args)
        if 'fitnet' in self.loss_type:
            self.loss = FitNetLoss(self.device) 
            self.loss.construct_adapter(teacher, model,
                          t_modules, s_modules, 
                          noisy_images, timesteps)
        elif self.loss_type == "imp_our_fitnet_learnable":
            self.learnable_w = LearnableReconWeight(init_value=0.)
        self.n = 0
    def __call__(self, model_output,t_model_output, s_features, t_features,x_t,timesteps,b,a,step=0):
 
        if self.loss_type == "logit":
            return (t_model_output - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0) * (self.lamb)
        elif self.loss_type == 'fitnet':
            return self.loss(s_features, t_features) * (self.lamb) #* 1e-6
        elif self.loss_type == 'LIFT_PLACE_fitnet': # LIFT
            patch_loss =  imp_regression_constraint_loss(model_output, t_model_output,patch_size=self.args.patch_size, args=self.args) * self.lamb
            fitnet_loss = self.loss(s_features, t_features) * 1e-6
            return patch_loss + fitnet_loss
        elif self.loss_type == 'logit_fitnet':
            output_loss = (t_model_output - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0) * (self.lamb)
            fitnet_loss = self.loss(s_features, t_features) * 1e-6
            return output_loss + fitnet_loss

        
        else:
            raise Exception("Unexpected loss type, (logit, fitnet, okd, itgc)")




def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False, args=None,return_eps=False):
    
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if return_eps:
        if args.loss_type == 'finetune':
            if keepdim:
                return (e - output).square().sum(dim=(1, 2, 3)), output
            else:
                return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
        
    else:
        if args.loss_type == 'finetune':
            if keepdim:
                return (e - output).square().sum(dim=(1, 2, 3)), spline, normal
            else:
                # print('finetune loss')
                return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) #, spline, normal
        elif args.loss_type == 'pruning':
            if keepdim:
                return (e - output).square().sum(dim=(1, 2, 3))#, spline, normal
            else:
                # print('finetune loss')
                return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)#, spline, normal

def noise_estimation_kd_loss(model,
                             teacher,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, 
                          kd_loss: None,
                          keepdim=False,step=0):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    batch = x.shape[0]
    output, teacher_output, s_features, t_features = hook_forward(model,teacher, x, t)
    orm_loss = kd_loss(output,teacher_output, s_features, t_features,x,t,b,a,step=step)
    if len(output) == 4:
        output = output[-1]

    distance = (teacher_output - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) , orm_loss, distance


loss_registry = {
    'simple': noise_estimation_loss,
}




def hook_modules(model):
    model = model.module if hasattr(model, "module") else model
    
    if hasattr(model, "denoiser"):
        model = model.denoiser
    else:
        model = model
    t_modules = []
    for module_name, module in model.mid.named_children():
        t_modules.append(module)

    for b in model.down:
        for module_name, module in b.named_children():
            for submodule_name, submodule in module.named_children():
                t_modules.append(submodule)

    for b in model.up:
        for module_name, module in b.named_children():
            for submodule_name, submodule in module.named_children():
                t_modules.append(submodule)
    return t_modules

def hook_forward(students,teacher, x, t):
    teacher = teacher.module if hasattr(teacher, "module") else teacher
    student = students.module if hasattr(students, "module") else students
    state = False
    t_features = []
    s_features = []
    def t_forward_hook(module, input, output):
        t_features.append(output.detach())
    def s_forward_hook(module, input, output):
        s_features.append(output)
    def hook_modules(model):
        t_modules = []
        for module_name, module in model.mid.named_children():
            # for submodule_name, submodule in module.named_children():
            t_modules.append(module)

        for b in model.down:
            for module_name, module in b.named_children():
                for submodule_name, submodule in module.named_children():
                    t_modules.append(submodule)

        for b in model.up:
            for module_name, module in b.named_children():
                for submodule_name, submodule in module.named_children():
                    t_modules.append(submodule)
        return t_modules
    t_modules = hook_modules(teacher)
    s_modules = hook_modules(student)
    hooks = []
    for t_module, s_module in zip(t_modules, s_modules):
        hooks.append(t_module.register_forward_hook(t_forward_hook))
        hooks.append(s_module.register_forward_hook(s_forward_hook))
    if state:
        output = students(x, t.float(),return_coef=True)
    else:
        output = student(x, t.float())
    
    
    with torch.no_grad():
        teacher_output = teacher(x, t.float())
    for hook in hooks:
        hook.remove()

    return output, teacher_output, s_features, t_features




def regression_constraint_loss(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 16,
    eps: float = 1e-4,
    args = None
) -> torch.Tensor:
    B, C, H, W = et_s.shape

    assert H % patch_size == 0 and W % patch_size == 0, \
        f"H ({H}) and W ({W}) must be divisible by patch_size ({patch_size})"

    n_h = H // patch_size
    n_w = W // patch_size
    s_patches = (
        et_s
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
      
    )
    t_patches = (
        et_t
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
    )

    n_patches = n_h * n_w
    s_flat = s_patches.contiguous()\
        .view(B, C, n_patches, patch_size * patch_size)
    t_flat = t_patches.contiguous()\
        .view(B, C, n_patches, patch_size * patch_size)

    x_mean = s_flat.mean(dim=-1, keepdim=True)   # [B, C, n_patches, 1]
    y_mean = t_flat.mean(dim=-1, keepdim=True)
    cov    = ((s_flat - x_mean) * (t_flat - y_mean)).sum(dim=-1, keepdim=True)
    var    = ((s_flat - x_mean) ** 2).sum(dim=-1, keepdim=True)


    slope     = (cov / (var + eps)).squeeze(-1)       # [B, C, n_patches]
    intercept = (y_mean - slope.unsqueeze(-1) * x_mean).squeeze(-1)
    # print(slope.max().item())
    # loss_s = (slope - 1.0).sum(2)
    # loss_i = intercept.sum(2)
    s_target = torch.ones_like(slope)
    i_target = torch.zeros_like(intercept)
    # print
    eps_hat = s_flat * slope.detach().unsqueeze(-1) + intercept.detach().unsqueeze(-1)
    recon_loss = ((eps_hat - t_flat)**2)
    # loss_s = ((slope - s_target)).sum(dim=(1, 2)).mean(dim=0)
    # loss_i = ((intercept - i_target)).sum(dim=(1, 2)).mean(dim=0)
    
    sch = (torch.abs((slope - s_target))).mean().item() + (torch.abs((intercept - i_target))).mean().item()

    if args.use_l2:
        loss_s = (torch.abs((slope - s_target))).mean(dim=(1, 2)).sum(dim=0)
        loss_i = (torch.abs((intercept - i_target))).mean(dim=(1, 2)).sum(dim=0)
        recon_loss = ((eps_hat - t_flat)**2).mean(dim=(1, 2,3)).sum(dim=0)
        
    else:
        loss_s = (torch.abs(slope - s_target)).mean(dim=(1, 2)).sum(dim=0)
        loss_i = (torch.abs(intercept - i_target)).mean(dim=(1, 2)).sum(dim=0)
        recon_loss = ((eps_hat - t_flat)**2).mean(dim=(1, 2,3)).sum(dim=0)
    return (loss_s+ loss_i)*(patch_size**2) + (1-min(1, sch))*recon_loss #,(1-min(1, sch))*recon_loss

import torch.nn.functional as F

@torch.no_grad()
def _ones_kernel(C, p, device, dtype):
    k = torch.ones((C, 1, p, p), device=device, dtype=dtype)
    return k




import torch

def imp_regression_constraint_loss_LIFT(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    mode: str = 'channel', # 'sample' (B개) 또는 'channel' (B*C개)
    eps: float = 1e-7,
    patch_size : int = 32,
    args = None,
    curriculum = True,
    ablation = None
) -> tuple:
    
    B, C, H, W = et_s.shape
    
    if mode == 'sample':
        s_grouped = et_s.view(B, 1, -1)
        t_grouped = et_t.view(B, 1, -1)
    else: 
        s_grouped = et_s.view(B, C, -1)
        t_grouped = et_t.view(B, C, -1)

    x_mean = s_grouped.mean(dim=-1, keepdim=True)
    y_mean = t_grouped.mean(dim=-1, keepdim=True)
    
    cov = ((s_grouped - x_mean) * (t_grouped - y_mean)).sum(dim=-1)
    var = ((s_grouped - x_mean) ** 2).sum(dim=-1)

    slope = cov / (var + eps)
    intercept = y_mean.squeeze(-1) - slope * x_mean.squeeze(-1)

    s_target = torch.ones_like(slope)
    i_target = torch.zeros_like(intercept)

    eps_hat = s_grouped * slope.detach().unsqueeze(-1) + intercept.detach().unsqueeze(-1)
    recon_loss = ((eps_hat - t_grouped) ** 2) # [B, C or 1, N]

    if curriculum:
        sch = (torch.abs(slope - s_target)).mean().item() + (torch.abs(intercept - i_target)).mean().item()
    else:
        sch = 1.0

    use_l2 = getattr(args, 'use_l2', False)
    
    if use_l2:
        loss_s = ((slope - s_target)**2).sum(dim=1).mean()
        loss_i = ((intercept - i_target)**2).sum(dim=1).mean()
        recon_loss = recon_loss.sum(dim=(1, 2)).mean()
    else:
        loss_s = torch.abs(slope - s_target).sum(dim=1).mean()
        loss_i = torch.abs(intercept - i_target).sum(dim=1).mean()
        recon_loss = recon_loss.sum(dim=(1, 2)).mean()
        
    tot = patch_size
    total_elements = tot if mode == 'channel' else C * tot
    
    if ablation == 'coarse':
        return (loss_s + loss_i) * total_elements
    elif ablation == 'fine':
        return recon_loss
    else:
        return (loss_s + loss_i) * total_elements + (1 - min(1, sch)) * recon_loss

import torch



# v2
def LIFT_PLACE(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 16,
    eps: float = 1e-7,
    args = None,
    curriculum = True,
    ablation = None
) -> torch.Tensor:
    
    B, C, H, W = et_s.shape
    total_pixels = H * W
    num_groups = total_pixels // patch_size

    diff = et_t - et_s
    importance_flat = (diff).pow(2).view(B, C, -1)
    
    s_flat = et_s.view(B, C, -1)
    t_flat = et_t.view(B, C, -1)

    sorted_indices = torch.argsort(importance_flat, dim=-1)
    s_sorted = torch.gather(s_flat, dim=-1, index=sorted_indices)
    t_sorted = torch.gather(t_flat, dim=-1, index=sorted_indices)

    s_grouped = s_sorted.view(B, C, num_groups, patch_size)
    t_grouped = t_sorted.view(B, C, num_groups, patch_size)
    
    s_var, s_mean = torch.var_mean(s_grouped, dim=-1, keepdim=True, correction=0)
    t_mean = t_grouped.mean(dim=-1, keepdim=True)
    

    cov = ((s_grouped - s_mean) * (t_grouped - t_mean)).mean(dim=-1)
    var = s_var.squeeze(-1) 

    slope = cov / (var + eps)
    intercept = t_mean.squeeze(-1) - slope * s_mean.squeeze(-1)

    s_target = 1.0
    i_target = 0.0

    eps_hat = s_grouped * slope.detach().unsqueeze(-1) + intercept.detach().unsqueeze(-1)
    recon_loss_map = (eps_hat - t_grouped).pow(2)
    
    recon_loss = recon_loss_map.sum(dim=(1, 2, 3)).mean(dim=0)

    diff_s = slope - s_target
    diff_i = intercept - i_target
    
    if curriculum:
        sch = diff_s.abs().mean() + diff_i.abs().mean()
    else:
        sch = 1.0

    if args.use_l2:
        loss_s = diff_s.pow(2).sum(dim=(1, 2)).mean(dim=0)
        loss_i = diff_i.pow(2).sum(dim=(1, 2)).mean(dim=0)
    else:
        loss_s = diff_s.abs().sum(dim=(1, 2)).mean(dim=0)
        loss_i = diff_i.abs().sum(dim=(1, 2)).mean(dim=0)

    # for ablation
    if ablation is None:
        return (loss_s + loss_i) * patch_size + (1.0 - min(1.0, float(sch))) * recon_loss
    elif ablation == 'coarse':
        return (loss_s + loss_i) * patch_size
    elif ablation == 'fine':
        return recon_loss





import os
import csv
import time

def append_loss_log_csv(args, loss_s, loss_i, sch, recon_loss):
    """
    Append one row to args.log_path/log.csv
    Works even if loss_s/loss_i/recon_loss are torch scalars.
    """

    # (선택) DDP에서 rank0만 로깅
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return
    except Exception:
        pass

    log_dir = getattr(args, "log_path", None)
    if log_dir is None:
        return  # log_path 없으면 조용히 스킵

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.csv")

    # step 관리: args.global_step 있으면 쓰고, 없으면 내부 카운터 사용
    step = getattr(args, "global_step", None)
    if step is None:
        step = getattr(args, "_loss_log_step", 0)
        setattr(args, "_loss_log_step", step + 1)

    # 자주 호출되면 IO가 부담될 수 있으니, (선택) log_every로 줄이기
    # log_every = getattr(args, "log_every", 1)
    log_every = 100
    if (step % log_every) != 0:
        return

    # torch scalar -> python float
    def to_float(x):
        try:
            return float(x.detach().item())
        except Exception:
            return float(x)

    row = {
        "time": time.time(),
        "step": int(step),
        "loss_s": to_float(loss_s),
        "loss_i": to_float(loss_i),
        "sch": float(sch),  # 이미 .item()으로 float이면 OK
        "recon_loss": to_float(recon_loss),
    }

    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def total_iters_from_pruning(pruning_ratio: float) -> int:
    # float 비교 이슈 방지
    pr = round(float(pruning_ratio), 1)
    table = {0.5: 200_000, 0.7: 240_000, 0.9: 500_000}
    if pr not in table:
        raise ValueError(f"Unsupported pruning_ratio={pr}. Supported: {list(table.keys())}")
    return table[pr]
import math
import torch

def _to_tensor_scalar(x, ref: torch.Tensor):
    return torch.tensor(float(x), device=ref.device, dtype=ref.dtype)

def cosine_schedule(step: int, total_steps: int, start=0.0, end=1.0, direction="increase"):
    step = max(0, min(step, total_steps))
    progress = step / max(1, total_steps)  # 0..1

    if direction == "increase":
        # 0 -> 1
        base = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif direction == "decrease":
        # 1 -> 0
        base = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError("direction must be 'increase' or 'decrease'")

    return start + (end - start) * base

def linear_schedule(step: int, total_steps: int, start=0.0, end=1.0, direction="increase"):
    step = max(0, min(step, total_steps))
    progress = step / max(1, total_steps)  # 0..1

    if direction == "increase":
        base = progress  # 0->1
    elif direction == "decrease":
        base = 1.0 - progress  # 1->0
    else:
        raise ValueError("direction must be 'increase' or 'decrease'")

    return start + (end - start) * base

class LearnableReconWeight(nn.Module):
    def __init__(self, init_value=0.5):
        super().__init__()
        init_value = float(init_value)
        init_value = min(max(init_value, 1e-6), 1-1e-6)
        # sigmoid(logit)=init_value 되도록 초기화
        logit = math.log(init_value / (1.0 - init_value))
        self.logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def forward(self):
        return torch.sigmoid(self.logit)  # scalar tensor


def imp_regression_constraint_loss_abl(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 256,
    eps: float = 1e-7,
    args=None,
    curriculum=True,
    ablation=None,
    global_step: int = 0,
    total_steps: int = 1,
    # schedule_type: str = "metric",   # ["metric", "cosine", "linear", "learnable"]
    schedule_direction: str = "increase",
    schedule_start: float = 0.0,
    schedule_end: float = 1.0,
    learnable_weight_module=None,    # LearnableReconWeight()
) -> torch.Tensor:
    
    if args.pruning_ratio == 0.3:
        if 'celeba' in args.config:
            total_steps=100000
        elif 'bedroom' in args.config:
            total_steps=200000
    elif args.pruning_ratio == 0.5:
        if 'celeba' in args.config:
            total_steps=200000
        elif 'bedroom' in args.config:
            total_steps=500000
    elif args.pruning_ratio == 0.7:
        if 'celeba' in args.config:
            total_steps=240000
        elif 'bedroom' in args.config:
            total_steps=500000
    elif args.pruning_ratio == 0.9:
        if 'celeba' in args.config:
            total_steps=500000
    else:
        assert f" Total steps is not defined"
    
    B, C, H, W = et_s.shape
    total_pixels = H * W
    importance_map = (et_t - et_s) ** 2

    assert total_pixels % patch_size == 0, \
        f"Total pixels ({total_pixels}) must be divisible by patch_size ({patch_size})"
    num_groups = total_pixels // patch_size

    s_flat = et_s.view(B, C, -1)
    t_flat = et_t.view(B, C, -1)
    importance_flat = importance_map.view(B, C, -1)

    sorted_indices = torch.argsort(importance_flat, dim=-1)
    s_sorted = torch.gather(s_flat, dim=-1, index=sorted_indices)
    t_sorted = torch.gather(t_flat, dim=-1, index=sorted_indices)

    s_grouped = s_sorted.view(B, C, num_groups, patch_size)
    t_grouped = t_sorted.view(B, C, num_groups, patch_size)

    x_mean = s_grouped.mean(dim=-1, keepdim=True)
    y_mean = t_grouped.mean(dim=-1, keepdim=True)
    cov = ((s_grouped - x_mean) * (t_grouped - y_mean)).sum(dim=-1)
    var = ((s_grouped - x_mean) ** 2).sum(dim=-1)

    slope = cov / (var + eps)
    intercept = y_mean.squeeze(-1) - slope * x_mean.squeeze(-1)

    s_target = torch.ones_like(slope)
    i_target = torch.zeros_like(intercept)

    eps_hat = s_grouped * slope.detach().unsqueeze(-1) + intercept.detach().unsqueeze(-1)
    recon_loss = ((eps_hat - t_grouped) ** 2)  # [B,C,G,P]

    if args.use_l2:
        loss_s = ((slope - s_target) ** 2).sum(dim=(1, 2)).mean(dim=0)
        loss_i = ((intercept - i_target) ** 2).sum(dim=(1, 2)).mean(dim=0)
        recon_loss = recon_loss.sum(dim=(1, 2, 3)).mean(dim=0)
    else:
        loss_s = (torch.abs(slope - s_target)).sum(dim=(1, 2)).mean(dim=0)
        loss_i = (torch.abs(intercept - i_target)).sum(dim=(1, 2)).mean(dim=0)
        recon_loss = recon_loss.sum(dim=(1, 2, 3)).mean(dim=0)

    coarse = (loss_s + loss_i) * (patch_size)

    if ablation == "coarse":
        return coarse
    if ablation == "fine":
        return recon_loss

    # ------------------------
    # w_recon 결정 (비교 실험 포인트)
    # ------------------------
    # if schedule_type == "metric":
    if curriculum==True:
        sch_metric = (torch.abs(slope - s_target)).mean() + (torch.abs(intercept - i_target)).mean()
        w_recon = (1.0 - sch_metric.clamp(0.0, 1.0)).detach()  # 기존처럼 item()으로 gradient 끊던 느낌 유지
    elif curriculum==False:
        w_recon = torch.tensor(1.0, device=coarse.device, dtype=coarse.dtype)
        # 기존: (1 - min(1, sch)) * recon_loss
        # -> w_recon = 1 - clamp(sch_metric, 0, 1)
        # w_recon = (1.0 - sch_metric.clamp(0.0, 1.0)).detach()  # 기존처럼 item()으로 gradient 끊던 느낌 유지
    elif curriculum == "cosine":
        w = cosine_schedule(global_step, total_steps, start=schedule_start, end=schedule_end, direction=schedule_direction)
        w_recon = _to_tensor_scalar(w, coarse)
    elif curriculum == "linear":
        w = linear_schedule(global_step, total_steps, start=schedule_start, end=schedule_end, direction=schedule_direction)
        w_recon = _to_tensor_scalar(w, coarse)
    elif curriculum == "learnable":
        if learnable_weight_module is None:
            raise ValueError("learnable_weight_module must be provided for schedule_type='learnable'")
        w_recon = learnable_weight_module()#.to(device=coarse.device, dtype=coarse.dtype)
    else:
        raise ValueError(f"Unknown schedule_type={curriculum}")
    
    if global_step % 100 == 0:
        logging.info(
            f"{args.loss_type} step: {global_step} l, W : {w_recon}"
        )

    return coarse + w_recon * recon_loss
