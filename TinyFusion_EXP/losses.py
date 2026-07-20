import torch
from torch import nn

# print(teacher.device)
import logging
import time 
import matplotlib.pyplot as plt
import math, json, os


    
def sigmoid_decay_schedule(step: int,
                           min_lambda: float = 0.1,
                           max_lambda: float = 1.0,
                           sharpness: float = 12.0,
                           args=0) -> float:

    total_steps=500000
        
    midpoint = total_steps / 2
    k  = sharpness / total_steps                # 기울기 조절
    sig = 1 / (1 + math.exp(-k * (step - midpoint)))   # 0→1
    return max_lambda - (max_lambda - min_lambda) * sig # 1→0.1


def spline_regressor_loss(et_s: torch.Tensor,
                                   et_t: torch.Tensor,
                                   n_seg: int = 4,
                                   eps: float = 1e-8) -> torch.Tensor:

    B, C, H, W = et_s.shape
    N = H * W
    s_flat = et_s.view(B, C, N)
    t_flat = et_t.view(B, C, N)

    pad = (-N) % n_seg
    if pad:
        zeros_s = torch.zeros(B, C, pad, device=et_s.device, dtype=et_s.dtype)
        zeros_t = torch.zeros(B, C, pad, device=et_t.device, dtype=et_t.dtype)
        s_flat = torch.cat([s_flat, zeros_s], dim=2)
        t_flat = torch.cat([t_flat, zeros_t], dim=2)
        N = N + pad

    N_seg = N // n_seg
    s_seg = s_flat.view(B, C, n_seg, N_seg)
    t_seg = t_flat.view(B, C, n_seg, N_seg)

    x_mean = s_seg.mean(dim=-1, keepdim=True)        # [B, C, n_seg, 1]
    y_mean = t_seg.mean(dim=-1, keepdim=True)        # [B, C, n_seg, 1]
    cov    = ((s_seg - x_mean) * (t_seg - y_mean)).sum(dim=-1, keepdim=True)
    var    = ((s_seg - x_mean) ** 2).sum(dim=-1, keepdim=True)

    slope     = cov / (var + eps)                    
    intercept = y_mean - slope * x_mean              

    slope     = slope.squeeze(-1)
    intercept = intercept.squeeze(-1)

    s_target = torch.ones_like(slope)
    i_target = torch.zeros_like(intercept)

    # loss_s = torch.abs(slope - s_target).sum()
    # loss_i = torch.abs(intercept - i_target).sum()
    # loss_s = (slope - s_target)**2#.sum()
    # loss_i = (intercept - i_target)**2#.sum()
    loss_s = torch.abs(slope - s_target).sum()
    loss_i = torch.abs(intercept - i_target).sum()
    return loss_s + loss_i


def patch_spline_regressor_loss(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 16,
    eps: float = 1e-8,
    return_beta = False
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

    s_target = torch.ones_like(slope)
    i_target = torch.zeros_like(intercept)
    loss_s = torch.abs(slope - 1.0)#.pow(2)
    loss_i = torch.abs(intercept)#.pow(2)
    # loss_s = (slope - 1.0).pow(2)
    # loss_i = (intercept).pow(2)
    # loss_s = torch.nn.MSELoss()(slope,s_target)
    # loss_i = torch.nn.MSELoss()(intercept,i_target)
    if return_beta:
        return (loss_s + loss_i) ,slope , intercept
    else:
        return (loss_s + loss_i) #* (patch_size**2)

#####
def regression_constraint_loss(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 16,
    eps: float = 1e-4,
    use_l2 = True,
    return_beta = False
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
    # print(sch)
    if use_l2:
        loss_s = (((slope - s_target)**2))#.mean(dim=(1, 2)).mean(dim=0)
        loss_i = (((intercept - i_target)**2))#.mean(dim=(1, 2)).mean(dim=0)
        
        # loss_s = (torch.sqrt((slope - s_target)**2))#.mean(dim=(1, 2)).mean(dim=0)
        # loss_i = (torch.sqrt((intercept - i_target)**2))#.mean(dim=(1, 2)).mean(dim=0)
    else:
        loss_s = (torch.abs(slope - s_target))#.mean(dim=(1, 2)).mean(dim=0)
        loss_i = (torch.abs(intercept - i_target))#.mean(dim=(1, 2)).mean(dim=0)

    recon_loss = ((eps_hat - t_flat)**2)#.mean(dim=(1, 2,3)).mean(dim=0)
    # return (loss_s+ loss_i)*(patch_size**2), (1-min(1, sch))*recon_loss 
    if return_beta:
        return (loss_s+ loss_i), (1-min(1, sch))*recon_loss,slope , intercept
    else:
        return (loss_s+ loss_i), (1-min(1, sch))*recon_loss#* (patch_size**2)

import torch

def imp_regression_constraint_loss(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 256, ## patch_size=16 -> 16*16=256
    eps: float = 1e-4,
    use_l2: bool = True,
    return_beta: bool = False
) -> tuple:
    
    B, C, H, W = et_s.shape
    total_pixels = H * W
    importance_map = (et_t- et_s)**2
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
    recon_loss = ((eps_hat - t_grouped) ** 2) # Shape: [B, C, num_groups, patch_size]

    sch = (torch.abs(slope - s_target)).mean().item() + (torch.abs(intercept - i_target)).mean().item()
    
    if use_l2:
        loss_s = (slope - s_target) ** 2
        loss_i = (intercept - i_target) ** 2
    else:
        loss_s = torch.abs(slope - s_target)
        loss_i = torch.abs(intercept - i_target)

    if return_beta:
        return (loss_s + loss_i), (1 - min(1, sch)) * recon_loss, slope, intercept
    else:
        return (loss_s + loss_i), (1 - min(1, sch)) * recon_loss



import torch.nn.functional as F
@torch.no_grad()
def _ones_kernel(C, p, device, dtype):
    # depthwise conv용 ones 커널: [C, 1, p, p]
    k = torch.ones((C, 1, p, p), device=device, dtype=dtype)
    return k

def regression_constraint_loss_conv(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 16,
    stride: int = None,       # None이면 grid (=patch_size)
    eps: float = 1e-4,
    use_l2 = True,
    return_beta = False
):
    """
    unfold 없이 conv2d만으로 패치 통계(평균/분산/공분산) 계산.
    - grid:  stride=patch_size
    - sliding: stride < patch_size
    제약: 패딩 없음(가장자리 제외). (H-p)%stride==0, (W-p)%stride==0 권장.
    """
    assert et_s.shape == et_t.shape
    B, C, H, W = et_s.shape
    p = patch_size
    if stride is None:
        stride = p

    # 간단 체크 (패딩 없이 딱 맞추는 설정 추천)
    assert H >= p and W >= p, "image smaller than patch"
    assert (H - p) % stride == 0 and (W - p) % stride == 0, \
        f"(H-p) and (W-p) must be divisible by stride. Got H={H}, W={W}, p={p}, stride={stride}"

    # depthwise conv로 창 합계 계산
    k = _ones_kernel(C, p, et_s.device, et_s.dtype)
    P = p * p

    sum_x  = F.conv2d(et_s, k, stride=stride, groups=C)             # [B,C,Ho,Wo]
    sum_y  = F.conv2d(et_t, k, stride=stride, groups=C)
    sum_x2 = F.conv2d(et_s * et_s, k, stride=stride, groups=C)
    sum_xy = F.conv2d(et_s * et_t, k, stride=stride, groups=C)
    sum_y2 = F.conv2d(et_t * et_t, k, stride=stride, groups=C)

    Ex  = sum_x  / P
    Ey  = sum_y  / P
    Ex2 = sum_x2 / P
    Exy = sum_xy / P
    Ey2 = sum_y2 / P

    var = (Ex2 - Ex * Ex).clamp_min(0.0)                            # [B,C,Ho,Wo]
    cov = (Exy - Ex * Ey)                                           # [B,C,Ho,Wo]

    slope     = cov / (var + eps)                                   # a
    intercept = Ey - slope * Ex                                     # b

    # 회귀 제약 손실 (a→1, b→0)
    if use_l2:
        reg_s = (slope - 1.0) ** 2
        reg_i = (intercept) ** 2
    else:
        reg_s = (slope - 1.0).abs()
        reg_i = (intercept).abs()
    reg_map = reg_s + reg_i                                    # [B,C,Ho,Wo]

    a = slope.detach()
    b = intercept.detach()
    rec_map = (a*a)*Ex2 - 2*a*Exy + Ey2 + 2*a*b*Ex - 2*b*Ey + (b*b) # [B,C,Ho,Wo]
    

    sch = (slope - 1.0).abs().mean().item() + (intercept).abs().mean().item()
    beta = (1-min(1, sch))


    if return_beta:
        return reg_map, beta * rec_map, slope , intercept
    else:
        return reg_map, beta * rec_map
