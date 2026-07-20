import torch
import orm_torch as ORM
import torch
from torch import nn
import orm_torch as ORM

# print(teacher.device)
import logging
import time 



# def hook_modules(model):
#     t_modules = []
#     for module_name, module in model.mid_block.named_children():
#         for submodule_name, submodule in module.named_children():
#             t_modules.append(submodule)

#     for b in model.down_blocks:
#         for module_name, module in b.named_children():
#             for submodule_name, submodule in module.named_children():
#                 t_modules.append(submodule)

#     for b in model.up_blocks:
#         for module_name, module in b.named_children():
#             for submodule_name, submodule in module.named_children():
#                 t_modules.append(submodule)
#     return t_modules

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

def compute_orm(feature_s,feature_t, batch_size):
    loss = 0
    for i in range(len(feature_s)):
        feature_s[i] = feature_s[i].reshape(batch_size, -1)
        feature_t[i] = feature_t[i].reshape(batch_size, -1)
    for i in range(len(feature_s)):
        for j in range(i , len(feature_s)):      
            if i != j:
                gram_t_i= ORM.gram_linear(feature_t[i])
                gram_t_j = ORM.gram_linear(feature_t[j])
                gram_s_j = ORM.gram_linear(feature_s[j])
                st = time.time()
                orms_t = ORM.orm(gram_t_i, gram_t_j)
                orms_s = ORM.orm(gram_t_i, gram_s_j)
                loss += (orms_t - orms_s).square().sum() 
            else:
                gram_t_i= ORM.gram_linear(feature_t[i])
                gram_s_j = ORM.gram_linear(feature_s[j])
                
                orms = ORM.orm(gram_t_i, gram_s_j)
                loss += (1- orms).sum()
    return loss #* batch_size

def distill_(teacher, model, noisy_images, timesteps, t_modules=None, s_modules=None):
    model_output = model(noisy_images, timesteps).sample
    with torch.no_grad():
        t_model_output = teacher(noisy_images, timesteps).sample        
    return model_output, t_model_output


# class fitnet_loss(torch.nn.Module):
#     def __init__(self, device):
#         super().__init__()
#         self.device = device
        
#     def construct_adapter(self,teacher, model,
#                           t_modules, s_modules, 
#                           noisy_images, timesteps):
#         _,_, t_feat, s_feat =  distill(teacher, model, 
#                      t_modules, s_modules, 
#                      noisy_images, timesteps)
#         self.adapters = []
#         for tf, sf in zip(t_feat,s_feat):
#             c_in = sf.shape[1]
#             c_out = tf.shape[1]
#             self.adapters.append(torch.nn.Conv2d(c_in,c_out,1).to(self.device))
#         self.params = []
#         for a in self.adapters:
#             self.params += list(a.parameters())
            
#     def forward(self, s_feat, t_feat):
#         loss = 0
#         for s_f, t_f, a in zip(s_feat, t_feat, self.adapters):
#             print(s_f.shape, t_f.shape)
#             # s_f = s_f.unsqueeze(2)
#             as_f = a(s_f)
#             loss += ((as_f - t_f).pow(2)).square().sum(dim=(1, 2, 3)).mean(dim=0)
#         return loss

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
            loss += ((adapted_s_f - t_f)**2).mean()
            # print(k, ":",((adapted_s_f - t_f).pow(2)).mean())
        return loss.mean()
def patch_spline_regressor_loss(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 16,
    eps: float = 1e-8
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

    # loss_s = (slope - 1.0).sum(2)
    # loss_i = intercept.sum(2)
    s_target = torch.ones_like(slope)
    i_target = torch.zeros_like(intercept)

    loss_s = torch.abs(slope - s_target).mean()#.sum(dim=(1, 2)).mean(dim=0)
    loss_i = torch.abs(intercept - i_target).mean()#.sum(dim=(1, 2)).mean(dim=0)

    return (loss_s + loss_i) #* (patch_size**2)


class distill_loss():
    def __init__(self, loss_type, lamb, device,
                 teacher, model,
                 t_modules, s_modules, 
                 noisy_images, timesteps):
        self.loss_type = loss_type
        self.lamb = lamb
        self.device= device
        if self.loss_type == "fitnet":
            self.loss = FitNetLoss(self.device) 
            self.loss.construct_adapter(teacher, model,
                          t_modules, s_modules, 
                          noisy_images, timesteps)
        elif self.loss_type == 'itgc':
            from ebm_model.vem import ResnetEBM
            # update for dynamic image size
            # define ebm_model
            size = 64
            VEM_kwargs={
                "input_nc": 64,
                "n_blocks": 5
            }
            ebm = ResnetEBM(**VEM_kwargs)
            self.loss = itgc(ebm)
            # self.params = self.loss.params
    def __call__(self, model_output,t_model_output, s_features, t_features,timesteps):
        
        if self.loss_type == "logit":
            return (t_model_output - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0) * (self.lamb)
        elif self.loss_type == 'fitnet':
            return self.loss(s_features, t_features) * 0.7 * (self.lamb)# * 1e-13
        elif self.loss_type == 'okd': # this case needs diff*1 + okd*1
            batch = t_features[0].shape[0]
            loss = ORM.compute_orm_parallel(s_features, t_features, batch)* (self.lamb)
            # loss = compute_orm(s_features,t_features, batch) * (self.lamb)
            return loss
        elif self.loss_type == 'okd_logit':
            batch = t_features[0].shape[0]
            loss = ORM.compute_orm_parallel(s_features, t_features, batch)* (self.lamb)
            # loss = compute_orm(s_features,t_features, batch) * (self.lamb)
            loss += (t_model_output - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0) * 0.7
            return loss
        elif self.loss_type == 'itgc':
            return self.loss(model_output, t_model_output,timesteps) * (self.lamb)
        elif self.loss_type == 'regress':
            return regressor_loss(model_output, t_model_output) * self.lamb
        elif self.loss_type == 'spline':
            return spline_regressor_loss(model_output, t_model_output,n_seg=64) * self.lamb
        else:
            raise Exception("Unexpected loss type, (logit, fitnet, okd, itgc)")
#####
def regressor_loss(et_s, et_t):
    reduce_dims = (2, 3)
    x_mean = et_s.mean(dim=reduce_dims, keepdim=True)
    y_mean = et_t.mean(dim=reduce_dims, keepdim=True)
    cov = ((et_s - x_mean) * (et_t - y_mean)).sum(dim=reduce_dims, keepdim=True)
    var = ((et_s - x_mean) ** 2).sum(dim=reduce_dims, keepdim=True)
    slope     = cov / var
    intercept = y_mean - slope * x_mean
    slope_target = torch.ones_like(slope,device=slope.device)
    intercept_target = torch.zeros_like(intercept,device=intercept.device)
    s_loss = (slope_target - slope).square().sum(dim=(1, 2, 3)).sum(dim=0)
    i_loss = (intercept_target - intercept).square().sum(dim=(1, 2, 3)).sum(dim=0)
    return s_loss + i_loss

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

    loss_s = torch.abs(slope - s_target).sum()
    loss_i = torch.abs(intercept - i_target).sum()

    return loss_s + loss_i
#####

class itgc(torch.nn.Module):
    def __init__(self, ebm,
                 kd_VID_langevin_steps=10,
                 kd_VID_step_size=50.0,
                 kd_VID_sigma_t=0.005,
                 kd_VID_MI_lambda=0.05,
                 kd_VID_l2_lambda=0.05):
        super().__init__()
        self.VEM_opt_kwargs= {
            # "class_name": "torch.optim.Adam",
            "lr": 0.0001,
            "betas": [
                0.0,
                0.999
            ],
            "eps": 1e-08
        }
        self.kd_VID_langevin_steps = kd_VID_langevin_steps
        self.kd_VID_step_size = kd_VID_step_size
        self.kd_VID_sigma_t = kd_VID_sigma_t
        self.kd_VID_MI_lambda = kd_VID_MI_lambda
        self.kd_VID_l2_lambda = kd_VID_l2_lambda
        # add ebm with diffusioknm timestep?
        self.ebm = ebm
        self.ebm_opt = torch.optim.Adam(self.ebm.parameters(),
                                        **self.VEM_opt_kwargs)
        
    def Langevin_dynamics(self, ebm, teacher_image, 
                          student_image, sigma_t, 
                          langevin_steps, step_size):
        fake_teacher_img_MCMC = torch.autograd.Variable(student_image.data, requires_grad=True)
        for k in range(langevin_steps):
            E_prime = torch.autograd.grad(ebm(fake_teacher_img_MCMC, student_image).sum(),
                                        [fake_teacher_img_MCMC], retain_graph=True)[0]
            fake_teacher_img_MCMC.data = fake_teacher_img_MCMC.data - step_size * E_prime \
                + sigma_t * torch.randn_like(fake_teacher_img_MCMC)

        fake_teacher_img_MCMC = fake_teacher_img_MCMC.detach()
        return fake_teacher_img_MCMC
    
    def forward(self,gen_img, gen_teacher_img,timesteps):
        fake_teacher_img_MCMC = self.Langevin_dynamics(self.ebm, gen_teacher_img, gen_img, self.kd_VID_sigma_t, self.kd_VID_langevin_steps, self.kd_VID_step_size)
        plus_output = self.ebm(gen_teacher_img, gen_img)
        minus_output = self.ebm(fake_teacher_img_MCMC, gen_img)
        kd_MI_loss = plus_output - minus_output
        return kd_MI_loss.mean() 
    
    def update_ebm(self,gen_teacher_img, gen_img,timesteps):
        self.ebm_opt.zero_grad(set_to_none=True)
        self.ebm.requires_grad_(True)
        fake_teacher_img_MCMC = self.Langevin_dynamics(self.ebm, gen_teacher_img, gen_img, self.kd_VID_sigma_t, self.kd_VID_langevin_steps, self.kd_VID_step_size)
        plus_output = self.ebm(gen_teacher_img, gen_img)
        minus_output = self.ebm(fake_teacher_img_MCMC, gen_img)
        loss_VEM = plus_output.mean() - minus_output.mean() + self.kd_VID_l2_lambda * (plus_output ** 2 + minus_output ** 2).mean()
        return loss_VEM
        # loss_VEM.backward()
        # self.ebm_opt.step()
        # return kd_MI_loss
        
    



def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def noise_estimation_kd_loss(model,
                             teacher,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, 
                          kd_loss: None,
                          keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    batch = x.shape[0]
    output, teacher_output, s_features, t_features = hook_forward(model,teacher, x, t)
    # orm_loss = compute_orm(s_features,t_features, batch)
    # print(teacher.device)
    orm_loss = kd_loss(output,teacher_output, s_features, t_features,t)
    # KD_LOSS += orm_loss
    # if keepdim:
    #     return 0.7*(teacher_output - output).square().sum(dim=(1, 2, 3)) + 0.3 * (e - output).square().sum(dim=(1, 2, 3))
    # else:
    distance = (teacher_output - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) , orm_loss, distance
    # return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) , orm_loss

    # return 0.7*(teacher_output - output).square().sum(dim=(1, 2, 3)).mean(dim=0) , 0.3 * (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) , orm_loss*batch
def patch_spline_regressor_loss(
    et_s: torch.Tensor,
    et_t: torch.Tensor,
    patch_size: int = 16,
    eps: float = 1e-8,
    use_l2 = False
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

    # loss_s = (slope - 1.0).sum(2)
    # loss_i = intercept.sum(2)
    s_target = torch.ones_like(slope)
    i_target = torch.zeros_like(intercept)

    # loss_s = ((slope - s_target)).sum(dim=(1, 2)).mean(dim=0)
    # loss_i = ((intercept - i_target)).sum(dim=(1, 2)).mean(dim=0)

    if use_l2:
        loss_s = (torch.abs((slope - s_target)**2)).mean(dim=(1, 2)).mean(dim=0)
        loss_i = (torch.abs((intercept - i_target)**2)).mean(dim=(1, 2)).mean(dim=0)
    else:
        loss_s = (torch.abs(slope - s_target)).mean(dim=(1, 2)).mean(dim=0)
        loss_i = (torch.abs(intercept - i_target)).mean(dim=(1, 2)).mean(dim=0)

    return (loss_s + loss_i)

loss_registry = {
    'simple': noise_estimation_loss,
}


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

def hook_forward(student,teacher, x, t):
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
    
    output = student(x, t.float())
    
    
    with torch.no_grad():
        teacher_output = teacher(x, t.float())
    for hook in hooks:
        hook.remove()

    return output, teacher_output, s_features, t_features



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
