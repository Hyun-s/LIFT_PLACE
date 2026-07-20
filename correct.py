import torch
from models.diffusion import get_timestep_embedding, nonlinearity

class corrector(torch.nn.Module):
    def __init__(self, t_dim, in_channels, hidden=128):
        super().__init__()
        self.t_dim = t_dim
        self.in_channels = in_channels
        self.input_dim = t_dim + in_channels*2
        self.a_t_est = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden), torch.nn.SiLU(),
            # torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(int(hidden), self.in_channels*1) 
        )
        self.b_t_est = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden), torch.nn.SiLU(),
            # torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(int(hidden), self.in_channels*1) 
        )
        # self.a_t_est = torch.nn.Linear(int(hidden/2), self.in_channels*1)   
        # self.b_t_est = torch.nn.Linear(int(hidden/2), self.in_channels*1)  
        self._reset_parameters() 
    def _reset_parameters(self):
        """초기 slope=1, intercept=0 이 되도록 가중치·bias를 세팅."""
        # (1) 첫 Linear: 작은 값 → SiLU(≈0) 로 만들기
        torch.nn.init.normal_(self.b_t_est[0].weight, mean=0.0, std=1e-2)
        torch.nn.init.zeros_(self.b_t_est[0].bias)

        # (2) 두 번째 Linear: weight ≈ 0, bias = [1,1,...,0,0,...]
        torch.nn.init.normal_(self.b_t_est[2].weight, mean=0.0, std=1e-3)   # 학습은 되게 아주 작게
        torch.nn.init.zeros_(self.b_t_est[2].bias)

                        
    def forward(self, temb,mean,sigma):
        x = torch.cat([temb,mean,sigma], dim=1)
        slope = self.a_t_est(x).unsqueeze(-1).unsqueeze(-1)
        intercept = self.b_t_est(x).unsqueeze(-1).unsqueeze(-1)

        return slope,intercept
    
    
# class group_conv(torch.nn.Conv2d):
#     def __init__(self, res, channels, chunk=1):
#         super().__init__(channels,channels*2)
#         # conv = torch.nn.Conv2d(channels,channels, (1,1),bias=True,groups=channels)
            

    
    
    
class corrected_diffusion(torch.nn.Module):
    def __init__(self, denoiser, device='cuda'):
        super().__init__()
        self.denoiser = denoiser.to(device)
        temb = get_timestep_embedding(torch.ones([1]), self.denoiser.ch).to(device)
        # temb = self.denoiser.temb.dense[0](temb)
        # temb = nonlinearity(temb)
        # temb = self.denoiser.temb.dense[1](temb)
        t_dim = temb.shape[1]
        # self.corrector = corrector(t_dim, self.denoiser.in_channels).to(device)
        # self.bn = torch.nn.BatchNorm2d(3).to(device)
        self.corrector = torch.nn.Conv2d(3,3,1, groups=3).to(device)
        # w = torch.ones(3,1,1,1).to(device)
        # w = torch.nn.Parameter(w)
        # b = torch.zeros(3).to(device)
        # b = torch.nn.Parameter(b)
        # self.bn.weight = w
        # self.bn.bias = b
        
    def get_temb(self, t):
        temb = get_timestep_embedding(t,self.denoiser.ch)
        # temb = self.denoiser.temb.dense[0](temb)
        # temb = nonlinearity(temb)
        # temb = self.denoiser.temb.dense[1](temb)
        return temb
    def forward(self,x,t,return_ab=False):
        with torch.no_grad():
            eps = self.denoiser(x,t).detach()
        # temb = self.get_temb(t)
        # mu_eps = eps.mean(dim=(2,3))
        # var_eps = eps.var(dim=(2,3))
        # a_t, b_t = self.corrector(temb, mu_eps, var_eps)
        # if return_ab:
        #     return (eps *a_t + b_t), eps, a_t, b_t
        # else:
        #     return (eps *a_t + b_t)
        if return_ab:
            return self.corrector(eps,t), eps#, a_t, b_t
        else:
            return self.corrector(eps,t)
        