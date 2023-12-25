from src.models.bridge import Bridge
from src.sde import SDE

import torch

import torch.nn as nn


class FlowMatching(Bridge):
    def __init__(self, unet):
        super().__init__()
        # CNN + positional embeding time conditioning
        # can use NN from MNIST DDPM
        self.vector_net = unet
        self.euler_dt = 0.01
        
    def __call__(self, x_0):
        # solve forward ODE via Euler or torchdiffeq solver
        x_t = x_0
        
        t_range = torch.arange(0, 1, step=self.euler_dt, device=x_0.device)
        
        for t in t_range:
            x_t = x_t + self.vector_net(x_t, t) * self.euler_dt
            
        return x_t
    
    def parameters(self):
        return self.vector_net.parameters()
    
    def get_drift(self):
        return self.vector_net
    
    def sample_path(self, x_0):

        x_t = x_0
        
        t_range = torch.arange(0, 1, step=self.euler_dt).unsqueeze(0).repeat(x_0.shape[0])

        x_t_storage = []
        
        for t in t_range:
            x_t = x_t + self.vector_net(x_t, t) * self.euler_dt
            x_t_storage.append(x_t.detach())
            
        return x_t, t

    @torch.no_grad()
    def sample(self, x_0, steps=100):
        
        drift_fn = lambda x, t: self.get_drift()(x, t)

        sigma_fn = lambda t: 0
        
        model_x_sample_fn = lambda x: SDE(drift_fn, sigma_fn, is_forward=True, skip_n_last=0).sample(
            x, num_steps=steps, device=next(self.vector_net.parameters()).device)[0][:, -1]
        
        return model_x_sample_fn(x_0)
    
    def step(self, x_0, x_1):
        batch_size = x_0.shape[0]
        t = torch.distributions.uniform.Uniform(0, 1).sample([batch_size]).to(x_0.device).unsqueeze(-1)

        x_t = t * x_1 + (1 - t) * x_0
        x_t_hat = self.vector_net(x_t, t)
        return self.loss(x_t_hat, x_0, x_1, t).mean()
    
    def loss(self, x_t_hat, x_0, x_1, t):
        return torch.norm((x_t_hat - (x_1 - x_0)).reshape([x_0.shape[0], -1]), dim=-1)
    