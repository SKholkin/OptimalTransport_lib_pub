
# from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import wandb
from src.dists_utils import Gaussian
from src.samplers.samplers import Sampler

class OptimalTransport(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass

class OTSampler(Sampler):
    def __init__(self, ot_model: OptimalTransport, input_sampler: Sampler, state_dict_path=None) -> None:
        self.ot_model = ot_model
        self.dim = input_sampler.dim
        self.ot_model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        self.x_sampler = input_sampler.x_sample

    def x_sample(self, batch_size):
        x_samples = self.x_sampler(batch_size)
        return x_samples
    
    def y_sample(self, batch_size):
        x_samples = self.x_sampler(batch_size)
        return self.ot_model(x_samples)
    
    def cond_y_sample(self, x):
        return self.__call__(x)
    
    def __call__(self, x):
        return self.ot_model(x)

class NeuralOptimalTransport(OptimalTransport):
    def __init__(self, T_net, f_net, cost_fn, lr=1e-4) -> None:
        super(NeuralOptimalTransport, self).__init__()
        self.T_net = T_net
        self.f_net = f_net
        self.opt_T = torch.optim.Adam(T_net.parameters(), lr=lr)
        self.opt_f = torch.optim.Adam(f_net.parameters(), lr=lr)
        self.cost_fn = cost_fn
    
    def forward(self, x):
        return self.T_net(x)

    def train_step(self, sampler_x, sampler_y, T_opt_steps=10, batch_size=512, no_opt_potential=False):

        model_device = next(iter(self.T_net.parameters())).device

        samples_x = sampler_x(batch_size).to(model_device)
        with torch.no_grad():
            transport_x = self.T_net(samples_x)
        loss_f = self.f_net(transport_x).mean() - self.f_net(sampler_y(batch_size).to(model_device)).mean()

        self.opt_f.zero_grad()
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(self.f_net.parameters(), 1)
        self.opt_f.step()

        for i in range(T_opt_steps):
            samples_x = sampler_x(batch_size).to(model_device)
            transport_x = self.T_net(samples_x)
            loss_T = (self.cost_fn(samples_x, transport_x) - self.f_net(transport_x).to(model_device)).mean()

            self.opt_T.zero_grad()
            loss_T.backward()
            torch.nn.utils.clip_grad_norm_(self.T_net.parameters(), 1)
            self.opt_T.step()

        if wandb.run:
            wandb.log({'loss_f': loss_f.item(), 'loss_T': loss_T.item()})
        # else:
        #     print(loss_f.item(), loss_T.item())


class GaussianEntropicNeuralOptimalTransport(OptimalTransport):
    def __init__(self, T_net, f_net, eps=1, lr=1e-4, end_normal=True) -> None:
        super(GaussianEntropicNeuralOptimalTransport, self).__init__()
        self.T_net = T_net
        self.f_net = f_net
        self.opt_T = torch.optim.Adam(T_net.parameters(), lr=lr)
        self.opt_f = torch.optim.Adam(f_net.parameters(), lr=lr)
        self.end_normal = end_normal
        self.eps = eps

    @staticmethod
    def quadratic_cost_fn(x, y):
        return F.mse_loss(x, y)

    def forward(self, x):
        mu, log_sigma = self.T_net(x)
        return mu + torch.exp(log_sigma) * torch.randn_like(mu)
    
    @torch.no_grad()
    def sample(self, x):
        return self.forward(x)
    
    def train_step(self, sampler_x, sampler_y, T_opt_steps=10, batch_size=512, no_opt_potential=False):

        model_device = next(iter(self.T_net.parameters())).device
        samples_x = sampler_x(batch_size).to(model_device)

        with torch.no_grad():
            mu, log_std = self.T_net(samples_x)
            transport_x = mu + torch.exp(log_std) * torch.randn_like(mu)

        loss_f = self.f_net(transport_x).mean() - self.f_net(sampler_y(batch_size).to(model_device)).mean()

        if not no_opt_potential:

            self.opt_f.zero_grad()
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(self.f_net.parameters(), 1)

            self.opt_f.step()
            
        for i in range(T_opt_steps):
            samples_x = sampler_x(batch_size).to(model_device)
            mu, log_std = self.T_net(samples_x)
            # print(log_std.mean())
            transport_x = mu + torch.exp(log_std) * torch.randn_like(mu)

            # prior_samples = samples_x + torch.randn_like(mu) * self.eps

            # kl_div = -(Gaussian.empirical_entropy(mu, log_std, transport_x) - Gaussian.empirical_entropy(mu, log_std, prior_samples))
            
            if not self.end_normal:
                kl_div = -Gaussian.entropy(mu, log_std).mean()
            else:
                kl_div = Gaussian.kl_div(mu, log_std, torch.zeros_like(mu, device=model_device), torch.zeros_like(log_std, device=model_device)).mean()

            cost = self.quadratic_cost_fn(transport_x, samples_x)
            
            transport_x_potential = self.f_net(transport_x).mean()

            # loss_T = Gaussian.kl_div(mu, log_std, samples_x, torch.log(torch.tensor(self.eps, device=model_device))).mean() + transport_x_potential

            loss_T = kl_div + (1 / self.eps) * cost - transport_x_potential
            # print()

            self.opt_T.zero_grad()
            loss_T.backward()
            torch.nn.utils.clip_grad_norm_(self.T_net.parameters(), 1)

            self.opt_T.step()

        if wandb.run:
            wandb.log({'loss_f': loss_f.item(), 'loss_T': loss_T.item(), 'kl_div': kl_div.item(), 'L2_cost': cost, 'f_potential_transport': transport_x_potential.item()})
        # else:
        #     print(f'Loss f: {loss_f.item()} Loss T: {loss_T.item()}')

def langevin_sample(x_0, score_fn, eta=0.05, n_steps=10):
    for i in range(n_steps):
        noise = torch.randn_like(x_0)
        x_0 = x_0 + eta / 2 * score_fn(x_0).detach() + math.sqrt(eta) * noise
        
    return x_0

class EgNOT(OptimalTransport):
    def __init__(self, f_net, eps, lr=1e-4) -> None:
        super().__init__()
        self.f_net = f_net
        # self.eps = eps
        self.opt = torch.optim.Adam(self.f_net.parameters(), lr=lr)
        self.eps = eps
    
    @staticmethod
    def quadratic_cost_fn(x, y):
        return F.mse_loss(x, y)

    def forward(self, x, proposal_z=None, ula_steps=10):
        return self.sample(x, proposal_z, ula_steps)
    
    @torch.no_grad()
    def sample(self, x, proposal_z=None, ula_steps=10, eta=0.05):
        score_fn = self.get_score_fn(x)
        if proposal_z is None:    
            proposal_z = torch.randn_like(x)
        
        return langevin_sample(proposal_z, score_fn, eta=eta, n_steps=ula_steps)
               
    def get_score_fn(self, x):
        def ret_val_fn(y):
            with torch.enable_grad():
                y = y.clone().detach()
                y.requires_grad = True
                log_p = self.f_net(y) - self.quadratic_cost_fn(x, y)
                grad_log_p = torch.autograd.grad(
                    outputs=log_p, 
                    inputs=y,
                    grad_outputs=torch.ones_like(log_p),
                    create_graph=True,
                    retain_graph=True,
                )[0] # (bs, *shape)
            return 1 / self.eps * grad_log_p
               
        return ret_val_fn

    def cond_sample_from_potential(self, x, z, ula_steps=10, eta=0.005):
        score_fn = self.get_score_fn(x)
        return langevin_sample(z, score_fn, eta=eta, n_steps=ula_steps)

    def train_step(self, sampler_x, sampler_y, cond_sampler_y_proposal=lambda x: torch.randn_like(x), batch_size=512, ula_steps=10, eta=0.005):
        model_device = next(iter(self.f_net.parameters())).device

        x_samples = sampler_x(batch_size).to(model_device)
        y_samples = sampler_y(batch_size).to(model_device)
        langevin_proposal = cond_sampler_y_proposal(x_samples).to(model_device)
        f_samples = self.cond_sample_from_potential(x_samples, langevin_proposal, ula_steps, eta)

        loss = -(self.f_net(y_samples).mean() - self.f_net(f_samples).mean())
        
        self.opt.zero_grad()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.f_net.parameters(), 1)

        self.opt.step()
        # print(loss.item())
        if wandb.run:
            wandb.log({'loss_f_egnot': loss.item()})

        return loss

class EgNOTWithEntNOT(OptimalTransport):
    def __init__(self, egNOT, entNOT) -> None:
        super().__init__()
        self.eps = egNOT.eps
        self.entNOT = entNOT
        self.entNOT.f_net = egNOT.f_net
        self.egNOT = egNOT
        # self.eps = eps

    def forward(self, x, proposal_z=None, ula_steps=10):
        return self.sample(x, proposal_z, ula_steps)

    def cond_sample_from_potential(self, x, z, ula_steps=10, eta=0.005):
        return self.egNOT.cond_sample_from_potential(x, z, ula_steps, eta)
    
    @torch.no_grad()
    def sample(self, x, ula_steps=10, eta=0.05):
        proposal_z = self.entNOT(x)
        return self.egNOT.sample(x, proposal_z, ula_steps=ula_steps, eta=eta)
    
    def train_step(self, sampler_x, sampler_y, batch_size=512, ula_steps=10, eta=0.005):
        self.egNOT.train_step(sampler_x, sampler_y, cond_sampler_y_proposal=lambda x: self.entNOT(x), batch_size=batch_size, ula_steps=ula_steps, eta=eta)
        self.entNOT.train_step(sampler_x, sampler_y, T_opt_steps=10, batch_size=batch_size, no_opt_potential=True)
