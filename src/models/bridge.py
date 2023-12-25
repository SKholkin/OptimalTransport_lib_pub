import torch
import torch.nn as nn
import numpy as np

from src.models.light_sb import LightSB
from src.samplers.samplers import Sampler, GTSDEByDrift
from src.sde import SDE
from src.dists_utils import Gaussian
import wandb
from copy import deepcopy


class XAndTimeAndYDataset(torch.utils.data.Dataset):
    def __init__(self, x, time, y):
        self.x = x
        self.t = time
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.y[idx]

from eot_benchmark.metrics import compute_BW_UVP_by_gt_samples

class Bridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.T = 100

    def __call__(self, x):
        pass

    def get_drift(self):
        return lambda x, t: 0
    
    def sample_t(self, x, t):
        pass
    
    def parameters(self):
        return None
    
    def sample_path(self, x):
        delta_t = 1. / self.T
        t = torch.arange(0, 100) * delta_t
        return x, t

def mle_fit_drift(drift_fn, x_t, t, opt, batch_size, n_epochs, device, delta_t=1./100):
    
    train_y = (x_t[:, 1:] - x_t[:, :-1]) / delta_t

    train_x = x_t[:, :-1]
    train_t = t[:, :-1].unsqueeze(2)
    
    train_x = train_x.reshape([train_x.shape[0] * train_x.shape[1], *train_x.shape[2:]])
    train_y = train_y.reshape([train_y.shape[0] * train_y.shape[1], *train_y.shape[2:]])
    train_t = train_t.reshape([-1, 1])
    
    idx = np.random.choice(train_x.shape[0], size=int(train_x.shape[0] / (batch_size / 512)), replace=False)
    
    train_x = train_x[idx]
    train_y = train_y[idx]
    train_t = train_t[idx]
    
    dataset = XAndTimeAndYDataset(train_x, train_t, train_y)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)
    
    for ep in range(n_epochs):
        loss_storage = []
        for n_iter, (x, t, y) in enumerate(iter(dataloader)):
            
            x, t, y = x.to(device), t.to(device), y.to(device)
                    
            y_pred = drift_fn(x, t.squeeze())
            
            loss = (torch.norm((y - y_pred).reshape([y.shape[0], -1]), dim=1) ** 2).mean()
            opt.zero_grad()
            
            loss.backward()
            
            opt.step()
            loss_storage.append(loss.item())

            # if wandb.run:
            #     wandb.log({'loss': loss.item()})


class BackwardMLEVargasBridge(Bridge):
    
    def __init__(self, forward_bridge, backward_bridge, sigma, device='cpu'):
        # will stay static
        self.fw_bridge = forward_bridge
        # will be trained
        self.bw_bridge = backward_bridge
        self.batch_size = 512
        self.device = device
        self.sigma = sigma
        self.opt = torch.optim.Adam(self.bw_bridge.parameters(), lr=1e-3)
        
    def train_epoch(self, x_0_batch, x_1_batch):
        # perform one IPFP backward fitting iteration
        
        x_t, t = self.fw_bridge.sample_path(x_0_batch)
        
        x_t = x_t.flip(1).detach().to(self.device)
        n_epochs = 1
        
        mle_fit_drift(self.bw_bridge.get_drift(), x_t, t, self.opt, self.batch_size, n_epochs, self.device)

    def __call__(self, x):
        
        drift_fn = self.bw_bridge.get_drift()

        sigma_fn = lambda t: torch.ones_like(t) * self.sigma

        model_x_sample_fn = lambda x: SDE(drift_fn, sigma_fn, is_forward=True).sample(x)[0][:, -1]

        return model_x_sample_fn(x)

    def sample_t(self, x, t):
        # brownian bridge?
        raise NotImplementedError()
    
    def parameters(self):
        return self.bw_bridge.parameters()
        
    def sample_path(self, x):
        
        drift_fn = self.bw_bridge.get_drift()

        sigma_fn = lambda t: torch.ones_like(t) * self.sigma

        x, t = SDE(drift_fn, sigma_fn, is_forward=True).sample(x)
        # print('sample path x, t', x.shape, t.shape)

        return x, t

from scipy.linalg import sqrtm

def transport(x, mu_1, cov_1, mu_2, cov_2, eps):
    cov_1_sqrt = torch.Tensor(sqrtm(cov_1))
    cov_1_inv = torch.linalg.inv(cov_1)
    cov_1_inv_sqrt = torch.Tensor(sqrtm(cov_1_inv))
    
    d_sigma = torch.tensor(sqrtm((4 * cov_1_sqrt @ cov_2 @ cov_1_sqrt + eps ** 4 * torch.eye(cov_2.shape[0]))))
    
    c_sigma = 1 / 2 * (cov_1_sqrt @ d_sigma @ cov_1_inv_sqrt - eps ** 2 * torch.eye(cov_2.shape[0]))
    
    mu = mu_2  + ((c_sigma.T @ cov_1_inv).unsqueeze(0).repeat([x.shape[0], 1, 1]) @ (x - mu_1).unsqueeze(-1)).squeeze()
    cov = cov_2 - c_sigma.T @ cov_1_inv @ c_sigma
    
    cond_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=cov)
    transported_samples_y = cond_dist.sample([1]).squeeze()

    return transported_samples_y
    
def eot_gaussian(x_samples, y_samples, eps):
    # estimate mean and cov
    
    mu_1 = torch.mean(x_samples, dim=0)
    mu_2 = torch.mean(y_samples, dim=0)
    
    cov_1 = ((x_samples - mu_1).unsqueeze(-1) @ (x_samples - mu_1).unsqueeze(-2)).mean(0)
    cov_2 = ((y_samples - mu_2).unsqueeze(-1) @ (y_samples - mu_2).unsqueeze(-2)).mean(0)

    return transport(x_samples, mu_1, cov_1, mu_2, cov_2, eps)

class BrownianBridge:
    def __init__(self, eps, x_0, x_1, t_0=0., t_1=1.) -> None:
        self.x_0 = x_0
        self.x_1 = x_1
        self.device = x_0.device
        # print(len(t_0), x_0.shape, len(0))

        if isinstance(t_0, float):
            self.t_0 = t_0 * torch.ones([x_0.shape[0], 1]).to(self.device)
            self.t_1 = t_1 * torch.ones([x_0.shape[0], 1]).to(self.device)
        else:
            self.t_0 = t_0
            self.t_1 = t_1

        self.eps = eps

    def get_mean_std(self, t):

        mean = self.x_0 + ((t - self.t_0) / (self.t_1 - self.t_0)) * (self.x_1 - self.x_0)

        std = torch.sqrt(self.eps * (t - self.t_0) * (self.t_1 - t) / (self.t_1 - self.t_0))

        return mean, std

    def sample(self, t):

        mean, std  = self.get_mean_std(t)
                    
        x_t = mean + std * torch.randn_like(self.x_1)

        return x_t

    def score(self, x_t, t):

        mean, std  = self.get_mean_std(t)

        return Gaussian.score(mean, std, x_t)
        

class LightSBBridge(Bridge):
    def __init__(self, dim, n_potentials, eps, is_diagonal=True, eot_finish=False, device='cpu', S_diagonal_init=0.1) -> None:
        super().__init__()
        self.lsb_model = LightSB(dim=dim, n_potentials=n_potentials, epsilon=eps, is_diagonal=is_diagonal,
                 sampling_batch_size=1, S_diagonal_init=S_diagonal_init, r_scale=1)
        self.eps = eps
        self.eot_finish = eot_finish
        self.device = device
        self.to(device)

    def init_by_samples(self, samples):
        self.lsb_model.init_r_by_samples(samples)

    def __call__(self, x):
        # add Gaussan entropic OT at the end
        # return self.lsb_model(x)

        if self.eot_finish:
            print('EOT')
            return eot_gaussian(self.lsb_model(x), self.saved_y_samples, self.eps)
        return self.lsb_model(x)

    def sample_t(self, x, t):
        self.lsb_model.sample_at_time_moment(x, t)
    
    def parameters(self):
        return self.lsb_model.parameters()

    def to(self, device):
        self.device = device
        self.lsb_model.to(device)
    
    def eval(self, sampler):
        
        x_sampler = sampler.x_sample
        y_sampler = sampler.y_sample
        x_gt = x_sampler(5120).to(self.device)
        y_gt = y_sampler(51200).to(self.device)
        
        if self.eot_finish:
            print('EOT')
            y_pred =  eot_gaussian(self.lsb_model(x_gt), y_gt, self.eps)
        else:
            y_pred =  self.lsb_model(x_gt)
        # y_pred = self.lsb_model(x_gt)

        y_pred, y_gt = y_pred.cpu(), y_gt.cpu()
        
        bw_uvp_terminal = compute_BW_UVP_by_gt_samples(y_pred, y_gt)
        print(f'BW_UVP_FW: {bw_uvp_terminal}')

    def fit(self, sampler: Sampler, max_iter=5000, batch_size=256, lr=1e-3, val_freq=300, eval_batch_size=5120, init=True, eps_milestones=[]):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        x_sampler = sampler.x_sample
        y_sampler = sampler.y_sample

        if len(eps_milestones) > 0:
            start_esp = self.eps * 10
            self.lsb_model.set_epsilon(start_esp)

        if self.eot_finish:
            self.saved_y_samples = y_sampler(51200)

        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.3) 

        if init:
            self.init_by_samples(y_sampler(self.lsb_model.n_potentials))

        for i in range(max_iter):
            if (i + 1) % 100 == 0:
                print(f'Iter {i}')
            
            X0, X1 = x_sampler(batch_size).to(self.device), y_sampler(batch_size).to(self.device)
            # self.saved_y_samples = deepcopy(X1)
            # print('X0, X1', X0.device, X1.device)
            
            log_potential = self.lsb_model.get_log_potential(X1)
            log_C = self.lsb_model.get_log_C(X0)

            D_loss = (-log_potential + log_C).mean()

            opt.zero_grad()
            
            D_loss.backward()
            
            # D_gradient_norm = torch.nn.utils.clip_grad_norm_(self.lsb_model.parameters(), max_norm=1)
            opt.step()
            # scheduler.step()
            # print('LR: ', scheduler.get_lr())

            if (i + 1) % val_freq == 0:
                print(f'Iter {i} validation...')
                self.eval(sampler)
            
            # if (i + 1) % (max_iter - 1000)  == 0:
            #     old_lr = next(iter(opt.param_groups))['lr']
            #     for g in opt.param_groups:
            #         g['lr'] = old_lr * 0.3
            
            if (i + 1) in eps_milestones:
                print('Lowering eps')
                new_eps = self.lsb_model.epsilon / (10 ** (1 / len(eps_milestones)))
                print('New eps', new_eps)
                self.lsb_model.set_epsilon(new_eps)

            
            # wandb.log({f'D gradient norm' : D_gradient_norm.item()}, step=step)
            # wandb.log({f'D_loss' : D_loss.item()}, step=step)

    def get_drift(self):
        # def ret_val(x, t):
        #     print(x.shape, t.shape)
        #     return self.lsb_model.get_drift(x, t.squeeze()) 
        # return ret_val
        return lambda x, t: self.lsb_model.get_drift(x, t.squeeze()) 

    def get_std_fn(self):
        return lambda t: self.eps
    
    def sample_path_drift(self, x):

        n_em_steps = 99
        eps_fn = lambda t: self.eps
        
        gt_sde = GTSDEByDrift(self.get_drift(), eps_fn, n_em_steps)

        def return_fn():
            x_samples = x
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t
        
        return return_fn()

    def sample_path_bb(self, x, n_steps=100):
        def brownian_bridge_sampling(x_0, x_1, t_0, t_1, t):
            
            # N x T x D
            # mean = x_0 + ((t - t_0) / (t_1 - t_0)).reshape([x_0.shape[0], x_0.shape[1]]) * (x_1 - x_0)

            mean = x_0 + ((t - t_0) / (t_1 - t_0)) * (x_1 - x_0)

            std = torch.sqrt(self.eps * (t - t_0) * (t_1 - t) / (t_1 - t_0))
                        
            x_t = mean + std * torch.randn_like(x_1)
            
            return x_t, t.repeat([x_t.shape[0]]).unsqueeze(-1)
            
        x_0 = x
        x_1 = self(x_0)

        delta_t = 1 / n_steps

        x_t_storage = []
        t_storage = []
        x_t = x_0

        for t in torch.arange(0, 1 - delta_t, delta_t):
            x_t, t = brownian_bridge_sampling(x_t, x_1, t, 1, t + delta_t)
            x_t_storage.append(x_t)
            t_storage.append(t)
            # t = t + delta_t
                
        return torch.stack(x_t_storage, dim=1), torch.stack(t_storage, dim=1)

    def sample_path(self, x, n_steps=100):
        
        def brownian_bridge_sampling(x_0, x_1, t_0, t_1, t):
            
            # N x T x D

            mean = x_0 + ((t - t_0) / (t_1 - t_0)).reshape([x_0.shape[0], x_0.shape[1], 1, 1, 1]) * (x_1 - x_0)
            
            std = self.eps * (t - t_0) * (t_1 - t) / (t_1 - t_0)
                        
            x_t = mean + std.reshape([std.shape[0], std.shape[1], 1, 1, 1]) * torch.randn_like(x_1)
            
            return x_t, t
            
        x_0 = x
        x_1 = self(x_0)

        delta_t = 1 / n_steps

        x_t_storage = []
        t_storage = []
        x_t = x_0

        for t in torch.arange(0, 1, delta_t):
            t = t + delta_t
            x_t, t = brownian_bridge_sampling(x_t, x_1, t, 1, t + delta_t)
            x_t_storage.append(x_t)
            t_storage.append(t)
        
        return torch.stack(x_t_storage, dim=1), torch.stack(t_storage, dim=1)
        # Do binary browninan bridge
        # get drift
        # and do EM

    def __str__(self) -> str:
        return f'LightSB_D_{self.lsb_model.dim}_P_{self.lsb_model.n_potentials}_eps_{self.lsb_model.epsilon}'
        
    def log_prob(self, x_1, cond_x):
        # log p(x_1 | cond_x)
        return self.lsb_model.log_prob(x_1, cond_x)

    def sample_bb(self, t, x_0, x_1, t_0=0, t_1=1):
        return BrownianBridge(self.eps, x_0, x_1, t_0, t_1).sample(t)

    def get_bb_mean_std(self, t, x_0, x_1, t_0=0, t_1=1):
        return BrownianBridge(self.eps, x_0, x_1, t_0, t_1).get_mean_std(t)

    def score_bb(self, x, x_0, x_1, t):
        # score of Brownian Bridge dist
        return BrownianBridge(self.eps, x_0, x_1).score(x, t)
    

class NeuralNetworkDrift(Bridge):
    def __init__(self, network, eps, skip_n_last_noise=0) -> None:
        super().__init__()
        self.network = network
        self.sigma = eps
        self.skip_n_last_noise = skip_n_last_noise

    def get_drift(self):
        return self.network
    
    def forward(self, x):
        return self(x)

    def __call__(self, x, em_steps=100):

        drift_fn = self.get_drift()

        sigma_fn = lambda t: torch.ones_like(t) * self.sigma
        
        # Time reverse happens in traning
        model_x_sample_fn = lambda x: SDE(drift_fn, sigma_fn, is_forward=True, skip_n_last=self.skip_n_last_noise).sample(
            x, num_steps=em_steps, device=next(self.network.parameters()).device)[0][:, -1]

        return model_x_sample_fn(x)

    def parameters(self):
        return self.network.parameters()

# class SongSDE(Bridge):
#     # SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS, Song'21
#     # Reverse of VP SDE
#     def __init__(self, data_sampler, score_model) -> None:
#         super().__init__()
#         self.data_sampler = data_sampler
#         self.dim = data_sampler(1).squeeze().shape
#         print('dim', self.dim)
#         self.vp_sde = VPSDESampler(data_sampler, self.dim)
#         self.score_model = score_model

#     def get_loss(self, batch_size=512):
#         # output score matching loss
#         model_device = next(iter(self.score_model.parameters())).device

#         x = self.data_sampler(batch_size).to(model_device)
#         t = torch.distributions.uniform.Uniform(0, 1).sample([batch_size]).to(x.device)
#         # print('T: ', t.shape, 'X: ', x.shape)
        
#         score_reference = self.vp_sde.score(x, t)
#         model_out = self.score_model(x, t)

#         loss = torch.nn.functional.mse_loss(score_reference, model_out)
#         return loss
    
#     def forward(self, x):
#         return self(x)

#     def get_drift(self, x, t):
#         std = self.get_std(t)
#         score = self.score_model(x, t)
#         return -(self.vp_sde.get_drift()(x, t) - std ** 2 * score)

#     def get_std(self, t):
#         return self.vp_sde.eps_fn(t)

#     def __call__(self, x, em_steps=100, skip_n_last_noise=2):
        
#         drift_fn = lambda x, t: self.get_drift(x, t)

#         sigma_fn = lambda t: self.get_std(t)
        
#         model_x_sample_fn = lambda x: SDE(drift_fn, sigma_fn, is_forward=False, skip_n_last=skip_n_last_noise).sample(
#             x, num_steps=em_steps, device=next(self.score_model.parameters()).device)[0][:, -1]
        
#         return model_x_sample_fn(x)
    
#     def train(self, max_iter=10000, lr=1e-3):
#         opt = torch.optim.Adam(self.score_model.parameters(), lr=lr)

#         for i in range(max_iter):
#             loss = self.get_loss()

#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#             if i % 100 == 0:
#                 print(i)


# Learnable Sampler? Parametric joint conditioanl distribution?
class WGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x):
        return super().__call__(x)
