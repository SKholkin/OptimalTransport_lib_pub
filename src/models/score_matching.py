from src.bridge import Bridge
from src.samplers.samplers import VPSDESampler, LSBSampler
from src.sde import SDE
from src.dists_utils import Gaussian

import torch
from tqdm import tqdm



def sampling_via_score_fn(x_samples, forward_drift, std_fn, score_fn, em_steps=100, skip_n_last_noise=2, device='cpu', is_grad=False, probability_flow_ode=False):
    # score_coef = 0.5 if probability_flow_ode else 1

    reverse_drift = lambda x, t: -(forward_drift(x, t) - (std_fn(t) ** 2) * score_fn(x, t))

    # if probability_flow_ode:
    #     std_fn = lambda t: 0

    # sde_std_fn = std_fn

    model_x_sample_fn = lambda x: SDE(reverse_drift, std_fn, is_forward=False, skip_n_last=skip_n_last_noise).sample(
        x_samples, num_steps=em_steps, device=device, grad=is_grad)[0][:, -1]
    
    return model_x_sample_fn(x_samples)


class SongSDE(Bridge):
    # SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS, Song'21
    # Reverse of VP SDE
    def __init__(self, data_sampler, score_model) -> None:
        super().__init__()
        self.data_sampler = data_sampler
        self.dim = data_sampler(10).squeeze().shape[-1]
        print('dim', self.dim)
        self.vp_sde = VPSDESampler(data_sampler, self.dim)
        self.score_model = score_model

    def get_loss(self, batch_size=512):
        # output score matching loss
        model_device = next(iter(self.score_model.parameters())).device

        x_0 = self.data_sampler(batch_size).to(model_device)
        t = torch.distributions.uniform.Uniform(0, 1).sample([batch_size]).to(x_0.device)
        x_t = self.vp_sde.sample_t(x_0, t)
        
        score_reference = self.vp_sde.score(x_0, x_t, t)
        model_out = self.score_model(x_t, t)
        std = self.get_std(t).reshape([-1, 1])

        loss = torch.nn.functional.mse_loss(score_reference * std, model_out * std)
        return loss
    
    def forward(self, x):
        return self(x)

    def get_drift(self, x, t):
        std = self.get_std(t)
        score = self.score_model(x, t)
        return -(self.vp_sde.get_drift()(x, t) - std ** 2 * score)

    def get_std(self, t):
        return self.vp_sde.eps_fn(t)

    def __call__(self, x, em_steps=100, skip_n_last_noise=0):
        
        drift_fn = lambda x, t: self.get_drift(x, t)

        sigma_fn = lambda t: self.get_std(t)
        
        model_x_sample_fn = lambda x: SDE(drift_fn, sigma_fn, is_forward=False, skip_n_last=skip_n_last_noise).sample(
            x, num_steps=em_steps, device=next(self.score_model.parameters()).device)[0][:, -1]
        
        return model_x_sample_fn(x)
    
    # def sample_path(self, x, em_steps=100, skip_n_last_noise=2):
        
    #     drift_fn = lambda x, t: self.get_drift(x, t)

    #     sigma_fn = lambda t: self.get_std(t)
        
    #     model_x_sample_fn = lambda x: SDE(drift_fn, sigma_fn, is_forward=False, skip_n_last=skip_n_last_noise).sample(
    #         x, num_steps=em_steps, device=next(self.score_model.parameters()).device)[0]
        
    #     return model_x_sample_fn(x)
    
    def sample(self, x_samples, probability_flow_ode=False):
        lsb_model = self.forward_sde.lsb_model
        forward_drift = lsb_model.get_drift()
        std_fn = lsb_model.get_std_fn()
        score_fn = self.score_net
        device = next(iter(self.score_net.parameters())).device
        return sampling_via_score_fn(x_samples, forward_drift, std_fn, score_fn, em_steps=100, skip_n_last_noise=2,
                                      device=device, is_grad=True, probability_flow_ode=probability_flow_ode)
    
    
    
    def train(self, max_iter=10000, lr=1e-3):
        opt = torch.optim.Adam(self.score_model.parameters(), lr=lr)
        pbar = tqdm(range(max_iter))

        for i in pbar:
            loss = self.get_loss()

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(f'Iter: {i} Loss: {loss.item()}')


class ScoreMatchingReverseLightSB(Bridge):
    def __init__(self, score_net, forward_sde: LSBSampler) -> None:
        super().__init__()
        self.score_net = score_net
        self.forward_sde = forward_sde

    def forward(self, x):
        return x
    
    def get_loss(self, batch_size=512):
        # score matching training 
        lsb_model = self.forward_sde.lsb_model

        model_device = next(iter(self.score_net.parameters())).device

        x_0 = self.forward_sde.x_sample(batch_size).to(model_device)
        # print(x_0.shape)  
        t = torch.distributions.uniform.Uniform(0, 1).sample([batch_size]).reshape([-1, 1]).to(model_device)
        x_1 = self.forward_sde.sample_conditional(x_0)
        x_t = lsb_model.sample_bb(t, x_0, x_1, torch.zeros([x_0.shape[0], 1]).to(model_device), torch.ones([x_0.shape[0], 1]).to(model_device))
        # print(x_t.shape)
        score_target = lsb_model.score_bb(x_t, x_0, x_1, t)

        mean, std = lsb_model.get_bb_mean_std(t, x_0, x_1, torch.zeros([x_0.shape[0], 1]).to(model_device), torch.ones([x_0.shape[0], 1]).to(model_device))
        
        model_out = self.score_net(x_t, t)

        loss = torch.nn.functional.mse_loss(score_target * std, model_out * std)

        return loss
    
    def sample(self, x_samples, probability_flow_ode=False):
        lsb_model = self.forward_sde.lsb_model
        forward_drift = lsb_model.get_drift()
        std_fn = lsb_model.get_std_fn()
        score_fn = self.score_net
        device = next(iter(self.score_net.parameters())).device
        return sampling_via_score_fn(x_samples, forward_drift, std_fn, score_fn, em_steps=100, skip_n_last_noise=2,
                                      device=device, is_grad=True, probability_flow_ode=probability_flow_ode)
    