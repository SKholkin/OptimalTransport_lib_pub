import torch
from torch import nn

from src.models.bridge import Bridge
from src.models.bridge import BrownianBridge


class I2SB(Bridge):
    def __init__(self, score_net, eps) -> None:
        super().__init__()
        self.score_net = score_net
        self.eps = eps

    def __call__(self, x_1, steps=100):
        return self.sample(x_1, steps=steps)

    def parameters(self):
        return self.score_net.parameters()
    
    def get_drift(self):
        return lambda x, t: -(self.eps ** 2) * self.score_net(x, t)
    
    def get_std(self):
        return lambda t: self.eps

    @torch.no_grad()
    def sample(self, x_1, steps=100):
        
        device=next(self.score_net.parameters()).device

        x_t = x_1
        delta_t =1. / steps
        for t in torch.arange(0, 1, step=delta_t).flip(0):
            t = t.to(device)
            x_0_prediction = x_t - self.score_net(x_t, t) * self.eps * t
            if float(t.item()) != 0.:
                x_t = BrownianBridge(self.eps, x_0_prediction, x_t, t_0=0., t_1=t).sample(t - delta_t)
            else:
                x_t = x_0_prediction

        return x_t

    def step(self, x_1, x_0):
        # x_1 is noise
        # x_0 is data
        
        batch_size = x_0.shape[0]
        t = torch.distributions.uniform.Uniform(0, 1).sample([batch_size]).to(x_0.device).unsqueeze(-1)

        x_t = BrownianBridge(self.eps, x_0, x_1, t_0=0., t_1=1.).sample(t)

        score_hat = self.score_net(x_t, t)

        return self.loss(score_hat, x_t, x_0, t).mean()

    def loss(self, score_hat, x_t, x_0, t):
        return torch.norm(((score_hat * self.eps * t - (x_t - x_0)).reshape([x_0.shape[0], -1])), dim=-1)
