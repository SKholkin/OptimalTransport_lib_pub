import torch
import math

from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

class Gaussian:
    def __init__(self, mean, std) -> None:
        pass

    @staticmethod
    def score(mean, std, x):
        return - (x - mean) / (std ** 2)

    @staticmethod
    def empirical_entropy(mean, log_std, x):
        batch_size = mean.shape[0]
        mean, log_std, x = mean.reshape([batch_size, -1]), log_std.reshape([batch_size, -1]), x.reshape([batch_size, -1])
        std = torch.exp(log_std)

        return (log_std.sum(-1) + 1 / 2 * torch.norm((x - mean) / std, dim=-1) ** 2 ).mean()
    
    @staticmethod
    def entropy(mean, log_std):
        p = Independent(Normal(mean, torch.exp(log_std)), 1)
        return p.entropy()

    @staticmethod
    def kl_div(mean_1, log_std_1, mean_2, log_std_2):
        batch_size = mean_1.shape[0]
        device = mean_1.device
        # print(log_std_1.shape)
        # print('diagonal cov matrix', (torch.exp(log_std_1).unsqueeze(-1) * torch.eye(mean_1.shape[-1]).unsqueeze(0).repeat([batch_size, 1, 1])).shape)
        
        # cov_matr_1 = (torch.exp(log_std_1 * 2).unsqueeze(-1) * torch.eye(mean_1.shape[-1], device=device).unsqueeze(0).repeat([batch_size, 1, 1]))
        # cov_matr_2 = (torch.exp(log_std_2 * 2).unsqueeze(-1) * torch.eye(mean_1.shape[-1], device=device).unsqueeze(0).repeat([batch_size, 1, 1]))

        # p = torch.distributions.multivariate_normal.MultivariateNormal(mean_1, cov_matr_1)
        # q = torch.distributions.multivariate_normal.MultivariateNormal(mean_2, cov_matr_2)

        p = Independent(Normal(mean_1, torch.exp(log_std_1)), 1)
        q = Independent(Normal(mean_2, torch.exp(log_std_2)), 1)

        # kl_1 = torch.distributions.kl.kl_divergence(p, q)

        return torch.distributions.kl.kl_divergence(p, q)

