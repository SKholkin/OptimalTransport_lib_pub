from typing import Any
import torch
import sklearn

from src.samplers.samplers import Sampler

class OneDistSampler:
    def __init__(self) -> None:
        pass

    def __call__(self, batch_size) -> Any:
        return self.sample(batch_size)

    def sample(self, batch_size):
        raise NotImplementedError('Abstract Class')
    
class JoinOneDistSamplers(Sampler):
    def __init__(self, x_sampler: OneDistSampler, y_sampler: OneDistSampler) -> None:
        super().__init__()
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        if self.x_sampler.dim != self.y_sampler.dim:
            raise ValueError(f'Dimensions of Distributions do not Aligh {self.x_sampler.dim} vs {self.y_sampler.dim}')

    def x_sample(self, batch_size):
        return self.x_sampler(batch_size)
    
    def y_sample(self, batch_size):
        return self.y_sampler(batch_size)

class GaussianDist(OneDistSampler):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def sample(self, batch_size):
        return torch.randn([batch_size, self.dim])

class TwoMoons(OneDistSampler):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 2

    def sample(self, batch_size):
        return 2 * torch.Tensor(sklearn.datasets.make_moons(n_samples=batch_size)[0])
