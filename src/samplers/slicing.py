from src.samplers.samplers import LSBSampler, Sampler, CustomSampler, VPSDESampler
import torch

class VPSDESlice(VPSDESampler):
    def __init__(self, x_sampler, dim, t_0, t_1) -> None:
        super().__init__(x_sampler, dim)
        self.t_0 = t_0
        self.t_1 = t_1
    
    def x_sample(self, batch_size):
        x_0_samples = VPSDESampler.x_sample(self, batch_size)
        return self.sample_t(x_0_samples, torch.zeros([batch_size]))
    
    def y_sample(self, batch_size):
        x_0_samples = VPSDESampler.x_sample(self, batch_size)
        return self.sample_t(x_0_samples, torch.ones([batch_size]))
    
    def __t_transform(self, t):
        return torch.clip(t  * (self.t_1 - self.t_0)  + self.t_0, min=0, max=1)

    def _marginal(self, x_0, t):
        # print(f'Right Marginal \nt before {t[0]} t after {self.__t_transform(t)[0]}')
        t = self.__t_transform(t)
        return VPSDESampler._marginal(self, x_0, t)

    def get_drift(self):
        return lambda x, t: VPSDESampler.get_drift(self)(x, self.__t_transform(t))
    
    def eps_fn(self, t):
        t = self.__t_transform(t)
        return VPSDESampler.eps_fn(self, t)
    

class ICFMSlice(Sampler):
    def __init__(self, sampler, t_0=0, t_1=1):
        self.x_sampler = sampler.x_sample
        self.y_sampler = sampler.y_sample
        self.t_0 = t_0
        self.t_1 = t_1
        self.dim = sampler.dim

    def x_sample(self, batch_size):
        x_0 = self.x_sampler(batch_size)
        x_1 = self.y_sampler(batch_size)
        return x_0 + self.t_0 * (x_1 - x_0)
    
    def y_sample(self, batch_size):
        x_0 = self.x_sampler(batch_size)
        x_1 = self.y_sampler(batch_size)
        return x_0 + self.t_1 * (x_1 - x_0)
