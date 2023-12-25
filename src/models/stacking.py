from src.models.bridge import Bridge, LightSBBridge
from src.samplers.samplers import LSBSampler, Sampler, CustomSampler

import torch

class StackedLSB(Bridge):
    def __init__(self, dim, n_potentials, eps, n_stack, is_diagonal=True):
        super().__init__()

        self.dim = dim
        self.n_potentials = n_potentials
        self.eps = eps
        self.is_diagonal = is_diagonal
        self.n_stack = n_stack
        self.lsb_stack = torch.nn.ModuleList([LightSBBridge(self.dim, self.n_potentials, self.eps, self.is_diagonal) for i in range(self.n_stack)])
        
    def fit(self, sampler: Sampler, max_iter=10000, batch_size=128, lr=1e-03, val_freq=300, eval_bs=512):
        
        self.lsb_samplers_stack = []
        original_sampler = sampler
        
        for i in range(self.n_stack):
            print(f'Fitting LSB {i}')
            lsb_i = self.lsb_stack[i]
            
            lsb_i.to('cuda:0')
            lsb_i.lsb_model.to('cuda:0')
            
            lsb_i.fit(sampler, max_iter=max_iter, batch_size=batch_size, lr=lr, val_freq=val_freq, eval_batch_size=eval_bs)
            lsb_sampler = LSBSampler(lsb_i, sampler.x_sample, device='cuda:0')
            sampler = CustomSampler(lsb_sampler.y_sample, original_sampler.y_sample)
            self.lsb_samplers_stack.append(lsb_sampler)
        
    def get_sampler(self):
        return self.lsb_samplers_stack[-1]
            
    def eval(self):
        lsb_sampler_last = self.lsb_samplers_stack[-2]
        lsb_last =  self.lsb_stack[-1]
        lsb_last.eval(lsb_sampler_last)
        
    def __str__(self):
        return f'LightSB_Stacked_D_{self.dim}_P_{self.n_potentials}_diag_{self.is_diagonal}_eps_{self.eps}'
    
    def create_samplers(self, sampler):

        self.lsb_samplers_stack = []
        original_sampler = sampler
        
        for i in range(self.n_stack):
            lsb_i = self.lsb_stack[i]
            
            lsb_i.to('cuda:0')
            lsb_i.lsb_model.to('cuda:0')
            
            lsb_sampler = LSBSampler(lsb_i, sampler.x_sample, device='cuda:0')
            sampler = CustomSampler(lsb_sampler.y_sample, original_sampler.y_sample)
            self.lsb_samplers_stack.append(lsb_sampler)

