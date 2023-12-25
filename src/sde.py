import math
import torch


class SDE:
    def __init__(self, drift, sigma_fn, is_forward=True, skip_n_last=0):
        self.drift = drift
        self.sigma_fn = sigma_fn
        self.is_forward = is_forward
        self.skip_n_last = skip_n_last

    def _sample(self, x_0, num_steps=100, device='cpu'):
        # sample using Euler-Maryama scheme
        # x_t_res: N X T X D 
        x_t = x_0.to(device)
        x_t_res = []

        delta_t = 1. / num_steps

        t = torch.arange(0, 1, step=delta_t).to(device)
        if not self.is_forward:
            t = t.flip(0)

        for i in range(num_steps):
            
            # print(x_t.device, t.device)
            if num_steps - i < self.skip_n_last:
                x_t = x_t + self.drift(x_t, t[i].reshape([1, -1]).repeat([x_t.shape[0], 1]).to(device)) * delta_t
            else:
                x_t = x_t + self.drift(x_t, t[i].reshape([1, -1]).repeat([x_t.shape[0], 1]).to(device)) * delta_t + math.sqrt(delta_t) * self.sigma_fn(t[i]) * torch.randn_like(x_t).to(device)
            
            x_t = x_t.detach()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            x_t_res.append(x_t.unsqueeze(1))

        x_t_res = torch.cat(x_t_res, dim=1)
        
        return x_t_res, t.reshape([1, -1]).repeat([x_t_res.shape[0], 1])

    def sample(self, x_0, num_steps=100, device='cpu', grad=False):
        if grad:
            return self._sample(x_0, num_steps, device)
        else:
            with torch.no_grad():
                return self._sample(x_0, num_steps, device)
    
    @torch.no_grad()
    def make_several_trajectories(self, x_0, num_steps=100, device='cpu', n_traj=4):
        # output: N X N_traj X C X H X W
        
        x_t_res = []

        for j in range(n_traj):

            x_t = x_0.to(device).clone()
            delta_t = 1. / num_steps

            t = torch.arange(0, 1, step=delta_t)
            if not self.is_forward:
                t = t.flip(0)

            for i in range(num_steps):
                noise = math.sqrt(delta_t) * self.sigma_fn(t[i]) * torch.randn_like(x_t).to(device)
                x_t = x_t + self.drift(x_t, t[i].reshape([1, -1]).repeat([x_t.shape[0], 1])) * delta_t
                if i < num_steps - 3:
                    x_t += noise

            x_t_res.append(x_t.unsqueeze(1))
        
        x_t_res = torch.cat(x_t_res, dim=1)
        
        return x_t_res
