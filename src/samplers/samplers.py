import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import torchvision

from eot_benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler, 
)
from eot_benchmark.image_benchmark import ImageBenchmark

# for GT SDE
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from src.sde import SDE

from typing import Any, List
from scipy.linalg import sqrtm


def random_color(im):
    hue = 360*np.random.rand()
    d = (im *(hue%60)/60)
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    c_im = torch.zeros((3, im.shape[1], im.shape[2]))
    H = round(hue/60) % 6    
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]

class Sampler:
    def __init__(self) -> None:
        pass
        device = 'cpu'
        self.dim = None

    def x_sample(self, batch_size):
        raise NotImplementedError('Abstract Class')
    
    def y_sample(self, batch_size):
        raise NotImplementedError('Abstract Class')

    # add cond_y_sample that uses y_sample when is not implemented

    def cond_y_sample(self, x):
        return self.y_sample(x.shape[0])
    
    def gt_sample(self, batch_size):
        raise NotImplementedError('GT sampler is not available')

    def cotrol_sample_x(self, batch_size):
        raise NotImplementedError('Control batch is not available')

    def cotrol_sample_y(self, batch_size):
        raise NotImplementedError('Control batch is not available')

    def path_sampler(self):
        raise NotImplementedError('Path sampler is not available')

    def __str__(self) -> str:
        return str(type(self)).split("'")[1]
    
    def __call__(self, x) -> Any:
        raise NotImplementedError('Not implemented conditional sampler')
    
    def sample_t(self, x_or_batch_size, t):
        raise NotImplementedError()
            

class CustomSampler(Sampler):
    def __init__(self, x_sampler, y_sampler):
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
    
    @torch.no_grad()
    def x_sample(self, batch_size):
        return self.x_sampler(batch_size)
    
    @torch.no_grad()
    def y_sample(self, batch_size):
        return self.y_sampler(batch_size)

def get_mnist_by_number(number, transform, is_train=True):
    dataset = torchvision.datasets.MNIST(root="./data", train=is_train, transform=transform, download=True)

    idx = (dataset.targets==number)
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    
    return dataset


class ColoredMnistSampler(Sampler):
    def __init__(self, batch_size=512) -> None:
        super().__init__()
        
        mnist_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                random_color,
                torchvision.transforms.Normalize([0.5],[0.5])
            ])
        
        dataset_2 = get_mnist_by_number(2, mnist_transform, is_train=True)
        
        dataset_3 =  get_mnist_by_number(3, mnist_transform, is_train=True)
        
        dataloader_2 = torch.utils.data.DataLoader(dataset_2, batch_size=batch_size, shuffle=True)        
        dataloader_3 = torch.utils.data.DataLoader(dataset_3, batch_size=batch_size, shuffle=True)

        control_batch_x_train = next(iter(torch.utils.data.DataLoader(dataset_2, batch_size=512, shuffle=False)))[0]
        control_batch_x_val= next(iter(torch.utils.data.DataLoader(get_mnist_by_number(2, mnist_transform, is_train=False), batch_size=512, shuffle=False)))[0]

        self.control_batch_x = torch.zeros_like(torch.cat([control_batch_x_train, control_batch_x_val], dim=0))
        self.control_batch_x[::2] = control_batch_x_train
        self.control_batch_x[1::2] = control_batch_x_val
        
        control_batch_y_train = next(iter(torch.utils.data.DataLoader(dataset_3, batch_size=512, shuffle=False)))[0]        
        control_batch_y_val= next(iter(torch.utils.data.DataLoader(get_mnist_by_number(3, mnist_transform, is_train=False), batch_size=512, shuffle=False)))[0]

        self.control_batch_y = torch.zeros_like(torch.cat([control_batch_y_train, control_batch_y_val], dim=0))
        # print(self.control_batch_y.shape)
        self.control_batch_y[::2] = control_batch_y_train
        self.control_batch_y[1::2] = control_batch_y_val

        self.dataset_2, self.dataset_3 = dataset_2, dataset_3

        self.dataloader_2, self.dataloader_3 = dataloader_2, dataloader_3
        
    def x_sample(self, batch_size):
        return next(iter(self.dataloader_2))[0][:batch_size]
    
    def y_sample(self, batch_size):
        return next(iter(self.dataloader_3))[0][:batch_size]

    def control_sample_x(self, batch_size):
        static_batch = self.control_batch_x[:batch_size // 2]
        # print(static_batch.shape)

        # static_batch = self.control_batch_y[:batch_size // 2]
        dynamic_batch = self.x_sample(batch_size // 2)
        # print(dynamic_batch.shape)
        # print(torch.cat([static_batch, dynamic_batch], dim=0).shape)

        return torch.cat([static_batch, dynamic_batch], dim=0)

    def control_sample_y(self, batch_size):
        # control_dataloader = torch.utils.data.DataLoader(self.dataset_3, batch_size=batch_size // 2, shuffle=False)
        static_batch = self.control_batch_y[:batch_size // 2]
        # static_batch = next(iter(control_dataloader))[0]
        dynamic_batch = self.y_sample(batch_size // 2)
        
        return torch.cat([static_batch, dynamic_batch], dim=0)


class GaussianToMNIST(Sampler):
    def __init__(self, batch_size=512, size=16) -> None:
        super().__init__()

        # self.mnist_size = [1, 28, 28]

        mnist_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size),
                torchvision.transforms.Normalize([0.5],[0.5])
            ])

        mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=mnist_transform, download=True)
        
        # idx = (mnist.targets==2)
        # mnist.targets = mnist.targets[idx]
        # mnist.data = mnist.data[idx]
        self.mnist = mnist
        
        self.mnist_dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
        self.x_sampler = torch.distributions.normal.Normal(0, 1)
        self.dim = size ** 2

    def x_sample(self, batch_size):
        return self.x_sampler.sample([batch_size, self.dim])
    
    def y_sample(self, batch_size):
        mnist_dataloader = torch.utils.data.DataLoader(self.mnist, batch_size=batch_size, shuffle=True)
        
        return next(iter(mnist_dataloader))[0].reshape([batch_size, -1])
    
    def mnist_dataloder(self, batch_size=512):
        return torch.utils.data.DataLoader(self.mnist, batch_size=batch_size, shuffle=False)


class GaussianToCIFAR(Sampler):
    def __init__(self, batch_size=512) -> None:
        super().__init__()

        # self.mnist_size = [1, 28, 28]

        cifar_transform = torchvision.transforms.Compose([                
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5],[0.5])
            ])
        
        cifar = torchvision.datasets.CIFAR10(root="./data", train=True, transform=cifar_transform, download=True)
        
        # idx = (mnist.targets==2)
        # mnist.targets = mnist.targets[idx]
        # mnist.data = mnist.data[idx]
        self.dataset = cifar
        
        self.mnist_dataloader = torch.utils.data.DataLoader(cifar, batch_size=batch_size, shuffle=True)
        self.x_sampler = torch.distributions.normal.Normal(0, 1)
        self.dim = 32 ** 2 * 3

    def x_sample(self, batch_size):
        return self.x_sampler.sample([batch_size, self.dim])
    
    def y_sample(self, batch_size):
        mnist_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        return next(iter(mnist_dataloader))[0].reshape([batch_size, -1])


from tqdm import tqdm
class CelebaCustomDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_size=162770):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        size = 32
        self.dataset_size = dataset_size
        celeba_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((size, size)),
                torchvision.transforms.Normalize([0.5],[0.5])
            ])
        
        self.basic_dataset = torchvision.datasets.CelebA(root="data", split='train', transform=celeba_transform, download=True)
        self.storage = []
        self.preload()
    
    def preload(self):
        
        for i in tqdm(range(min(self.dataset_size, len(self.basic_dataset)))):

            self.storage.append(self.basic_dataset[i])

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, idx):
        return self.storage[idx]
    

class GaussianToCelebA(Sampler):
    def __init__(self, eps=1, batch_size=512, dataset_size=162770) -> None:
        super().__init__()

        size = 32

        celeba_train = CelebaCustomDataset(dataset_size)

        self.celeba_train = celeba_train
        self.eps = eps
        
        self.dataloader = torch.utils.data.DataLoader(celeba_train, batch_size=batch_size, shuffle=True)
        self.x_sampler = torch.distributions.normal.Normal(0, 1)
        self.dim = size ** 2 * 3
        self.basic_batch_size = 2048
        self.dataloader_iter = iter(torch.utils.data.DataLoader(self.celeba_train, batch_size=self.basic_batch_size, shuffle=True))

    def x_sample(self, batch_size):
        return self.x_sampler.sample([batch_size, self.dim])
    
    def y_sample(self, batch_size):
        dataloader = torch.utils.data.DataLoader(self.celeba_train, batch_size=batch_size, shuffle=True)
        return next(iter(dataloader))[0].reshape([batch_size, -1])
    

class GroundTruthSDE(nn.Module):
    
    def __init__(self, potential_params, eps, n_steps, is_diagonal=False):
        super().__init__()
        probs, mus, sigmas = potential_params
        self.eps = eps
        self.n_steps = n_steps
        self.is_diagonal = is_diagonal
        
        self.register_buffer("potential_probs", probs)
        self.register_buffer("potential_mus", mus)
        self.register_buffer("potential_sigmas", sigmas)

    def forward(self, x):
        t_storage = [torch.zeros([1])]
        trajectory = [x.cpu()]
        for i in range(self.n_steps):
            delta_t = 1 / self.n_steps
            t = torch.tensor([i / self.n_steps])
            drift = self.get_drift(x, t)

            rand = np.sqrt(self.eps) * np.sqrt(delta_t)*torch.randn(*x.shape).to(x.device)

            x = (x + drift * delta_t + rand).detach()

            trajectory.append(x.cpu())
            t_storage.append(t)
            
        return torch.stack(trajectory, dim=0).transpose(0, 1), torch.stack(t_storage, dim=0).unsqueeze(1).repeat([1, x.shape[0], 1])

    def sample(self, x):
        return self.forward(x)

    def get_drift(self, x_current, t):
        probs = self.potential_probs
        sigmas = self.potential_sigmas
        mus = self.potential_mus

        n_components, dim = mus.shape[0], mus.shape[1]

        if self.is_diagonal:
            a = torch.diag(self.eps*(1.0-t)*torch.ones(dim))[None, :, :].expand(n_components, dim, dim)

            identity_diag = torch.diagonal(a).transpose(0, 1).to(x_current.device)
            a = sigmas + identity_diag
            new_comp = Independent(Normal(loc=mus, scale=torch.sqrt(a)), 1)

        else:
            identity = torch.diag(self.eps*(1.0-t)*torch.ones(dim))[None, :, :].expand(n_components, dim, dim).to(x_current.device)
            new_comp = MultivariateNormal(loc=mus, covariance_matrix=(sigmas) + identity)

        phi_t_distr = MixtureSameFamily(Categorical(self.potential_probs), new_comp)
        
        x_current.requires_grad = True
        log_p = phi_t_distr.log_prob(x_current.reshape([x_current.shape[0], -1]))
        
        return self.eps*torch.autograd.grad(log_p, x_current, grad_outputs=torch.ones(x_current.shape[0]).to(x_current.device))[0]


class GTSDEByDrift():
    def __init__(self, drift_fn, eps_fn, n_steps) -> None:
        self.drift_fn = drift_fn
        self.n_steps = n_steps
        self.eps_fn = eps_fn

    def forward(self, x):
        t_storage = [torch.zeros([1])]
        trajectory = [x.cpu()]
        for i in range(self.n_steps):
            delta_t = 1 / self.n_steps
            t = torch.tensor([i / self.n_steps])
            drift = self.drift_fn(x, t)
            
            rand = np.sqrt(self.eps_fn(t)) * np.sqrt(delta_t)*torch.randn(*x.shape).to(x.device)

            x = (x + drift * delta_t + rand).detach()

            trajectory.append(x.cpu())
            t_storage.append(t)
            
        return torch.stack(trajectory, dim=0).transpose(0, 1), torch.stack(t_storage, dim=0).unsqueeze(1).repeat([1, x.shape[0], 1])
    
    def sample(self, x):
        return self.forward(x)


class EOTGMMSampler(Sampler):
    def __init__(self, dim, eps, batch_size=64, download=False) -> None:
        super().__init__()
        eps = eps if int(eps) < 1 else int(eps)

        self.dim = dim
        self.eps = eps
        self.x_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=dim, eps=eps,
                                               batch_size=batch_size, device=f"cpu", download=download)
        self.y_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=dim, eps=eps,
                                                    batch_size=batch_size, device=f"cpu", download=download)
        self.gt_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps, 
                                                                        batch_size=batch_size,  device=f"cpu", download=download)
    
    def x_sample(self, batch_size):
        return self.x_sampler.sample(batch_size)
    
    def y_sample(self, batch_size):
        return self.y_sampler.sample(batch_size)
    
    def gt_sample(self, batch_size):
        return self.gt_sampler.sample(batch_size)
    
    def conditional_y_sample(self, x):
        return self.gt_sampler.conditional_plan.sample(x)
    
    def gt_sde_path_sampler(self):
        
        mus = self.y_sampler.conditional_plan.potential_mus
        probs = self.y_sampler.conditional_plan.potential_probs
        sigmas = self.y_sampler.conditional_plan.potential_sigmas
        potential_params = (probs, mus, sigmas)

        n_em_steps = 99
        gt_sde = GroundTruthSDE(potential_params, self.eps, n_em_steps)
        
        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t
        
        return return_fn
    
    def brownian_bridge_sampler(self):
        
        def return_fn(batch_size):
            x_0 = self.x_sample(batch_size)
            
            x_1 = self.gt_sampler.conditional_plan.sample(x_0)
            t_0 = 0
            t_1 = 1
            n_timesteps = 100
            
            t = torch.arange(n_timesteps).reshape([-1, 1]).repeat([1, batch_size]).transpose(0, 1) / n_timesteps
            
            x_0 = x_0.unsqueeze(1).repeat([1, n_timesteps, 1])
            x_1 = x_1.unsqueeze(1).repeat([1, n_timesteps, 1])
            
            # N x T x D
                        
            mean = x_0 + ((t - t_0) / (t_1 - t_0)).reshape([x_0.shape[0], x_0.shape[1], 1]) * (x_1 - x_0)
            
            std = torch.sqrt(self.eps * (t - t_0) * (t_1 - t) / (t_1 - t_0))
            
            x_t = mean + std.reshape([std.shape[0], std.shape[1], 1]) * torch.randn_like(mean)

            return x_t, t
                    
        return return_fn


    def path_sampler(self):
        mus = self.y_sampler.conditional_plan.potential_mus
        probs = self.y_sampler.conditional_plan.potential_probs
        sigmas = self.y_sampler.conditional_plan.potential_sigmas
        potential_params = (probs, mus, sigmas)

        n_em_steps = 99
        gt_sde = GroundTruthSDE(potential_params, self.eps, n_em_steps)

        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t

        return return_fn
    
    def __str__(self) -> str:
        return f'EOTSampler_D_{self.dim}_eps_{self.eps}'


class ImageEOTBenchmark(Sampler):
    
    def __init__(self, eps, batch_size=64, glow_device='cpu', download=False) -> None:
        super().__init__()
        eps = eps if int(eps) < 1 else int(eps)

        benchmark = ImageBenchmark(batch_size=batch_size, eps=eps, glow_device=glow_device,
                                   samples_device=glow_device, download=download, num_workers=0)
        
        self.x_sampler = benchmark.X_sampler
        self.y_sampler = benchmark.Y_sampler
        self.gt_sampler = benchmark.GT_sampler
        self.device = glow_device

        self.eps = eps
        
    def x_sample(self, batch_size):
        return self.x_sampler.sample(batch_size)
    
    def y_sample(self, batch_size):
        return self.y_sampler.sample(batch_size)
    
    def gt_sample(self, batch_size):
        return self.gt_sampler.sample(batch_size)

    def path_sampler(self):
        
        mus = self.y_sampler.conditional_plan.potential_mus
        probs = self.y_sampler.conditional_plan.potential_probs
        sigmas = self.y_sampler.conditional_plan.potential_sigmas
        potential_params = (probs, mus.reshape([probs.shape[0], -1]), sigmas.reshape([probs.shape[0], -1]))
        n_em_steps = 99
        
        gt_sde = GroundTruthSDE(potential_params, self.eps, n_em_steps, is_diagonal=True)

        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t
        
        return return_fn


def build_mixture_samples(dim, sigma, batch_size):
    eps = sigma if int(sigma) < 1 else int(sigma)
    X_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=dim, eps=eps,
                                               batch_size=batch_size, device=f"cpu", download=False)
    Y_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=dim, eps=eps,
                                              batch_size=batch_size, device=f"cpu", download=False)
    gt_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps, 
                                                                    batch_size=batch_size,  device=f"cpu", download=False)
    
    return Y_sampler, X_sampler, gt_sampler


class IsotropicGaussianSampler(Sampler):
    def __init__(self, mean_1, sigma_1, mean_2, sigma_2) -> None:
        super().__init__()
        self.dist_x = torch.distributions.multivariate_normal.MultivariateNormal(mean_1, covariance_matrix=torch.eye(mean_1.shape[0]) * sigma_1 ** 2)
        self.dist_y = torch.distributions.multivariate_normal.MultivariateNormal(mean_2, covariance_matrix=torch.eye(mean_2.shape[0]) * sigma_2 ** 2)

    def x_sample(self, batch_size):
        return self.dist_x.sample([batch_size])
    
    def y_sample(self, batch_size):
        return self.dist_y.sample([batch_size])


class ReverseSampler(Sampler):
    def __init__(self, sampler: Sampler) -> None:
        super().__init__()
        self.sampler_to_reverse = sampler
        self.dim = sampler.dim

    def x_sample(self, batch_size):
        return self.sampler_to_reverse.y_sample(batch_size)

    def y_sample(self, batch_size):
        return self.sampler_to_reverse.x_sample(batch_size)

    def __str__(self) -> str:
        return 'Reverse_' + str(self.sampler_to_reverse)


class LSBSampler(Sampler):
    
    def __init__(self, lsb_model, x_0_sampler, device='cpu') -> None:
        self.lsb_model = lsb_model
        self.x_0_sampler = x_0_sampler
        self.eps = lsb_model.lsb_model.epsilon
        self.device = device
        self.lsb_model.to(device)
    
    @torch.no_grad()
    def x_sample(self, batch_size):
        return self.x_0_sampler(batch_size)
    
    @torch.no_grad()
    def y_sample(self, batch_size):
        x_0_samples = self.x_sample(batch_size).to(self.device)
        return self.lsb_model(x_0_samples)
        
    def gt_sample(self, batch_size):
        raise NotImplementedError('GT sampler is not available')
        
    def sample_conditional(self, x_0):
        return self.lsb_model(x_0)

    def __call__(self, x) -> Any:
        return self.sample_conditional(x)
    
    @torch.no_grad()
    def path_sampler(self):
        
        n_em_steps = 99
        eps_fn = lambda t: self.eps
        
        gt_sde = GTSDEByDrift(self.lsb_model.get_drift(), eps_fn, n_em_steps)

        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size).to(self.device)
#             x_t, t = gt_sde.sample(x_samples)
#             t = t.transpose(0, 1)
#             print(x_t.shape, t.shape)
            x_t, t = self.lsb_model.sample_path_bb(x_samples)
            
            return x_t, t
        
        return return_fn


class VPVEHybridSDESampler(Sampler):
    def __init__(self, x_sampler, dim, sigma_max=1):
        # DDPM'20 J.Ho betas
        self.beta_min = 0.1
        self.beta_max = 20
        self.beta_sch = lambda t: self.beta_min + t * (self.beta_max - self.beta_min)
        
        self.dim = dim
        self.x_sampler = x_sampler
        
        self.sigma_min = 10e-5
        self.sigma_max = sigma_max
#         self.sigma_max = torch.sqrt(torch.tensor(1 / 2))
        
    @torch.no_grad()
    def x_sample(self, batch_size):
        return self.x_sampler(batch_size)
    
    @torch.no_grad()
    def y_sample(self, batch_size):
        return MultivariateNormal(torch.zeros([self.dim]), torch.eye(self.dim)).sample([batch_size]) * self.sigma_max ** 2
        
    def gt_sample(self, batch_size):
        raise NotImplementedError('GT sampler is not available')
        
    def sample_conditional(self, x_0):
        # Multivariate normal
        return MultivariateNormal(torch.zeros([self.dim]), torch.eye(self.dim)).sample([x_0.shape[0]]) * self.sigma_max ** 2
    
    def get_drift(self):
        return lambda x, t: - 1 / 2 * self.beta_sch(t) * x
    
    def eps_fn(self, t):
        return (self.sigma_min * (self.sigma_max / self.sigma_min) ** t * torch.sqrt(2 * torch.log(torch.tensor(self.sigma_max / self.sigma_min)))) ** 2
        
    @torch.no_grad()
    def path_sampler(self):
        
        n_em_steps = 99

        
        gt_sde = GTSDEByDrift(self.get_drift(), self.eps_fn, n_em_steps)
        
        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t
        
        return return_fn

class VPSDESampler(Sampler):
    def __init__(self, x_sampler, dim):
        # DDPM'20 J.Ho betas
        self.beta_min = 0.1
        self.beta_max = 20
        self.beta_sch = lambda t: self.beta_min + t * (self.beta_max - self.beta_min)
        
        self.dim = dim
        self.x_sampler = x_sampler
        self.device = 'cpu'
        
    # change for slicing
    @torch.no_grad()
    def x_sample(self, batch_size):
        return self.x_sampler(batch_size)
    
    # change for slicing
    @torch.no_grad()
    def y_sample(self, batch_size):
        return self.sample_t(self.x_sampler(batch_size), torch.ones([batch_size]))
        # return torch.distributions.normal.Normal(torch.zeros(self.dim), 1, validate_args=None).sample([batch_size])
    
    def gt_sample(self, batch_size):
        raise NotImplementedError('GT sampler is not available')
        
    def sample_conditional(self, x_0):
        # Multivariate normal
        return self.y_sample(batch_size=x_0.shape[0])
        # return torch.distributions.normal.Normal(torch.zeros(self.dim), 1, validate_args=None).sample([x_0.shape[0]])
    
    def get_drift(self):
        return lambda x, t: - 1 / 2 * self.beta_sch(t) * x
    
    def eps_fn(self, t):
        return torch.sqrt(self.beta_sch(t))
    
    @torch.no_grad()
    def path_sampler(self):
        
        n_em_steps = 99

        eps_fn = lambda t: self.eps_fn(t) ** 2
        
        gt_sde = GTSDEByDrift(self.get_drift(), eps_fn, n_em_steps)
        
        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size).cpu()
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t
        
        return return_fn
    
    def get_normal_dist_score(self, x, mean, std):
        return - (x - mean) / (std ** 2)
    
    def _marginal(self, x_0, t):
        mean = x_0 * torch.exp( - 1 / 2 * (self.sch_intergral(t))).unsqueeze(-1)
        std = torch.sqrt(1 - torch.exp(-self.sch_intergral(t)) + 1e-8).unsqueeze(-1)

        return mean, std
    
    def sample_t(self, x_0, t):
        mean, std = self._marginal(x_0, t)
        return mean + std * torch.randn_like(x_0)

    def score(self, x_0, x_t, t):
        mean, std = self._marginal(x_0, t)
        score = self.get_normal_dist_score(x_t, mean, std)
        return score
        
    def sch_intergral(self, t):
        return self.beta_min * t + t ** 2 / 2 * (self.beta_max - self.beta_min)

    # @torch.no_grad()
    # def sample_t(self, x_or_batch_size, t):
    #     if isinstance(x_or_batch_size, torch.Tensor):
    #         # sample starting form x
    #         x = x_or_batch_size
    #         batch_size = x.shape[0]
    #     else:
    #         batch_size = x_or_batch_size
    #         x = self.x_sample(batch_size)

    #     mean, std = self.marginal(x, t)
    #     marginal_dist = torch.distributions.normal.Normal(mean, std)
    #     return marginal_dist.sample()


class ChainSampler(Sampler):
    def __init__(self, list_of_samplers: List[Sampler]) -> None:
        super().__init__()
        self.samplers = list_of_samplers
    
    @torch.no_grad()
    def x_sample(self, batch_size):
        return self.samplers[0].x_sample(batch_size)
    
    @torch.no_grad()
    def y_sample(self, batch_size):
        return self.samplers[-1].y_sample(batch_size)

    def __call__(self, x) -> Any:
        for smp in self.samplers:
            x = smp(x)
        return x

    def path_sampler(self):
        return super().path_sampler()


class EOTGaussianSample(Sampler):
    def __init__(self, sampler_x, sampler_y, eps, fit_batch_size_x=1000, fit_batch_size_y=1000) -> None:
        super().__init__()
        self.sampler_x = sampler_x
        self.sampler_y = sampler_y
        self.eps = eps
        self.fit_batch_size_x = fit_batch_size_x
        self.fit_batch_size_y = fit_batch_size_y
        self.fit()

    def fit(self):
        # batch_size = 10000
        x_samples = self.sampler_x(self.fit_batch_size_x).cpu()
        y_samples = self.sampler_y(self.fit_batch_size_y).cpu()

        self.mu_1 = torch.mean(x_samples, dim=0)
        self.mu_2 = torch.zeros_like(self.mu_1)
        
        # self.mu_2 = torch.mean(y_samples, dim=0)
        
        self.cov_1 = ((x_samples - self.mu_1).unsqueeze(-1) @ (x_samples - self.mu_1).unsqueeze(-2)).mean(0)
        # self.cov_2 = ((y_samples - self.mu_2).unsqueeze(-1) @ (y_samples - self.mu_2).unsqueeze(-2)).mean(0)
        self.cov_2 = torch.eye(self.cov_1.shape[0]) * (self.eps ** 2)

        self.cov_1_sqrt = torch.Tensor(sqrtm(self.cov_1))
        self.cov_1_inv = torch.linalg.inv(self.cov_1)
        self.cov_2_inv = torch.linalg.inv(self.cov_2)
        
        self.cov_1_inv_sqrt = torch.Tensor(sqrtm(self.cov_1_inv))
        
        self.d_sigma = torch.Tensor(sqrtm((4 * self.cov_1_sqrt @ self.cov_2 @ self.cov_1_sqrt + self.eps ** 4 * torch.eye(self.cov_2.shape[0]))))
        
        self.c_sigma = 1 / 2 * (self.cov_1_sqrt @ self.d_sigma @ self.cov_1_inv_sqrt - self.eps ** 2 * torch.eye(self.cov_2.shape[0]))
        
        # mu = self.mu_2  + ((self.c_sigma.T @ self.cov_1_inv).unsqueeze(0).repeat([x.shape[0], 1, 1]) @ (x - mu_1).unsqueeze(-1)).squeeze()
        # cov = cov_2 - self.c_sigma.T @ cov_1_inv @ self.c_sigma
        
    def forward(self, x, eps=1e-8):
        
        mu = self.mu_2  + ((self.c_sigma.T @ self.cov_1_inv).unsqueeze(0).repeat([x.shape[0], 1, 1]) @ (x - self.mu_1).unsqueeze(-1)).squeeze()
        cov = self.cov_2 - self.c_sigma.T @ self.cov_1_inv @ self.c_sigma
        cov = (cov.T + cov) / 2 + torch.eye(cov.shape[0]) * eps

        cond_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=cov)

        return cond_dist.sample()

    def backward(self, y, eps=1e-8):
        
        mu = self.mu_1  + ((self.c_sigma @ self.cov_2_inv).unsqueeze(0).repeat([y.shape[0], 1, 1]) @ (y - self.mu_2).unsqueeze(-1)).squeeze()
        cov = self.cov_1 - self.c_sigma @ self.cov_2_inv @ self.c_sigma.T
        cov = (cov.T + cov) / 2 + torch.eye(cov.shape[0]) * eps
        cond_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=cov)

        return cond_dist.sample()
    
    def x_sample(self, batch_size):
        return self.sampler_x(batch_size)
    
    def y_sample(self, batch_size):
        return self.sampler_y(batch_size)

    def reverse(self):
        stakan = self.forward
        self.forward = self.backward
        self.backward = stakan

        stakan = self.sampler_x
        self.sampler_x = self.sampler_y
        self.sampler_y = stakan

        # return EOTGaussianSample() # and do it in reverse
    
    def __call__(self, x) -> Any:
        return self.forward(x)



class EOTGaussianDiagonalSample(Sampler):
    def __init__(self, sampler_x, sampler_y, eps, fit_batch_size_x=1000, fit_batch_size_y=1000) -> None:
        super().__init__()
        self.sampler_x = sampler_x
        self.sampler_y = sampler_y
        self.eps = eps
        self.fit_batch_size_x = fit_batch_size_x
        self.fit_batch_size_y = fit_batch_size_y
        self.fit()

    def fit(self):
        # batch_size = 10000
        x_samples = self.sampler_x(self.fit_batch_size_x).cpu()
        y_samples = self.sampler_y(self.fit_batch_size_y).cpu()

        self.mu_1 = torch.mean(x_samples, dim=0)

        self.mu_2 = torch.zeros_like(self.mu_1)

        # self.mu_2 = torch.mean(y_samples, dim=0)
        
        # self.cov_1 = ((x_samples - self.mu_1).unsqueeze(-1) @ (x_samples - self.mu_1).unsqueeze(-2)).mean(0)
        self.cov_1 = torch.diagonal(((x_samples - self.mu_1).unsqueeze(-1) @ (x_samples - self.mu_1).unsqueeze(-2)).mean(0))
        # print(self.cov_1.shape)
        self.cov_1 = torch.diag(self.cov_1)
        # print(self.cov_1.shape)
        self.cov_2 = torch.eye(self.cov_1.shape[0]) * (self.eps ** 2)
        # self.cov_2 = ((y_samples - self.mu_2).unsqueeze(-1) @ (y_samples - self.mu_2).unsqueeze(-2)).mean(0)
        # print(torch.diag(self.cov_2))

        self.cov_1_sqrt = torch.Tensor(sqrtm(self.cov_1))
        self.cov_1_inv = torch.linalg.inv(self.cov_1)
        self.cov_2_inv = torch.linalg.inv(self.cov_2)
        
        self.cov_1_inv_sqrt = torch.Tensor(sqrtm(self.cov_1_inv))
        
        self.d_sigma = torch.Tensor(sqrtm((4 * self.cov_1_sqrt @ self.cov_2 @ self.cov_1_sqrt + self.eps ** 4 * torch.eye(self.cov_2.shape[0]))))
        
        self.c_sigma = 1 / 2 * (self.cov_1_sqrt @ self.d_sigma @ self.cov_1_inv_sqrt - self.eps ** 2 * torch.eye(self.cov_2.shape[0]))
        
        # mu = self.mu_2  + ((self.c_sigma.T @ self.cov_1_inv).unsqueeze(0).repeat([x.shape[0], 1, 1]) @ (x - mu_1).unsqueeze(-1)).squeeze()
        # cov = cov_2 - self.c_sigma.T @ cov_1_inv @ self.c_sigma
        
    def forward(self, x, eps=1e-8):
        
        mu = self.mu_2  + ((self.c_sigma.T @ self.cov_1_inv).unsqueeze(0).repeat([x.shape[0], 1, 1]) @ (x - self.mu_1).unsqueeze(-1)).squeeze()
        cov = self.cov_2 - self.c_sigma.T @ self.cov_1_inv @ self.c_sigma
        # cov = (cov.T + cov) / 2 + torch.eye([cov.shape[0]]) * eps
        # torch.set_printoptions(threshold=10000)  
        # print(cov)

        cond_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=cov)

        return cond_dist.sample()

    def backward(self, y, eps=1e-8):
        
        mu = self.mu_1  + ((self.c_sigma @ self.cov_2_inv).unsqueeze(0).repeat([y.shape[0], 1, 1]) @ (y - self.mu_2).unsqueeze(-1)).squeeze()
        cov = self.cov_1 - self.c_sigma @ self.cov_2_inv @ self.c_sigma.T
        # cov = (cov.T + cov) / 2 + torch.eye([cov.shape[0]]) * eps
        # torch.set_printoptions(threshold=10000)
        # print(cov)

        cond_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=cov)

        return cond_dist.sample()
    
    def x_sample(self, batch_size):
        return self.sampler_x(batch_size)
    
    def y_sample(self, batch_size):
        return self.sampler_y(batch_size)

    def reverse(self):
        stakan = self.forward
        self.forward = self.backward
        self.backward = stakan

        stakan = self.sampler_x
        self.sampler_x = self.sampler_y
        self.sampler_y = stakan

        # return EOTGaussianSample() # and do it in reverse
    
    def __call__(self, x) -> Any:
        return self.forward(x)


def get_task(task_name, dim, eps, config):
    eps_str = eps
    eps = float(eps_str.split('_')[-1])
    if  eps_str.split('_')[0] == 'triangle':
        eps = eps / 2
    if task_name == 'mixture':
        return EOTGMMSampler(dim, eps)
    elif task_name == 'gaussian':
        return IsotropicGaussianSampler(torch.zeros([dim]), eps, torch.ones([dim]) * 5, eps)
    elif task_name == 'colored_mnist':
        return ColoredMnistSampler(config.batch_size)
    elif task_name == 'mnist':
        return GaussianToMNIST(float(eps), config.batch_size)
    raise ValueError('Unknown task')

def triangle_sigma(t, max_sigma):
    return (0.5 - torch.abs(t - 0.5)) * max_sigma

def parabolic_sigma(t, max_sigma):
    return -4 * max_sigma * (t - 1 / 2) ** 2 + max_sigma

from functools import partial
def get_sigma_fn(sigma_type):
    try:
        sigma = float(sigma_type)
        sigma_fn = lambda t: torch.ones_like(t) * sigma
    except ValueError:
        if 'triangle' in sigma_type:
            max_sigma = float(sigma_type.split('_')[1])
            sigma_fn = partial(triangle_sigma, max_sigma=max_sigma)
        elif 'parabolic' in sigma_type:
            max_sigma = float(sigma_type.split('_')[1])
            sigma_fn = partial(parabolic_sigma, max_sigma=max_sigma)
        else:
            raise ValueError('Unknown type of sigma schedule')
    return sigma_fn
