import torch
import torchvision
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os.path as osp
import wandb
import math
from copy import deepcopy

from src.models.bridge import Bridge
from src.samplers.samplers import Sampler

import ot
from cleanfid import fid


from eot_benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler, 
)

from eot_benchmark.metrics import compute_BW_UVP_by_gt_samples

def build_mixture_samples(dim, sigma, batch_size):
    eps = sigma if int(sigma) < 1 else int(sigma)
    X_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=dim, eps=eps,
                                               batch_size=batch_size, device=f"cpu", download=False)
    Y_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=dim, eps=eps,
                                              batch_size=batch_size, device=f"cpu", download=False)
    gt_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps, 
                                                                    batch_size=batch_size,  device=f"cpu", download=False)
    
    return Y_sampler, X_sampler, gt_sampler


def get_isotropic_gaussians(dim, mean_1, sigma_1, mean_2, sigma_2):
    if mean_2.shape[0] != mean_1.shape[0]:
        raise ValueError('Dim of means are not equal')
    dist_1 = torch.distributions.multivariate_normal.MultivariateNormal(mean_1, covariance_matrix=torch.eye(dim) * sigma_1 ** 2)
    dist_2 = torch.distributions.multivariate_normal.MultivariateNormal(mean_2, covariance_matrix=torch.eye(dim) * sigma_2 ** 2)

    return dist_1, dist_2

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def pca_plot(x_0_gt, x_1_gt, x_1_pred, n_plot, save_name='plot_pca_samples.png', is_wandb=True):

    x_0_gt, x_1_gt, x_1_pred = x_0_gt.cpu(), x_1_gt.cpu(), x_1_pred.cpu()
    fig,axes = plt.subplots(1, 3,figsize=(12,4),squeeze=True,sharex=True,sharey=True)
    pca = PCA(n_components=2).fit(x_1_gt)
    
    x_0_gt_pca = pca.transform(x_0_gt[:n_plot])
    x_1_gt_pca = pca.transform(x_1_gt[:n_plot])
    x_1_pred_pca = pca.transform(x_1_pred[:n_plot])
    
    axes[0].scatter(x_0_gt_pca[:,0], x_0_gt_pca[:,1], c="g", edgecolor = 'black',
                    label = r'$x\sim P_0(x)$', s =30)
    axes[1].scatter(x_1_gt_pca[:,0], x_1_gt_pca[:,1], c="orange", edgecolor = 'black',
                    label = r'$x\sim P_1(x)$', s =30)
    axes[2].scatter(x_1_pred_pca[:,0], x_1_pred_pca[:,1], c="yellow", edgecolor = 'black',
                    label = r'$x\sim T(x)$', s =30)
    
    for i in range(3):
        axes[i].grid()
        axes[i].set_xlim([-5, 5])
        axes[i].set_ylim([-5, 5])
        axes[i].legend()
    
    fig.tight_layout(pad=0.5)
    im = fig2img(fig)
    im.save(save_name)

    if is_wandb:
        wandb.log({f'Plot PCA samples' : [wandb.Image(fig2img(fig))]})

def save_pics_grid(pics, n_rows, name, is_wandb=False):
    if is_wandb:
        tmp_name = 'tmp_torch_pic_save.png'
    else:
        tmp_name = name + '.png'
    torchvision.utils.save_image(torchvision.utils.make_grid(pics, nrow=n_rows), fp=tmp_name)

    if is_wandb:
        im = Image.open(tmp_name)
        wandb.log({name: wandb.Image(im)})


def draw_pics(x_gt, y_pred, save_name='pics_ot.png'):
    # x_1 = model.sample(x_0)
    x_1 = y_pred
    x_0 = x_gt
    n_pics = 8
    x_0 = x_0[:n_pics]
    x_1 = x_1[:n_pics]
    x_0, x_1 = x_0.permute([1, 0, 2, 3]), x_1.permute([1, 0, 2, 3])


    x_0 = torch.cat([x_0[:, i] for i in range(n_pics)], dim=1) 
    x_1 = torch.cat([x_1[:, i] for i in range(n_pics)], dim=1)
    x_0 = x_0 * 0.5 + 0.5
    x_1 = x_1 * 0.5 + 0.5

    x_0, x_1 = x_0.permute([1, 2, 0]), x_1.permute([1, 2, 0])

    out = torch.cat([x_0, x_1], dim=1)

    # if gt is not None:
    #     gt = gt[:n_pics]
    #     gt = gt.permute([1, 0, 2, 3])
    #     gt = torch.cat([gt[:, i] for i in range(n_pics)], dim=1) * 0.5 + 0.5
    #     gt = gt.squeeze().permute([1, 2, 0])
    #     out = torch.cat([out, gt], dim=1)
    
    out = out.cpu().numpy()

    # plt.rcParams["figure.figsize"] = [5, 5]

    if x_0.shape[-1] == 1:
        # out[:100] = (out[:100] - 0.5) / 0.5
        plt.imshow(out, cmap='gray')
    else:
        plt.imshow(out)
    plt.savefig(osp.join('pics', save_name), dpi = 1000)


def inverse_to_uint8_pic(images):
    return torch.tensor((images * 0.5 + 0.5) * 255, dtype=torch.uint8)

from torchmetrics.image.fid import FrechetInceptionDistance
from copy import deepcopy

class FIDCalculator:
    def __init__(self, true_sampler, batch_size=4096, device='cpu'):
        print('FID: calculation for True samples')
        self.fid = FrechetInceptionDistance(feature=64)
        self.true_samples = true_sampler(batch_size)
        self.batch_size = batch_size
        self.fid.update(inverse_to_uint8_pic(self.true_samples), real=True)
        self.device = device

    def calculate(self, false_samples):
        fid_copy = deepcopy(self.fid)
        fid_copy.to(self.device)
        fid_copy.update(inverse_to_uint8_pic(false_samples.to(self.device)), real=False)
        return fid_copy.compute()

def create_bridge_checkpoint(model: Bridge, sampler: Sampler, ckpt_name_prefix=''):
    ckpt_name = ckpt_name_prefix + str(model) + '_' + str(sampler)
    torch.save(model.state_dict(), ckpt_name + ".pth")
    
def from_vector_to_pic(vec):
    return vec.reshape([vec.shape[0], 1, int(math.sqrt(vec.shape[-1])), int(math.sqrt(vec.shape[-1]))])

def compute_BW(process, x_0_sampler, x_1_sampler, batch_size=51200):

    x_0_samples = x_0_sampler(batch_size)
    x_1_samples = x_1_sampler(batch_size)
    
    x_1_model_samples = process(x_0_samples)
    
    bw_uvp_x_1 = compute_BW_UVP_by_gt_samples(x_1_model_samples, x_1_samples)
    
    print(f'BW_UVP: {bw_uvp_x_1}')

    return bw_uvp_x_1

def eval_sampler(sampler_x, sampler_y, cond_sampler_to_test, batch_size=5120, device='cpu'):
    
    x_samples = sampler_x(batch_size)
    
    y_samples = sampler_y(batch_size).cpu()
    
    # device = cond_sampler_to_test.device
    with torch.no_grad():
        y_model_samples = cond_sampler_to_test(x_samples.to(device)).cpu()
    
    bw_uvp_y = compute_BW_UVP_by_gt_samples(y_model_samples, y_samples)
    
    print(f'BW_UVP: {bw_uvp_y}')

    return bw_uvp_y

class EMA:
    def __init__(self, model, decay, warmup=5000) -> None:
        pass
        self.model = model
        self.ema_state_dict = deepcopy(self.model.state_dict())
        self.decay = decay
        self.warmup = warmup
        self.model_device = next(iter(model.parameters())).device
        self.model_copy = deepcopy(model)
        self.model_copy.cpu()

    def update(self):
        self.warmup = max(self.warmup - 1, 0)
        if self.warmup == 0:
            # update model.state_dict[ema]
            source_dict = self.model.state_dict()
            target_dict = self.ema_state_dict
            for key in source_dict.keys():
                target_dict[key].data.copy_(
                    target_dict[key].data * self.decay + source_dict[key].data * (1 - self.decay)
                )

    def state_dict(self):
        return self.ema_state_dict
    
    def ema_model(self):
        self.model_copy.load_state_dict(self.ema_state_dict)
        return self.model_copy.to(self.model_device)

def get_reshape_to_pic_fn(samples):
    if samples.shape[-1] % 3 == 0:
        size = int(math.sqrt(samples.shape[-1] // 3))
    else:
        size = int(math.sqrt(samples.shape[-1]))

    reshape_to_pic_fn = lambda x: x.reshape([x.shape[0], -1, size, size])
    return reshape_to_pic_fn


def compute_fid(model, x_0_sampler, dataset_name='cifar10', batch_size=1024, num_gen=10000):
    reshape_to_pic_fn = get_reshape_to_pic_fn(x_0_sampler(10))
    
    def gen_batch(unused_latent):
        device = next(iter(model.vector_net.parameters())).device
        # print(device)
        
        with torch.no_grad():
            x = x_0_sampler(batch_size).to(device)
            x_1 = reshape_to_pic_fn(model.sample(x)).cpu()
        
        # x_1 = torch.clone(x_1)
        # x_1 = torch.randn([batch_size, 3, 32, 32])
        # print(x_1.shape)
        
        img = (x_1 * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
        
        return img
    
    print("Start computing FID")
    score = fid.compute_fid(
        gen=gen_batch,
        dataset_name=dataset_name,
        batch_size=batch_size,
        dataset_res=32,
        num_gen=num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
    )
    print(f'FID: {score}')
    return score

def energy_distance(x, y):

    Kxx = torch.norm((x.unsqueeze(0).repeat([x.shape[0], 1, 1]).transpose(0, 1) - x.unsqueeze(0).repeat([x.shape[0], 1, 1])), dim=-1)
    Kyy = torch.norm((y.unsqueeze(0).repeat([y.shape[0], 1, 1]).transpose(0, 1) - y.unsqueeze(0).repeat([y.shape[0], 1, 1])), dim=-1)
    Kxy = torch.norm((x.unsqueeze(0).repeat([x.shape[0], 1, 1]).transpose(0, 1) - y.unsqueeze(0).repeat([y.shape[0], 1, 1])), dim=-1)
    
    m = x.shape[0]
    n = y.shape[0]

    c1 = 1 / ( m * (m - 1))
    A = torch.sum(Kxx - torch.diag(torch.diagonal(Kxx)))

    # Term II
    c2 = 1 / (n * (n - 1))
    B = torch.sum(Kyy - torch.diag(torch.diagonal(Kyy)))

    # Term III
    c3 = 1 / (m * n)
    C = torch.sum(Kxy)

    # estimate MMD
    mmd_est = -0.5*c1*A - 0.5*c2*B + c3*C

    return mmd_est


import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def energy_distance_np(x, y):
    Kxx = pairwise_distances(x, x)
    Kyy = pairwise_distances(y, y)
    Kxy = pairwise_distances(x, y)

    m = x.shape[0]
    n = y.shape[0]

    c1 = 1 / ( m * (m - 1))
    A = np.sum(Kxx - np.diag(np.diagonal(Kxx)))

    # Term II
    c2 = 1 / (n * (n - 1))
    B = np.sum(Kyy - np.diag(np.diagonal(Kyy)))

    # Term III
    c3 = 1 / (m * n)
    C = np.sum(Kxy)

    # estimate MMD
    mmd_est = -0.5*c1*A - 0.5*c2*B + c3*C

    return mmd_est


def compute_w2(x, y):
    M = ot.dist(x, y)
    W_2 = ot.emd2(torch.ones([x.shape[0]]) / x.shape[0], torch.ones([y.shape[0]]) / y.shape[0], M , numItermax=300000)
    return W_2

from torch import nn

from torch.nn.functional import softmax, log_softmax
import torch
import geotorch

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

def lsb_components_stats(self, x):
    S = self.get_S()
    r = self.get_r()
    epsilon = self.epsilon

    log_alpha = self.log_alpha
    eps_S = epsilon*S

    samples = []
    batch_size = x.shape[0]
    sampling_batch_size = self.sampling_batch_size
    sampling_batch_size = 512
    
    num_sampling_iterations = (
        batch_size//sampling_batch_size if batch_size % sampling_batch_size == 0 else (batch_size//sampling_batch_size) + 1
    )
    
    for i in range(num_sampling_iterations):
        sub_batch_x = x[sampling_batch_size*i:sampling_batch_size*(i+1)]

        if self.is_diagonal:
            x_S_x = (sub_batch_x[:, None, :]*S[None, :, :]*sub_batch_x[:, None, :]).sum(dim=-1)
            x_r = (sub_batch_x[:, None, :]*r[None, :, :]).sum(dim=-1)
            r_x = r[None, :, :] + S[None, :]*sub_batch_x[:, None, :]
        else:
            x_S_x = (sub_batch_x[:, None, None, :]@(S[None, :, :, :]@sub_batch_x[:, None, :, None]))[:, :, 0, 0]
            x_r = (sub_batch_x[:, None, :]*r[None, :, :]).sum(dim=-1)
            r_x = r[None, :, :] + (S[None, :, : , :]@sub_batch_x[:, None, :, None])[:, :, :, 0]

        exp_argument = (x_S_x + 2*x_r)/(2*epsilon) + log_alpha[None, :]

        probs_mixture = torch.softmax(exp_argument, dim=-1).detach().cpu().numpy()[0]

#         print('Probabilities and indices of max probability component for data sample: ', torch.max(torch.softmax(exp_argument, dim=-1).detach().cpu(), dim=-1))
        n_unique_comp = torch.max(torch.softmax(exp_argument, dim=-1).detach().cpu(), dim=-1)[1].unique().shape[0]
        print('n_unique_comp: ', n_unique_comp)
#         print('Max, median, min: ', probs_mixture.max(), torch.median(torch.tensor(probs_mixture)), probs_mixture.min())
#         print(probs_mixture)
    return n_unique_comp

def cond_sample_in_chunks(x_0, cond_sampler, n_chunks):
    n_samples = x_0.shape[0]
    batch_size = n_samples // n_chunks
    res = []
    
    for i in range(n_chunks):
        x_0_chunk = x_0[i * batch_size:(i + 1) * batch_size]
        # print(f'Chunk: {i}')
        x_1_chunk = cond_sampler(x_0_chunk).detach()
        
        res.append(x_1_chunk)
    return torch.cat(res, dim=0)

def save_pic_vectors(samples_list, n_pics_to_plot=20, save_name='SomeName'):
    samples_0 = samples_list[0]
    reshape_to_pic_fn = get_reshape_to_pic_fn(samples_0)

    for i in range(len(samples_list)):
        samples_list[i] = reshape_to_pic_fn(samples_list[i])[:n_pics_to_plot]
    
    to_plot = torch.cat(samples_list, dim=0)

    save_pics_grid(torch.clamp(to_plot * 0.5 + 0.5, 0, 1), n_rows=n_pics_to_plot, name=save_name, is_wandb=wandb.run)
