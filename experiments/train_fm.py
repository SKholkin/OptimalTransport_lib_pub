

import sys 
from argparse import ArgumentParser
from datetime import datetime
from nip import load, parse
import wandb
import torch
import math
from tqdm import tqdm

sys.path.append("..")

from src.utils import EMA
from src.utils import save_pic_vectors
from src.models.neural_optimal_transport import OTSampler

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config')
    parser.add_argument('--max_iter', type=int, default=10000, help='Max number of iterations for training')
    parser.add_argument('--sample_freq', type=int, default=250, help='Frequency of running validation')
    parser.add_argument('--save_freq', type=int, default=250, help='Frequency of saving checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
    parser.add_argument('--device', type=int, default=0, help='Choose the GPU')

    args = parser.parse_args()

    max_iter = args.max_iter
    sample_freq = args.sample_freq
    save_freq = args.save_freq
    batch_size = args.batch_size

    if args.device == 'cpu':
        device = 'cpu'
    else:
        device = f'cuda:{args.device}'

    config = load(args.config)

    # make vector net

    sampler = config['sampler']
    sampler_x = sampler.x_sample
    sampler_y = sampler.y_sample
    cond_sampler_y = sampler.cond_y_sample

    if isinstance(sampler, OTSampler):
        sampler.ot_model.to(device)

    fm_model = config['flow_matching']

    fm_model.to(device)

    opt = torch.optim.Adam(fm_model.parameters(), lr=2e-4)

    ema = EMA(fm_model, decay=0.9999, warmup=5000)
    
    wandb.init(project="Backward_proc", name=f"Reversing_ENOT_via_Flow_Matching_CIFAR")
    
    for i in tqdm(range(1, max_iter + 1)):
        
        x_1 = sampler_x(batch_size).to(device)
        
        x_0 = cond_sampler_y(x_1).to(device)

        loss = fm_model.step(x_0, x_1)
        
        opt.zero_grad()
        
        loss.backward()
        ema.update()

        if wandb.run:
            wandb.log({'loss': loss.item()})
        
        opt.step()
        
        if i % sample_freq == 0:
            n_pics_to_plot = 20
            
            x_gt = sampler_x(batch_size).to(device)
            y_gt = cond_sampler_y(x_gt)
            x_pred = fm_model.sample(y_gt, steps=100).detach().cpu()

            save_pic_vectors([x_gt.cpu(), y_gt.cpu(), x_pred.cpu()], n_pics_to_plot=20, save_name='Flow_Matching_reverse')
            
            # x_gt, y_gt, x_pred = reshape_to_pic_fn(x_gt).cpu(), reshape_to_pic_fn(y_gt).cpu(), reshape_to_pic_fn(x_pred).cpu()

            # to_plot = torch.cat([y_gt[:n_pics_to_plot], x_gt[:n_pics_to_plot], x_pred[:n_pics_to_plot]], dim=0)

            # save_pics_grid(to_plot * 0.5 + 0.5, n_rows=n_pics_to_plot, name=f"Flow_Matching_reverse", is_wandb=wandb.run)
            
            y_gt = torch.randn_like(sampler_x(batch_size), device=device)

            x_pred = fm_model.sample(y_gt, steps=100).detach()

            save_pic_vectors([x_gt.cpu(), y_gt.cpu(), x_pred.cpu()], n_pics_to_plot=20, save_name='Flow_Matching_reverse_gt_sampler')


            # y_gt, x_pred = reshape_to_pic_fn(y_gt).cpu(), reshape_to_pic_fn(x_pred).cpu()

            # to_plot = torch.cat([y_gt[:n_pics_to_plot], x_pred[:n_pics_to_plot]], dim=0)
            
            # save_pics_grid(to_plot * 0.5 + 0.5, n_rows=n_pics_to_plot, name=f"Flow_Matching_reverse_gt_sampler", is_wandb=wandb.run)

            # for EMA model
            ema_model = ema.ema_model()
            
            n_pics_to_plot = 20
            
            x_gt = sampler_x(batch_size).to(device)
            y_gt = cond_sampler_y(x_gt)
            x_pred = ema_model.sample(y_gt, steps=100).detach()

            save_pic_vectors([x_gt.cpu(), y_gt.cpu(), x_pred.cpu()], n_pics_to_plot=20, save_name='Flow_Matching_reverse_EMA')

            
            # x_gt, y_gt, x_pred = reshape_to_pic_fn(x_gt).cpu(), reshape_to_pic_fn(y_gt).cpu(), reshape_to_pic_fn(x_pred).cpu()

            # to_plot = torch.cat([y_gt[:n_pics_to_plot], x_gt[:n_pics_to_plot], x_pred[:n_pics_to_plot]], dim=0)

            # save_pics_grid(to_plot * 0.5 + 0.5, n_rows=n_pics_to_plot, name=f"Flow_Matching_reverse_EMA", is_wandb=wandb.run)
            
            y_gt = torch.randn_like(sampler_x(batch_size), device=device)

            x_pred = ema_model.sample(y_gt, steps=100).detach()

            save_pic_vectors([x_gt.cpu(), y_gt.cpu(), x_pred.cpu()], n_pics_to_plot=20, save_name='Flow_Matching_reverse_gt_sampler_EMA')

        if i % save_freq == 0:
            datetime_marker_str = datetime.now().strftime("%d:%m:%y_%H:%M:%S")
            print('Saving model..')
            torch.save(fm_model.state_dict(), f'ckpts/fm_model_{i}_{datetime_marker_str}.pth')
            torch.save(ema.state_dict(), f'ckpts/fm_model_{i}_ema_{datetime_marker_str}.pth')
            

