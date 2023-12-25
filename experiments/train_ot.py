
import sys 
from argparse import ArgumentParser
from datetime import datetime
from nip import load, parse
import wandb
import torch
import math
from tqdm import tqdm

sys.path.append("..")

from src.models.neural_optimal_transport import EgNOTWithEntNOT

from src.utils import get_reshape_to_pic_fn, energy_distance_np, cond_sample_in_chunks, save_pics_grid

def eval_pic_sampler(cond_sampler, sampler_x, sampler_y, device='cuda:0', pic_name='Empty_pic_name', suffix='', n_pics_to_plot=20):

    reshape_to_pic_fn = get_reshape_to_pic_fn(sampler_x(10))

    x_samples = sampler_x(100).to(device)
    transport_x = cond_sampler(x_samples).detach().cpu()
    x_samples, transport_x = reshape_to_pic_fn(x_samples).cpu(), reshape_to_pic_fn(transport_x).cpu()
    
    if n_pics_to_plot is not None:
        to_plot = torch.cat([x_samples[:n_pics_to_plot], transport_x[:n_pics_to_plot]], dim=0)
        print(f'Saving pics..')
        save_pics_grid(torch.clamp(to_plot * 0.5 + 0.5, 0, 1), n_rows=n_pics_to_plot, name=pic_name, is_wandb=wandb.run)
    
    x_samples = sampler_x(20000).cpu()
    D_e_batch_size = 5000
    transport_x = cond_sample_in_chunks(sampler_x(D_e_batch_size).to(device), cond_sampler, n_chunks=10).cpu()
    
    D_e = energy_distance_np(transport_x, sampler_y(D_e_batch_size))
    
    print(f'Energy distance {suffix}:', D_e)
    
    if wandb.run:
        wandb.log({f'Energy Distance {suffix}': D_e})


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config')
    parser.add_argument('--max_iter', type=int, default=10000, help='Max number of iterations for training')
    parser.add_argument('--val_freq', type=int, default=250, help='Frequency of running validation')
    parser.add_argument('--save_freq', type=int, default=250, help='Frequency of saving checkpoints')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('--device', type=int, default=0, help='Choose the GPU')

    args = parser.parse_args()

    max_iter = args.max_iter
    val_freq = args.val_freq
    save_freq = args.save_freq
    batch_size = args.batch_size
    
    if args.device == 'cpu':
        device = 'cpu'
    else:
        device = f'cuda:{args.device}'

    config = load(args.config)

    sampler = config['sampler']
    sampler_x = sampler.x_sample
    sampler_y = sampler.y_sample

    model = config['ot_model']

    model.to(device)
    
    wandb.init(project="Backward_proc", name=f"OT_d_{sampler.dim}_eps_{model.eps}")

    eta = 3e-7

    for i in tqdm(range(1, max_iter + 1)):
        model.train_step(sampler_x, sampler_y, ula_steps=100, eta=eta, batch_size=batch_size)
        
        if i % val_freq == 0:
            print('Validation..')
            
            ula_steps_list = [0, 2, 5, 10, 20, 50, 100, 200, 500]
            
            for n_ula_steps in ula_steps_list:
                # print('ULA steps: ', n_ula_steps)
                egnot_cond_sampler_fn = lambda x: model.sample(x, ula_steps=n_ula_steps, eta=eta)
                
                eval_pic_sampler(egnot_cond_sampler_fn, sampler_x, sampler_y, device=device, suffix=f'ULA steps {n_ula_steps}', pic_name='Vanilla_EgNOT_pic_name', n_pics_to_plot=20)
        
        if i % save_freq == 0:
            print('Saving model..')
            torch.save(model.state_dict(), f'ckpts/OT_Sampler_ModelName_{datetime.now().strftime("%H:%M:%S-%d-%m")}_conf_{args.config[:-5]}.pth')
            if isinstance(model, EgNOTWithEntNOT):
                torch.save(model.entNOT.state_dict(), f'ckpts/NOT_Sampler_ModelName_{datetime.now().strftime("%H:%M:%S-%d-%m")}_conf_{args.config[:-5]}.pth')
                torch.save(model.egNOT.state_dict(), f'ckpts/EgNOT_Sampler_ModelName_{datetime.now().strftime("%H:%M:%S-%d-%m")}_conf_{args.config[:-5]}.pth')
