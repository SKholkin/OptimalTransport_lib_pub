from torch import nn 
import torch

class MLP_v0(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Add positional embedding for time
        self.dim = dim

        layer_dims = [dim + 1, 512, 512, 512, dim]

        modules = []

        for i in range(len(layer_dims) - 2):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(layer_dims[-2], dim))

        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        x = torch.cat([x, t.reshape([-1, 1])], dim=-1)
        return self.net(x)
    

import math
class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze()
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        # emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')

        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    
class MLP(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels, time_embed_dim, num_hidden_blocks):
        super().__init__()
        self.to_time_embed = SinusoidalPosEmb(time_embed_dim)
        self.in_layer = nn.Sequential(
            nn.Linear(in_channels + time_embed_dim, hidden_channels),
            nn.SiLU(),
        )
        
        self.hidden_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            ) for _ in range(num_hidden_blocks)
        ])

        self.out_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, time):
        time_embed = self.to_time_embed(time)
        x = self.in_layer(torch.cat([x, time_embed], dim=-1))
        for hidden_block in self.hidden_blocks:
            x = hidden_block(x) + x
        x = self.out_layer(x)

        return x
    
class MLP_v1(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_blocks):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
        )
        
        self.hidden_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            ) for _ in range(num_hidden_blocks)
        ])

        self.out_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x):
        x = self.in_layer(x)
        for hidden_block in self.hidden_blocks:
            x = hidden_block(x) + x
        x = self.out_layer(x)

        return x
    
class Variational_MLP_v1(MLP_v1):
    
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_blocks):
        super().__init__(in_channels, out_channels, hidden_channels, num_hidden_blocks)

        self.out_layer = nn.Linear(hidden_channels, out_channels * 2)
        
    def forward(self, x):
        x = self.in_layer(x)
        for hidden_block in self.hidden_blocks:
            x = hidden_block(x) + x
        mean, std = self.out_layer(x).chunk(2, dim=-1)

        return mean, std
    

        
