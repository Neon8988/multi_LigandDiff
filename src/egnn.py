import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from src import utils
from typing import Callable, Union
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import zeros_
from src.gvp_model import GVPNetwork



class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented.")

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)





class Dynamics(nn.Module):
    def __init__(
            self, in_node_nf,n_dims,  ligand_node_nf, 
            hidden_nf=32, activation='silu', n_layers=2,attention=False,
            normalization_factor=100,  drop_rate=0.0,
            device='cpu',model='gvp_dynamics'
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.ligand_node_nf = ligand_node_nf+1#1 is for time embedding
        self.model = model

        self.ligand_site_embedding=DenseLayer(self.ligand_node_nf,hidden_nf,activation=activation)
        self.h_embedding=DenseLayer(in_node_nf,hidden_nf,activation=activation)
        in_node_nf = in_node_nf + self.ligand_node_nf
        self.h_embedding_out=DenseLayer(hidden_nf, in_node_nf,activation=None)
        
        
        self.model == 'gvp_dynamics'

        self.dynamics = GVPNetwork(
            in_dims=(hidden_nf*2, 0), # (scalar_features, vector_features)
        out_dims=(hidden_nf, 1),
        hidden_dims=(hidden_nf, hidden_nf//2),
        drop_rate=drop_rate,
        vector_gate=True,
        num_layers=n_layers,
        attention=attention,
        normalization_factor=normalization_factor,
        )
        
    


    def forward(self, t, xh, batch_seg, ligand_site):

        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)
        edge_index = radius_graph(x, r=1e+50, batch=batch_seg, loop=False,max_num_neighbors=100)#.to(self.device)

        # conditioning on time 
        if np.prod(t.size()) == 1:
            # t is the same for all elements in batch.
            h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
        else:
            # t is different over the batch dimension.
            h_time = t[batch_seg]  

        ligand_site_t=torch.cat([ligand_site,h_time],dim=-1)

        ligand_site_t=self.ligand_site_embedding(ligand_site_t) #(B*N, hidden_nf)
        h=self.h_embedding(h) #(B*N, hidden_nf)
        h=torch.cat([h,ligand_site_t],dim=-1)
    
        if self.model == 'gvp_dynamics':
            h_final, vel = self.dynamics(h,x, edge_index)
            h_final=self.h_embedding_out(h_final)
            vel=vel.squeeze()
            
        else:
            raise NotImplementedError

        # Slice off ligand_site size
        h_final = h_final[:, :-self.ligand_node_nf]
        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        return torch.cat([vel, h_final], dim=1)

    


