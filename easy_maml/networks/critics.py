import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from typing import Optional
import os

import numpy as np
import torch
from torch import distributions
from collections import OrderedDict
from torch.func import functional_call

from easy_maml.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        critics_Ckp: Optional[str],
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        if critics_Ckp != None:
            if os.path.isfile(critics_Ckp):
                self.network.load_state_dict(torch.load(critics_Ckp))
        self.critics_Ckp = critics_Ckp

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
        

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        predicted_q_values = self.forward(obs)
        loss = F.mse_loss(predicted_q_values, q_values.view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Need to return a tensor and keep the gradients
        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }
    
    
    def checkpoint_save(self):
        if self.critics_Ckp != None:
            torch.save(self.network.state_dict(), self.critics_Ckp)


    ##################################
    ##   MAML specific functions    ##
    ##################################

    def maml_init(self,
                    baseline_learning_rate: float,
                    baseline_outer_lr: int,
                    learn_inner_lr: bool=False,
                    MAML:bool=True,
                    **kwargs):
        
        # Create a copy of the original parameters in the network
        self.network_state_dict_ori = self.network.state_dict()
        for item in self.network_state_dict_ori.items():
            item[1].requires_grad= True
            ptu.to_device(item[1])
        parameters = self.network_state_dict_ori.values()

        # Each layer may have different inner learning rates
        self.inner_lrs = {
            k: ptu.to_device(torch.tensor(baseline_learning_rate, requires_grad=learn_inner_lr))
            for k in self.network_state_dict_ori.keys()
        }
        
        # Initialise the optimizer of the outer loop with the parameters and the learning rates
        self.optimizer = optim.Adam(
            itertools.chain(parameters, self.inner_lrs.values()),
            lr=baseline_outer_lr,
        )
        
        self.MAML= MAML
        # Change forward function pointer to the MAML dedicated one
        self.forward = self.forward_maml

    def clone(self):
        # clone the original parameters and create the inner loop parameters from it
        network_dict_copy = OrderedDict([(k,torch.clone(v)) for (k,v) in self.network_state_dict_ori.items()])
        self.parameters_save = network_dict_copy

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, loss):
        loss.backward()
        self.optimizer.step()

    def forward_maml(self, obs: torch.Tensor) -> torch.Tensor:
        # Forward pass on critic network using the inner loop parameters
        return functional_call(self.network, self.parameters_save ,obs)
    

    def inner_loop_update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # Obtain a loss
        predicted_q_values = self.forward(obs)
        loss = F.mse_loss(predicted_q_values, q_values.view(-1, 1))

        # Compute the gradients regarding the inner parameters and add it to the computation graph
        gradients = torch.autograd.grad(loss, self.parameters_save.values(), create_graph=True)

        # Update inner parameters
        self.parameters_save = OrderedDict([(k, v - self.inner_lrs[k] * g) for (k, v), g in zip(self.parameters_save.items(), gradients)])

        # Return the loss
        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }


    def outer_loop_update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # Obtain a loss
        predicted_q_values = self.forward(obs)
        loss = F.mse_loss(predicted_q_values, q_values.view(-1, 1))

        # Return the loss and its position in computation graph as a tensor
        return {
            "Baseline Loss": loss,
        }