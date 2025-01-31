import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from typing import Optional
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import distributions
from torch.func import functional_call

from easy_maml.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        policy_Ckp: Optional[str],
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            if policy_Ckp != None:
                if os.path.isfile(policy_Ckp):
                    self.logits_net.load_state_dict(torch.load(policy_Ckp))

            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)

            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )

            if policy_Ckp != None:
                if os.path.isfile(policy_Ckp):
                    state_dict = torch.load(policy_Ckp)
                    self.logstd = nn.Parameter(state_dict.popitem(last=True)[1])
                    self.mean_net.load_state_dict(state_dict)

            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete
        self.policy_Ckp = policy_Ckp


        

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        
        obs = ptu.from_numpy(obs).to(ptu.device)

        distribution=self.forward(obs)
        action = distribution.sample()

        return action.cpu().numpy()

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """

        # Policy forward pass
        if self.discrete:
            logits = self.logits_net(obs)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)  
            std=torch.exp(self.logstd)
            return distributions.Normal(mean, std)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError

    def checkpoint_save(self):
        """Save Networks parameters"""

        if self.discrete:
            model_dict = self.logits_net.state_dict()
        else:
            model_dict = self.mean_net.state_dict()
            model_dict["logstd"] = self.logstd
        if self.policy_Ckp != None:
            torch.save(model_dict, self.policy_Ckp)

class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    )-> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        distribution = self.forward(obs)

        if self.discrete:
            log_prob = distribution.log_prob(actions)
        else:
            log_prob = distribution.log_prob(actions).sum(axis=-1)
        loss = -(log_prob * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
    
    ##################################
    ##   MAML specific functions    ##
    ##################################

    def maml_init(self,
                    learning_rate: float,
                    outer_lr: int,
                    learn_inner_lr: bool=False,
                    MAML:bool=True,
                    **kwargs):
        
        if self.discrete is True:
            # Create a copy of the original parameters in the network
            self.logits_net_state_dict_ori = self.logits_net.state_dict()
            for item in self.logits_net_state_dict_ori.items():
                item[1].requires_grad= True
                ptu.to_device(item[1])
            parameters = self.logits_net_state_dict_ori.values()

            # Each layer may have different inner learning rates
            self.inner_lrs = {
                k: ptu.to_device(torch.tensor(learning_rate, requires_grad=learn_inner_lr))
                for k in self.logits_net_state_dict_ori.keys()
            }
        else:
            # Create a copy of the original parameters in the network
            self.mean_net_state_dict_ori = self.mean_net.state_dict()
            for item in self.mean_net_state_dict_ori.items():
                item[1].requires_grad= True
                ptu.to_device(item[1])
            self.logstd_ori = self.logstd
            del self.logstd
            parameters = itertools.chain([self.logstd_ori], self.mean_net_state_dict_ori.values())

            # Each layer may have different inner learning rates
            self.inner_lrs = {
                k: ptu.to_device(torch.tensor(learning_rate, requires_grad=learn_inner_lr))
                for k in self.mean_net_state_dict_ori.keys()
            }
            self.inner_lrs["logstd"]= ptu.to_device(torch.tensor(learning_rate, requires_grad=learn_inner_lr))


        self.optimizer = optim.Adam(
            itertools.chain(parameters, self.inner_lrs.values()),
            lr=outer_lr,
        )
        
        self.MAML= MAML
        # Change forward function pointer to the MAML dedicated one
        self.forward = self.forward_maml
        
    def clone(self):
        """Clone the policy original parameters as inner loop parameters."""
        if self.discrete:
            logits_net_bis_dict = OrderedDict([(k,torch.clone(v)) for (k,v) in self.logits_net_state_dict_ori.items()])
            self.parameters_save = logits_net_bis_dict
        else:
            mean_net_bis_dict = OrderedDict([(k,torch.clone(v)) for (k,v) in self.mean_net_state_dict_ori.items()])
            self.logstd = torch.clone(self.logstd_ori)
            self.parameters_save = mean_net_bis_dict
            self.parameters_save["logstd"] = self.logstd

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, loss):
        loss.backward()
        self.optimizer.step()

    def forward_maml(self, obs: torch.FloatTensor):
        """Redefinition of the forward pass for MAML purposes."""

        # Forward pass on actor network using the inner loop parameters
        if self.discrete:
            logits = functional_call(self.logits_net, self.parameters_save ,obs)
            return distributions.Categorical(logits=logits)
        else:
            log_std = self.parameters_save.popitem(last=True)[1]
            mean = functional_call(self.mean_net, self.parameters_save ,obs)
            self.parameters_save["logstd"]=log_std
            std=torch.exp(log_std)
            return distributions.Normal(mean, std)
        
    def inner_loop_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the inner loop policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # Obtain a loss
        distribution = self.forward(obs)
        if self.discrete:
            log_prob = distribution.log_prob(actions)
        else:
            log_prob = distribution.log_prob(actions).sum(axis=-1)
        loss = -(log_prob * advantages).mean()

        # Compute the gradients regarding the inner parameters and add it to the computation graph
        gradients = torch.autograd.grad(loss, self.parameters_save.values(), create_graph=True)

        # Update inner parameters
        self.parameters_save = OrderedDict([(k, v - self.inner_lrs[k] * g) for (k, v), g in zip(self.parameters_save.items(), gradients)])

        # Return the loss 
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
    

    def outer_loop_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the outer loop policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # Obtain a loss
        distribution = self.forward(obs)
        if self.discrete:
            log_prob = distribution.log_prob(actions)
        else:
            log_prob = distribution.log_prob(actions).sum(axis=-1)
        loss = -(log_prob * advantages).mean()

        # Return the loss and its position in computation graph as a tensor
        return {
            "Actor Loss": loss,
        }
