from typing import Optional, Sequence
import numpy as np
import torch

from easy_maml.networks.policies import MLPPolicyPG
from easy_maml.networks.critics import ValueCritic
from easy_maml.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
        policy_Ckp: Optional[str],
        critics_Ckp: Optional[str],
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate, policy_Ckp
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate, critics_Ckp
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        # Initialise update pointers to tradditional gradient updates
        self.actor_update=self.actor.update
        if use_baseline is True:
            self.critic_update=self.critic.update


    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # Flatten the lists of arrays into single arrays.
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: update the PG actor/policy
        info = self.actor_update(obs=obs,actions=actions,advantages=advantages)

        # step 4: update the PG critic/baseline
        if self.critic is not None:
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic_update(obs, q_values)

            info.update(critic_info)
        return info
    
    def checkpoint_save(self):
        """Save current network."""
        self.actor.checkpoint_save()
        if self.critic is not None:
            self.critic.checkpoint_save()

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: compute sum of rewards
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: computes sum of rewards to go
            q_values = [self._discounted_reward_to_go(r) for r in rewards]
        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        # Compute the advantages
        if self.critic is None:
            advantages = q_values
        else:
            obs = ptu.from_numpy(obs)

            # Get value from critic
            values = ptu.to_numpy(self.critic(obs).squeeze().detach())
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                advantages = q_values - values
            else:
                # GAE
                batch_size = obs.shape[0]

                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    delta = rewards[i] + self.gamma * values[i + 1] * (1 - terminals[i]) - values[i]
                    advantages[i]=delta + self.gamma * self.gae_lambda * (1 - terminals[i])* advantages[i + 1]

                # Remove dummy advantage
                advantages = advantages[:-1]

        # Normalize advantage
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)


        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Efficiently computes the total discounted return for the entire sequence of rewards.
        Each index in the output list will contain the same total discounted return.
        """
        n = len(rewards)
        total_discounted_return = sum(self.gamma ** i * rewards[i] for i in range(n))
        return [total_discounted_return] * n

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Efficiently computes the discounted reward-to-go.
        The reverse iteration allows calculation in O(n) time.
        """
        n = len(rewards)
        discounted_rewards = np.zeros(n)
        running_add = 0
        for t in reversed(range(n)):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    



class MAMLAgent(PGAgent):
    def init_maml(self, **kwargs):
        """
        Init the MAML algorithm
        """
        # Initialise actor/critic saves
        self.actor.maml_init(**kwargs)
        if self.critic is not None:
            self.critic.maml_init(**kwargs)

        # Initialise update pointers to inner loop
        self.actor_update=self.actor.inner_loop_update
        if self.critic is not None:
            self.critic_update=self.critic.inner_loop_update

    def clone(self):
        """Clone network weights"""
        # Clone network weigths
        self.actor.clone()
        # Change update pointers to inner loop
        self.actor_update=self.actor.inner_loop_update
        if self.critic is not None:
            self.critic.clone()
            self.critic_update=self.critic.inner_loop_update

    def step(self, actor_outer_loss:torch.tensor,critic_outer_loss:torch.tensor=None):
        """Perform optimizer step and zero the gradients"""
        self.actor.step(actor_outer_loss)
        if self.critic is not None:
            self.critic.step(critic_outer_loss)
        self.actor.zero_grad()
        if self.critic is not None:
            self.critic.zero_grad()

    def maml_outer_loss(self) -> dict:
        """Prepare Outer loop update"""
        # Change update pointers to outer loop
        self.actor_update=self.actor.outer_loop_update
        if self.critic is not None:
            self.critic_update=self.critic.outer_loop_update

    def update_maml_info(self) -> dict:
        """Update info Dict with MAML info"""
        maml_info = {k:ptu.to_numpy(v) for (k,v) in self.actor.inner_lrs.items()}
        if self.critic is not None:
            maml_info.update({k:ptu.to_numpy(v) for (k,v) in self.critic.inner_lrs.items()})
        return maml_info
