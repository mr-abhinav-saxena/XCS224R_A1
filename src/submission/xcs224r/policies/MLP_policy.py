"""
TO EDIT: Defines a pytorch policy as the agent's actor.

Functions to edit:
    1. get_action (line 111)
    2. forward (line 126)
    3. update (line 141)
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from ..infrastructure import pytorch_util as ptu
from ..policies.base_policy import BasePolicy


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    logits_na: nn.Sequential
        A neural network that outputs dicrete actions
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim: int,
                 ob_dim: int,
                 n_layers: int,
                 size: int,
                 learning_rate: float=1e-4,
                 training: bool=True,
                 nn_baseline: bool=False,
                 **kwargs: Any
                 ) -> None:
        super().__init__(**kwargs)

        # Initialize variables for environment (action/observation dimension, number of layers, etc.)
        self.ac_dim = ac_dim # action dimension
        self.ob_dim = ob_dim # observation dimension
        self.n_layers = n_layers # number of layers
        self.size = size # hidden layer size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # NOTE: This works for a continuous action space. All our environments use a continuous action space.
        self.logits_na = None
        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    ##################################

    def save(self, filepath: str) -> None:
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if len(obs.shape) > 1: # if obs is a batch of observations, do nothing
            observation = obs
        else:
            observation = obs[None] # if obs is a single observation, add a batch dimension

        # TODO return the action that the policy prescribes

        # *** START CODE HERE ***
        # 1. Convert the input numpy `observation` array to a PyTorch float tensor using `ptu.from_numpy()`.
        # 2. Query the policy network (self) by passing the observation: `self(observation)`
        #    to get the corresponding diagonal Gaussian distribution.
        # 3. Draw a sample from the distribution using `sample()`.
        # 4. Convert the sampled action tensor back to a numpy array using `ptu.to_numpy()` and return it.
        
        observation = ptu.from_numpy(observation)
        actions_distribution = self.forward(observation) # self(observation) is the same as self.forward(observation), but the latter is more explicit
        action = actions_distribution.sample()
        action = ptu.to_numpy(action)
        
        return action

        # *** END CODE HERE ***

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action distribution (torch.distributions.Distribution): a pytorch distribution representing
                a diagonal Gaussian distribution whose mean (loc) is computed by
                self.mean_net and standard deviation (scale) is torch.exp(self.logstd)

        Note:
            PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) a combination of torch.distributions.Normal
                             and torch.distributions.Independent

            Please consult the following documentation for further details on
            the use of probability distributions in Pytorch:
            https://pytorch.org/docs/stable/distributions.html
        """

        # *** START CODE HERE ***
        # 1. Compute the mean vector by feeding the observation through `self.mean_net(observation)`.
        # 2. Compute the diagonal standard deviations by exponentiating the learnable parameter: `torch.exp(self.logstd)`.
        # 3. OPTION 1: Using multivariate normal distribution functionality:
        #    Structure the diagonal covariance matrices:
        #    - Convert standard deviations to variances by squaring them: `batch_variance = batch_std ** 2`.
        #    - Form a standard diagonal matrix of variances: `scale_tril = torch.diag(...)`
        #    - Batch this diagonal matrix to match the input batch size: repeat the matrix along the batch
        #      dimension (e.g. using `scale_tril.repeat(batch_dim, 1, 1)` where batch_dim is `batch_mean.shape[0]`).
        #    - Instantiate and return a `distributions.MultivariateNormal` mapping loc to the batch mean and scale_tril
        #    to the batched diagonal matrices.
        #    OPTION 2: Using independent normal distributions functionality:
        #    - Instantiate and return a `distributions.Independent` distribution, where the base distribution is a `distributions.Normal` with loc set to the batch mean and scale set to the batch standard deviation. The `reinterpreted_batch_ndims=1` argument tells PyTorch to treat the last dimension of the Normal distribution's output as independent, which is necessary to create a diagonal Gaussian distribution.

        batch_mean = self.mean_net(observation)
        batch_std = torch.exp(self.logstd)

        ## OPTION 1: Using MultivariateNormal with scale_tril (lower triangular matrix)
        #batch_variance = batch_std ** 2
        #scale_tril = torch.diag(batch_variance) # creates a diagonal covariance matrix with variances on the diagonal
        #batch_dim = batch_mean.shape[0] # number of observations in the batch
        #scale_tril = scale_tril.repeat(batch_dim, 1, 1) # repeats the diagonal matrix along the batch dimension to create a batch of diagonal matrices
            # For each observation in the batch, the mean of action is given by the corresponding row of batch_mean, and the covariance matrix is given by the corresponding diagonal matrix in scale_tril. This creates a batch of multivariate normal distributions, one for each input observation.
        #action_distribution = distributions.MultivariateNormal(loc=batch_mean, scale_tril=scale_tril) # SHAPE: (batch_dim, ac_dim) for loc / mean and (batch_dim, ac_dim, ac_dim) for scale_tril / covariance matrix

        ## OPTION 2: Using Independent(Normal) to create a diagonal Gaussian distribution
            # The Independent distribution treats the Normal distributions as independent, effectively creating a diagonal Gaussian distribution (all dimensions of action are independent of each other)
        action_distribution = distributions.Independent(
            distributions.Normal(loc=batch_mean, scale=batch_std),
            reinterpreted_batch_ndims=1) # SHAPE: (batch_dim, ac_dim) for loc and scale / std, but the Independent distribution treats the last dimension as independent, resulting in a distribution over (batch_dim,) where each element is a vector of size ac_dim. The reinterpreted_batch_ndims=1 argument tells PyTorch to treat the last dimension of the Normal distribution's output as independent, which is necessary to create a diagonal Gaussian distribution. If we were to omit this argument, PyTorch would treat the entire output of the Normal distribution as a single multivariate distribution, which is not what we want for a diagonal Gaussian.
            # reinterpreted_batch_dim=1 indicates that the last dimension of the Normal distribution's output should be treated as independent, which is necessary to create a diagonal Gaussian distribution.

        return action_distribution

        # *** END CODE HERE ***

    def update(self, observations: np.ndarray, actions: np.ndarray) -> dict:
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss. Recall that to update the policy
        # you need to backpropagate the gradient and step the optimizer.

        # *** START CODE HERE ***
        # 1. Convert `observations` and target expert `actions` from numpy arrays to PyTorch tensors using `ptu.from_numpy()`.
        # 2. Query the policy: invoke `self(observations)` to obtain the diagonal Gaussian distribution.
        # 3. OPTION 1: If using a deterministic policy (e.g. by taking the mean of the distribution), instantiate a Mean Squared Error loss module (`nn.MSELoss()`) and calculate the loss between the predicted actions and the target expert actions.
        #    Draw a differentiable action sample using `rsample()` (enabling gradient propagation). Sampling with reparameterization trick allows gradients to flow through the sampling process, which is necessary for backpropagation.
        #    Now, Instantiate a Mean Squared Error loss module (`nn.MSELoss()`) and calculate the loss between the predicted actions and the target expert actions.
        #
        #    OPTION 2: If using a probabilistic policy, calculate the negative log-likelihood of the target expert actions under the predicted action distribution. 
        #    This can be done by calling `log_prob()` on the action distribution with the target actions as input,
        #    and then taking the mean of the resulting log probabilities across the batch. The loss should be the negative of this mean log probability (since we want to maximize the likelihood, which is equivalent to minimizing the negative log-likelihood).
        # 5. Clear past gradients: `self.optimizer.zero_grad()`.
        # 6. Execute the backward pass: `loss.backward()`.
        # 7. Take an optimization step: `self.optimizer.step()`.
        # 8. Return the training loss inside a dictionary mapping the key `"Training Loss"` to `ptu.to_numpy(loss)`.

        observations = ptu.from_numpy(observations)
        target_actions = ptu.from_numpy(actions)

        action_distributions = self.forward(observations)

        ## Aproach 1: Using MSELoss for deterministic policy - NOT A GOOD OPTION IF THERE IS MULTIMODALITY IN THE ACTION DISTRIBUTION
        #predicted_actions = action_distributions.rsample() # Sampling with reparameterization trick to allow gradients to flow through the sampling process 
            # This gives us a differentiable action sample for each observation in the batch
        #loss = nn.MSELoss(predicted_actions, target_actions)

        ## Approach 2: Using Negative Log-Likelihood for probabilistic policy - WORKS BETTER IF THERE IS MULTIMODALITY IN THE ACTION DISTRIBUTION
        loss = -1 * action_distributions.log_prob(target_actions).mean() # Log probability of the target actions under the predicted action distribution

        self.optimizer.zero_grad() # Clear past gradients
        loss.backward() # Backpropagate the loss
        self.optimizer.step() # Update the policy parameters

        return {"Training Loss": ptu.to_numpy(loss)}

        # *** END CODE HERE ***
