import torch
import torch.nn as nn
from torchrl.modules import MLP, CategoricalHead
from torchrl.modules.distributions import CategoricalDistribution # Ensure this is correctly imported
from rl_framework.framework.base_policy import BasePolicyNetwork
from rl_framework.framework.base_value import BaseValueNetwork
from typing import Any

class CartPolePolicy(BasePolicyNetwork):
    """
    A policy network for the CartPole environment.
    Uses an MLP to process observations and a CategoricalHead to output
    a distribution over discrete actions.
    """
    def __init__(self, observation_spec: Any, action_spec: Any, hidden_sizes: list[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        # Assuming observation_spec.shape is like torch.Size([4]) for CartPole
        # and action_spec.n is 2 for CartPole
        num_inputs = observation_spec.shape[-1]
        num_outputs = action_spec.n # Number of discrete actions

        # Define the MLP body
        self.mlp_body = MLP(
            in_features=num_inputs,
            out_features=hidden_sizes[-1], # Output features of MLP is input to head
            num_cells=hidden_sizes[:-1], # Hidden layers for the MLP body
            activation_class=nn.ReLU,
            activate_last_layer=True # Activation before passing to the head
        )
        
        # Define the head for categorical distribution
        self.distribution_head = CategoricalHead(
            in_features=hidden_sizes[-1], 
            num_outputs=num_outputs
        )

    def forward(self, observation: torch.Tensor) -> CategoricalDistribution:
        """
        Defines the forward pass of the policy network.

        Args:
            observation (torch.Tensor): The input observation from the environment.
                                        Expected shape: [batch_size, num_inputs]

        Returns:
            CategoricalDistribution: A distribution object from TorchRL.
        """
        # Ensure observation is a tensor and 2D (batch_size, features)
        if not isinstance(observation, torch.Tensor):
            # This might happen if observation_spec is complex; for CartPole it's simple
            observation = observation.to_tensordict()["obs"] # Example if it's a TensorDict
        
        if observation.ndim == 1: # If a single observation, add batch dimension
            observation = observation.unsqueeze(0)
            
        features = self.mlp_body(observation)
        action_logits = self.distribution_head(features) # Output of CategoricalHead are logits
        
        # Create the distribution object
        # TorchRL's CategoricalDistribution expects logits
        return CategoricalDistribution(logits=action_logits)


class CartPoleValue(BaseValueNetwork):
    """
    A value network for the CartPole environment.
    Uses an MLP to process observations and outputs a single state value.
    """
    def __init__(self, observation_spec: Any, hidden_sizes: list[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        num_inputs = observation_spec.shape[-1]

        self.mlp = MLP(
            in_features=num_inputs,
            out_features=1, # Output a single value
            num_cells=hidden_sizes,
            activation_class=nn.ReLU,
            activate_last_layer=False # No activation on the final value output
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the value network.

        Args:
            observation (torch.Tensor): The input observation from the environment.
                                        Expected shape: [batch_size, num_inputs]

        Returns:
            torch.Tensor: A tensor representing the state value.
                          Expected shape: [batch_size, 1].
        """
        if not isinstance(observation, torch.Tensor):
             # This might happen if observation_spec is complex; for CartPole it's simple
            observation = observation.to_tensordict()["obs"] # Example if it's a TensorDict

        if observation.ndim == 1: # If a single observation, add batch dimension
            observation = observation.unsqueeze(0)
            
        return self.mlp(observation)


if __name__ == '__main__':
    # Example Usage (requires dummy specs similar to what CartPoleEnvWrapper would provide)
    from torchrl.data import UnboundedContinuousTensorSpec, DiscreteTensorSpec
    
    # Create dummy specs for CartPole
    obs_spec = UnboundedContinuousTensorSpec(shape=torch.Size([4]), dtype=torch.float32)
    action_spec = DiscreteTensorSpec(n=2, shape=torch.Size([1]), dtype=torch.int64)

    # Test Policy Network
    policy_net = CartPolePolicy(observation_spec=obs_spec, action_spec=action_spec)
    print("Policy Network:", policy_net)
    dummy_obs = torch.randn(3, 4) # Batch of 3 observations
    action_dist = policy_net(dummy_obs)
    print("Action Distribution:", action_dist)
    sampled_action = action_dist.sample()
    print("Sampled Action:", sampled_action)
    print("Log Prob of Sampled Action:", action_dist.log_prob(sampled_action))

    # Test Value Network
    value_net = CartPoleValue(observation_spec=obs_spec)
    print("\nValue Network:", value_net)
    state_values = value_net(dummy_obs)
    print("State Values:", state_values)
    print("State Values Shape:", state_values.shape)

    # Test with single observation
    single_obs = torch.randn(4)
    single_action_dist = policy_net(single_obs)
    print("\nSingle Action Distribution:", single_action_dist)
    single_value = value_net(single_obs)
    print("Single State Value:", single_value)
