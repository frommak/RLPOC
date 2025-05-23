import torch
import torch.nn as nn
from torchrl.modules.models import MLP # CategoricalHead removed
from torchrl.modules.distributions.discrete import OneHotCategorical # Changed from CategoricalDistribution
from rl_framework.framework.base_policy import BasePolicyNetwork
from rl_framework.framework.base_value import BaseValueNetwork
from typing import Any

class CartPolePolicy(BasePolicyNetwork):
    """
    A policy network for the CartPole environment.
    Uses an MLP to process observations and a CategoricalHead to output
    a distribution over discrete actions.
    """
    def __init__(self, observation_spec: Any, action_spec: Any, hidden_sizes: list[int] = None, device=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64] # Default hidden layer sizes

        num_inputs = observation_spec.shape[-1]
        num_outputs = action_spec.n # Number of discrete actions for logit output

        # The network directly outputs logits for OneHotCategorical
        self.network = MLP(
            in_features=num_inputs,
            out_features=num_outputs, # MLP outputs logits directly
            num_cells=hidden_sizes,   # All specified sizes are hidden layers
            activation_class=nn.ReLU,
            activate_last_layer=False, # Logits usually don't have activation
            device=device
        )

    def forward(self, observation: torch.Tensor) -> OneHotCategorical:
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
            # For CartPole, obs is expected to be a tensor.
            # If it were a TensorDict, one would extract the tensor: e.g., observation.get("obs")
            raise ValueError("CartPolePolicy expects a torch.Tensor observation.")
        
        if observation.ndim == 1: # If a single observation, add batch dimension
            observation = observation.unsqueeze(0)
            
        logits = self.network(observation) # Get logits from the MLP
        
        # Create and return an OneHotCategorical distribution
        return OneHotCategorical(logits=logits)


class CartPoleValue(BaseValueNetwork):
    """
    A value network for the CartPole environment.
    Uses an MLP to process observations and outputs a single state value.
    """
    def __init__(self, observation_spec: Any, hidden_sizes: list[int] = None, device=None): # Added device
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        num_inputs = observation_spec.shape[-1]

        self.mlp = MLP(
            in_features=num_inputs,
            out_features=1, 
            num_cells=hidden_sizes,
            activation_class=nn.ReLU,
            activate_last_layer=False,
            device=device 
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
    device_for_test = "cpu"
    obs_spec = UnboundedContinuousTensorSpec(shape=torch.Size([4]), dtype=torch.float32, device=device_for_test)
    action_spec = DiscreteTensorSpec(n=2, shape=torch.Size([1]), dtype=torch.int64, device=device_for_test)

    # Test Policy Network
    policy_net = CartPolePolicy(observation_spec=obs_spec, action_spec=action_spec, device=device_for_test)
    print("Policy Network:", policy_net)
    dummy_obs = torch.randn(3, 4, device=device_for_test) # Batch of 3 observations
    action_dist = policy_net(dummy_obs)
    print("Action Distribution:", action_dist)
    sampled_action = action_dist.sample()
    print("Sampled Action:", sampled_action)
    print("Log Prob of Sampled Action:", action_dist.log_prob(sampled_action))

    # Test Value Network
    value_net = CartPoleValue(observation_spec=obs_spec, device=device_for_test) # Added device
    print("\nValue Network:", value_net)
    state_values = value_net(dummy_obs)
    print("State Values:", state_values)
    print("State Values Shape:", state_values.shape)

    # Test with single observation
    single_obs = torch.randn(4)
    single_action_dist = policy_net(single_obs.to(device_for_test))
    print("\nSingle Action Distribution:", single_action_dist)
    single_value = value_net(single_obs.to(device_for_test))
    print("Single State Value:", single_value)
