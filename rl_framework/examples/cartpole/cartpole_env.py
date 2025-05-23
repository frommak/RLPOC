import gymnasium as gym
import torch
from torchrl.data import UnboundedContinuousTensorSpec, DiscreteTensorSpec # Using specific TensorSpec types
from rl_framework.framework.base_environment import BaseEnvironment
from typing import Any

class CartPoleEnvWrapper(BaseEnvironment):
    """
    A wrapper for the Gymnasium CartPole-v1 environment, making it compatible
    with the BaseEnvironment interface.
    """
    def __init__(self, device: str = "cpu", render_mode: str = None):
        super().__init__()
        self.render_mode = render_mode
        # Pass render_mode to gym.make if it's 'human'
        if self.render_mode and self.render_mode == 'human':
            self.env = gym.make('CartPole-v1', render_mode='human')
        else:
            self.env = gym.make('CartPole-v1')
        self.device = torch.device(device)
        self._observation_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([4]), 
            dtype=torch.float32,
            device=self.device
        )
        self._action_spec = DiscreteTensorSpec(
            n=2, 
            shape=torch.Size([1]), 
            dtype=torch.int64, # Typically actions are int64
            device=self.device
        )

    def reset(self, seed: int = None) -> torch.Tensor:
        """
        Resets the environment to an initial state.

        Args:
            seed (int, optional): The seed to use for the environment's random
                number generator. Defaults to None.

        Returns:
            torch.Tensor: The initial observation from the environment, as a tensor.
        """
        if seed is not None:
            # Gymnasium's new API for seeding is to pass it to reset directly.
            observation, info = self.env.reset(seed=seed)
        else:
            observation, info = self.env.reset()
        return torch.tensor(observation, dtype=torch.float32, device=self.device)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        """
        Takes an action in the environment and advances it by one step.

        Args:
            action (torch.Tensor): The action to take in the environment (a tensor).

        Returns:
            tuple[torch.Tensor, float, bool, bool, dict]: A tuple containing:
                - observation (torch.Tensor): The observation from the environment.
                - reward (float): The reward received.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
                - info (dict): Additional information.
        """
        # Gymnasium expects a Python number for discrete actions
        action_np = action.cpu().item() 
        observation, reward, terminated, truncated, info = self.env.step(action_np)
        return (
            torch.tensor(observation, dtype=torch.float32, device=self.device),
            float(reward), # Ensure reward is float
            bool(terminated),
            bool(truncated),
            info
        )

    def close(self) -> None:
        """
        Closes the environment and cleans up any resources.
        """
        self.env.close()

    def observation_spec(self) -> UnboundedContinuousTensorSpec:
        """
        Returns the specification of the observation space.
        """
        return self._observation_spec

    def action_spec(self) -> DiscreteTensorSpec:
        """
        Returns the specification of the action space.
        """
        return self._action_spec

    def render(self):
        """Calls the underlying environment's render method."""
        if self.render_mode == 'human':
            return self.env.render()
        # For other modes, one might return env.render(mode='rgb_array') etc.
        # but BaseEnvironment doesn't specify a return type for render.
        return None

    # For compatibility with TorchRL collector if it needs a callable env factory
    @classmethod
    def from_config(cls, config: dict = None, **kwargs):
        # config might contain device settings, etc.
        device = kwargs.get("device", "cpu")
        render_mode = kwargs.get("render_mode", None) # Get render_mode from kwargs
        return cls(device=device, render_mode=render_mode)

if __name__ == '__main__':
    # Example usage:
    env = CartPoleEnvWrapper(device="cpu")
    print("Observation Spec:", env.observation_spec())
    print("Action Spec:", env.action_spec())

    obs = env.reset(seed=42)
    print("Initial observation:", obs)

    action = env.action_spec().rand() # Get a random action
    print("Taking action:", action)
    # env.render() # Example of rendering, call this in a loop to see the animation
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    print("Next observation:", next_obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    
    env.close()
