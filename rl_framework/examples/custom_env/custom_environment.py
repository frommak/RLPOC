import torch
from rl_framework.framework.base_environment import BaseEnvironment
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    TensorDict, # For creating observation and action tensordicts
)
from typing import Any, cast

class CustomComplexEnv(BaseEnvironment):
    """
    A custom environment with a complex (dictionary) observation space
    and a complex (composite) action space.
    """
    def __init__(self, device: str = "cpu", max_steps: int = 100):
        super().__init__()
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.current_step = 0

        # Define observation space components
        self.image_shape = (3, 16, 16)
        self.vector_size = 10
        self._observation_spec = CompositeSpec({
            "image": UnboundedContinuousTensorSpec(shape=self.image_shape, dtype=torch.float32, device=self.device),
            "vector": UnboundedContinuousTensorSpec(shape=(self.vector_size,), dtype=torch.float32, device=self.device)
        }, shape=()) # Pass empty shape for CompositeSpec itself

        # Define action space components
        self.discrete_action_size = 5  # e.g., 5 discrete choices
        self.continuous_action_size = 3
        self._action_spec = CompositeSpec({
            "discrete_action": DiscreteTensorSpec(n=self.discrete_action_size, shape=(1,), dtype=torch.int64, device=self.device),
            "continuous_action": UnboundedContinuousTensorSpec(shape=(self.continuous_action_size,), dtype=torch.float32, device=self.device)
        }, shape=())

        # Initial state for the vector component
        self.vector_state = torch.zeros(self.vector_size, dtype=torch.float32, device=self.device)

    def _generate_observation(self) -> TensorDict:
        # For simplicity, image is random, vector is based on self.vector_state
        # In a real env, these would come from complex dynamics
        img = torch.rand(self.image_shape, dtype=torch.float32, device=self.device)
        
        # Create a TensorDict for the observation
        # Ensure the keys match self._observation_spec
        observation_td = TensorDict({
            "image": img,
            "vector": self.vector_state.clone() 
        }, batch_size=[], device=self.device) # batch_size=[] for single instance
        return observation_td

    def reset(self, seed: int = None) -> TensorDict:
        """
        Resets the environment to an initial state.
        Returns a TensorDict observation.
        """
        if seed is not None:
            torch.manual_seed(seed) # Basic seeding for reproducibility
        
        self.current_step = 0
        self.vector_state = torch.randn(self.vector_size, dtype=torch.float32, device=self.device)
        return self._generate_observation()

    def step(self, action: TensorDict) -> tuple[TensorDict, float, bool, bool, dict]:
        """
        Takes an action (TensorDict) in the environment.
        Returns (obs (TensorDict), reward, terminated, truncated, info).
        """
        if not isinstance(action, TensorDict):
            raise TypeError(f"Action must be a TensorDict, got {type(action)}")

        discrete_act_val = action.get("discrete_action").item() # Get Python value
        continuous_act_val = action.get("continuous_action")

        # Simple dynamics:
        # Vector state changes based on discrete action
        self.vector_state = torch.roll(self.vector_state, shifts=discrete_act_val, dims=0)
        # Image observation changes minimally (or fixed for simplicity)
        
        # Reward function (example)
        reward = (torch.sum(continuous_act_val) - discrete_act_val * 0.1).item()
        
        self.current_step += 1
        terminated = False # No specific termination condition for this example
        truncated = self.current_step >= self.max_steps
        
        next_observation = self._generate_observation()
        info = {}

        return next_observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        """
        Closes the environment and cleans up any resources.
        """
        pass # No specific resources to clean up for this simple env

    def observation_spec(self) -> CompositeSpec:
        return self._observation_spec

    def action_spec(self) -> CompositeSpec:
        return self._action_spec
    
    # For compatibility with TorchRL collector if it needs a callable env factory
    @classmethod
    def from_config(cls, config: dict = None, **kwargs):
        device = kwargs.get("device", "cpu")
        max_steps = kwargs.get("max_steps", 100)
        return cls(device=device, max_steps=max_steps)

if __name__ == '__main__':
    env = CustomComplexEnv(device="cpu")
    print("Observation Spec:", env.observation_spec())
    print("Action Spec:", env.action_spec())

    # Test reset
    initial_obs_td = env.reset(seed=42)
    print("\nInitial Observation (TensorDict):")
    initial_obs_td.info() # Print TensorDict structure and content

    # Test action sampling and step
    # Create a random action TensorDict matching the action_spec
    random_action_td = env.action_spec().rand()
    print("\nRandom Action (TensorDict):")
    random_action_td.info()

    next_obs_td, reward, terminated, truncated, info = env.step(random_action_td)
    
    print("\nNext Observation (TensorDict):")
    next_obs_td.info()
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    env.close()

    # Example of accessing components
    print("\nAccessing components from initial_obs_td:")
    print("Image shape:", initial_obs_td.get("image").shape)
    print("Vector shape:", initial_obs_td.get("vector").shape)

    print("\nAccessing components from random_action_td:")
    print("Discrete action:", random_action_td.get("discrete_action"))
    print("Continuous action shape:", random_action_td.get("continuous_action").shape)

    # Check batch size (should be empty for single instance)
    print("\nBatch size of initial_obs_td:", initial_obs_td.batch_size)
    print("Batch size of random_action_td:", random_action_td.batch_size)
    
    # Check specs again to ensure they are correct
    print("\nRe-checking observation spec device:", env.observation_spec()["image"].device) # Access sub-spec
    print("Re-checking action spec device:", env.action_spec()["discrete_action"].device)   # Access sub-spec

    # Ensure that the observations and actions are indeed TensorDicts
    assert isinstance(initial_obs_td, TensorDict), "Reset should return a TensorDict"
    assert isinstance(random_action_td, TensorDict), "Action spec rand() should produce a TensorDict"
    assert isinstance(next_obs_td, TensorDict), "Step should return a TensorDict observation"

    print("\nAll example checks passed.")
